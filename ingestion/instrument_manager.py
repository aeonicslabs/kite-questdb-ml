"""
Instrument Manager — resolves BankNifty/Nifty option chains dynamically.

Responsibilities:
  - Fetch the full NSE F&O instruments dump from Kite on startup
  - Find ATM strike for NIFTY and BANKNIFTY
  - Build the list of instrument_tokens to subscribe (ATM ± N strikes, current + next expiry)
  - Cache instrument metadata for tick enrichment
  - Refresh daily before market open
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import get_settings

logger = logging.getLogger(__name__)


class InstrumentManager:
    """
    Manages the instrument universe for BankNifty and Nifty options.

    Usage:
        mgr = InstrumentManager(kite)
        mgr.refresh()
        tokens = mgr.get_subscription_tokens()
        meta = mgr.get_instrument_meta(token)
    """

    # Underlying names as they appear in Kite instruments dump
    UNDERLYINGS = {
        "BANKNIFTY": {"exchange": "NFO", "index_symbol": "NSE:NIFTY BANK"},
        "NIFTY": {"exchange": "NFO", "index_symbol": "NSE:NIFTY 50"},
    }

    def __init__(self, kite: KiteConnect) -> None:
        self._kite = kite
        self._settings = get_settings()
        self._instruments_df: Optional[pd.DataFrame] = None
        # token → metadata dict
        self._meta: Dict[int, Dict] = {}
        # token → trading symbol (for enriching ticks)
        self._token_to_symbol: Dict[int, str] = {}
        # Current spot prices (updated from index ticks)
        self._spot_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Fetch instruments dump and rebuild token universe."""
        logger.info("Fetching instruments dump from Kite...")
        raw = self._kite.instruments("NFO")
        self._instruments_df = pd.DataFrame(raw)
        self._instruments_df["expiry"] = pd.to_datetime(
            self._instruments_df["expiry"], errors="coerce"
        )
        logger.info(
            "Loaded %d NFO instruments", len(self._instruments_df)
        )
        self._build_meta()

    def update_spot(self, underlying: str, price: float) -> None:
        """Call this when you receive an index tick to update ATM calculation."""
        self._spot_prices[underlying] = price

    def get_subscription_tokens(self) -> List[int]:
        """
        Return list of instrument_tokens to subscribe via WebSocket.
        Includes: index tokens + all option tokens in universe.
        """
        tokens = list(self._meta.keys())
        # Add index tokens
        for underlying, info in self.UNDERLYINGS.items():
            sym = info["index_symbol"]
            try:
                quote = self._kite.ltp([sym])
                token = list(quote.values())[0]["instrument_token"]
                tokens.append(token)
                self._token_to_symbol[token] = sym
                self._spot_prices[underlying] = list(quote.values())[0]["last_price"]
            except Exception as e:
                logger.warning("Could not fetch index token for %s: %s", sym, e)
        return list(set(tokens))

    def get_instrument_meta(self, token: int) -> Optional[Dict]:
        """Lookup instrument metadata by token."""
        return self._meta.get(token)

    def get_token_symbol(self, token: int) -> Optional[str]:
        return self._token_to_symbol.get(token)

    def get_all_meta(self) -> Dict[int, Dict]:
        return self._meta.copy()

    def get_atm_tokens(self, underlying: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Return (ATM_CE_token, ATM_PE_token) for the nearest expiry.
        Returns (None, None) if spot price not yet known.
        """
        spot = self._spot_prices.get(underlying)
        if spot is None:
            return None, None

        df = self._instruments_df
        if df is None:
            return None, None

        und_df = df[df["name"] == underlying].copy()
        if und_df.empty:
            return None, None

        # Nearest weekly expiry
        today = date.today()
        future_expiries = sorted(
            [e for e in und_df["expiry"].dropna().unique() if e.date() >= today]
        )
        if not future_expiries:
            return None, None

        near_expiry = future_expiries[0]
        expiry_df = und_df[und_df["expiry"] == near_expiry]

        # ATM strike = nearest available strike to spot
        strikes = sorted(expiry_df["strike"].unique())
        atm_strike = min(strikes, key=lambda s: abs(s - spot))

        ce = expiry_df[
            (expiry_df["strike"] == atm_strike) & (expiry_df["instrument_type"] == "CE")
        ]
        pe = expiry_df[
            (expiry_df["strike"] == atm_strike) & (expiry_df["instrument_type"] == "PE")
        ]

        ce_token = int(ce["instrument_token"].iloc[0]) if not ce.empty else None
        pe_token = int(pe["instrument_token"].iloc[0]) if not pe.empty else None
        return ce_token, pe_token

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _build_meta(self) -> None:
        """Build option universe for BANKNIFTY and NIFTY."""
        if self._instruments_df is None:
            return

        cfg = self._settings.kite
        strikes_range = cfg.option_strikes_range
        expiry_count = cfg.option_expiries
        df = self._instruments_df

        self._meta = {}
        self._token_to_symbol = {}

        today = date.today()

        for underlying in self.UNDERLYINGS:
            und_df = df[
                (df["name"] == underlying) &
                (df["instrument_type"].isin(["CE", "PE"]))
            ].copy()

            if und_df.empty:
                logger.warning("No options found for %s", underlying)
                continue

            # Get upcoming expiries
            future_expiries = sorted(
                [e for e in und_df["expiry"].dropna().unique() if e.date() >= today]
            )[:expiry_count]

            for expiry in future_expiries:
                expiry_df = und_df[und_df["expiry"] == expiry]
                strikes = sorted(expiry_df["strike"].unique())

                # We don't know spot yet at startup — subscribe all if no spot,
                # else limit to ATM ± strikes_range
                spot = self._spot_prices.get(underlying)
                if spot:
                    atm = min(strikes, key=lambda s: abs(s - spot))
                    atm_idx = strikes.index(atm)
                    lo = max(0, atm_idx - strikes_range)
                    hi = min(len(strikes), atm_idx + strikes_range + 1)
                    selected_strikes = strikes[lo:hi]
                else:
                    # Without spot: subscribe a limited range at startup
                    # This will be dynamically narrowed on first index tick
                    mid = len(strikes) // 2
                    lo = max(0, mid - strikes_range)
                    hi = min(len(strikes), mid + strikes_range + 1)
                    selected_strikes = strikes[lo:hi]

                for _, row in expiry_df[
                    expiry_df["strike"].isin(selected_strikes)
                ].iterrows():
                    token = int(row["instrument_token"])
                    self._meta[token] = {
                        "instrument_token": token,
                        "trading_symbol": row["tradingsymbol"],
                        "name": row["name"],
                        "expiry": str(row["expiry"].date()),
                        "strike": float(row["strike"]),
                        "instrument_type": row["instrument_type"],
                        "exchange": "NFO",
                        "lot_size": int(row.get("lot_size", 1)),
                        "tick_size": float(row.get("tick_size", 0.05)),
                    }
                    self._token_to_symbol[token] = row["tradingsymbol"]

            logger.info(
                "%s: subscribed %d option tokens across %d expiries",
                underlying, len([t for t, m in self._meta.items()
                                  if m["name"] == underlying]), len(future_expiries),
            )
