"""
Feature Engineering Engine.

Consumes tick batches from a queue and produces feature rows
written to QuestDB's `features` table.

Features computed (all vectorized with numpy):
  Order book:
    - bid_ask_spread, mid_price
    - order_book_imbalance (L1 and L3)
  Price dynamics:
    - price_change_1s, 5s, 30s
    - vwap_deviation
    - RSI(14), Bollinger Bands(20)
  Volume:
    - volume_rate (per second in rolling 30s window)
    - buy_sell_ratio
  OI (options):
    - oi_change_rate
    - pcr_oi (put/call OI ratio across index options)
  Options:
    - iv_approx (Newton-Raphson on Black-Scholes)
    - atm_distance_pct
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from storage.questdb_writer import QuestDBWriter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# RING BUFFER — fixed-size, fast append, numpy slice
# ------------------------------------------------------------------

class RingBuffer:
    """Fixed-size circular buffer for time-series data per instrument."""

    def __init__(self, maxlen: int = 600) -> None:
        self._maxlen = maxlen
        self._buf: Deque[Dict] = deque(maxlen=maxlen)

    def append(self, item: Dict) -> None:
        self._buf.append(item)

    def __len__(self) -> int:
        return len(self._buf)

    def prices(self) -> np.ndarray:
        return np.array([x["last_price"] for x in self._buf], dtype=np.float64)

    def volumes(self) -> np.ndarray:
        return np.array([x.get("volume", 0) for x in self._buf], dtype=np.float64)

    def oi(self) -> np.ndarray:
        return np.array([x.get("oi", 0) for x in self._buf], dtype=np.float64)

    def buy_qty(self) -> np.ndarray:
        return np.array([x.get("buy_quantity", 0) for x in self._buf], dtype=np.float64)

    def sell_qty(self) -> np.ndarray:
        return np.array([x.get("sell_quantity", 0) for x in self._buf], dtype=np.float64)

    def timestamps(self) -> np.ndarray:
        return np.array([x.get("_recv_time", 0.0) for x in self._buf], dtype=np.float64)

    def last(self) -> Optional[Dict]:
        return self._buf[-1] if self._buf else None

    def slice_by_age(self, max_age_s: float) -> "RingBuffer":
        """Return a view of ticks within the last max_age_s seconds."""
        now = time.monotonic()
        rb = RingBuffer(self._maxlen)
        for item in self._buf:
            if now - item.get("_recv_time", now) <= max_age_s:
                rb.append(item)
        return rb


# ------------------------------------------------------------------
# BLACK-SCHOLES APPROXIMATE IV (Newton-Raphson)
# ------------------------------------------------------------------

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2))) / 2.0


def approx_iv(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.065,
    option_type: str = "CE",
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Approximate implied volatility using Newton-Raphson.
    Returns NaN if unable to converge or inputs invalid.
    """
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    intrinsic = max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
    if option_price < intrinsic:
        return float("nan")

    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        try:
            price = black_scholes_call(S, K, T, r, sigma)
            # Vega
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            vega = S * _norm_cdf(d1) * math.sqrt(T)
            if vega < 1e-10:
                break
            diff = price - option_price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5.0))
        except (ValueError, ZeroDivisionError, OverflowError):
            break
    return float("nan")


# ------------------------------------------------------------------
# FEATURE ENGINE
# ------------------------------------------------------------------

class FeatureEngine:
    """
    Stateful feature computation engine.

    Maintains per-instrument RingBuffers.
    On each tick batch, updates buffers and emits a feature row to QuestDB.

    Usage:
        engine = FeatureEngine(db_writer)
        await engine.run(feature_queue)  # blocks until cancelled
    """

    def __init__(
        self,
        db_writer: QuestDBWriter,
        on_features: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self._db_writer = db_writer
        self._on_features = on_features  # optional callback for ML inference
        self._settings = get_settings()

        # Per-instrument state
        self._buffers: Dict[int, RingBuffer] = defaultdict(lambda: RingBuffer(maxlen=600))
        # OI tracking for PCR
        self._oi_by_symbol: Dict[str, float] = {}

        self._features_computed = 0

    async def run(self, feature_queue: "asyncio.Queue[List[Dict]]") -> None:
        """Drain feature_queue and compute features for each tick."""
        while True:
            try:
                ticks = await feature_queue.get()
                for tick in ticks:
                    self._process_tick(tick)
                feature_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("FeatureEngine error: %s", e)

    def _process_tick(self, tick: Dict) -> None:
        token = tick.get("instrument_token")
        if not token:
            return

        # Stamp receive time for windowing
        tick["_recv_time"] = time.monotonic()

        buf = self._buffers[token]
        buf.append(tick)

        # Track OI for PCR
        sym = tick.get("trading_symbol", "")
        oi = tick.get("oi", 0)
        if oi:
            self._oi_by_symbol[sym] = oi

        # Only compute features if we have enough history
        if len(buf) < 5:
            return

        try:
            feats = self._compute_features(tick, buf)
            if feats:
                self._db_writer.write_features(feats)
                self._features_computed += 1
                if self._on_features:
                    self._on_features(feats)
        except Exception as e:
            logger.debug("Feature computation error for %s: %s", token, e)

    def _compute_features(self, tick: Dict, buf: RingBuffer) -> Optional[Dict]:
        token = int(tick.get("instrument_token", 0))
        symbol = tick.get("trading_symbol", "")
        last_price = float(tick.get("last_price", 0.0) or 0.0)

        if last_price <= 0:
            return None

        # Depth
        depth = tick.get("depth", {})
        bids = depth.get("buy", [])
        asks = depth.get("sell", [])

        bid1_p = float(bids[0].get("price", 0.0)) if bids else 0.0
        ask1_p = float(asks[0].get("price", 0.0)) if asks else 0.0
        bid1_q = float(bids[0].get("quantity", 0)) if bids else 0.0
        ask1_q = float(asks[0].get("quantity", 0)) if asks else 0.0

        mid_price = (bid1_p + ask1_p) / 2 if bid1_p and ask1_p else last_price
        bid_ask_spread = ask1_p - bid1_p if bid1_p and ask1_p else 0.0

        # OBI L1
        total_l1 = bid1_q + ask1_q
        obi_l1 = (bid1_q - ask1_q) / total_l1 if total_l1 > 0 else 0.0

        # OBI L3 (top 3 levels)
        bid_vol_3 = sum(float(b.get("quantity", 0)) for b in bids[:3])
        ask_vol_3 = sum(float(a.get("quantity", 0)) for a in asks[:3])
        total_l3 = bid_vol_3 + ask_vol_3
        obi_l3 = (bid_vol_3 - ask_vol_3) / total_l3 if total_l3 > 0 else 0.0

        # Price windows
        prices = buf.prices()
        now_m = time.monotonic()
        timestamps = buf.timestamps()

        def price_n_seconds_ago(n: float) -> Optional[float]:
            cutoff = now_m - n
            mask = timestamps <= cutoff
            if mask.any():
                return float(prices[mask][-1])
            return None

        p_1s = price_n_seconds_ago(1)
        p_5s = price_n_seconds_ago(5)
        p_30s = price_n_seconds_ago(30)

        price_change_1s = (last_price - p_1s) / p_1s if p_1s else 0.0
        price_change_5s = (last_price - p_5s) / p_5s if p_5s else 0.0
        price_change_30s = (last_price - p_30s) / p_30s if p_30s else 0.0

        # VWAP
        vols = buf.volumes()
        if len(prices) > 1 and vols[-1] > 0:
            price_vol = prices * (np.diff(vols, prepend=vols[0]))
            total_vol = np.diff(vols, prepend=vols[0]).sum()
            vwap = price_vol.sum() / total_vol if total_vol > 0 else last_price
        else:
            vwap = last_price
        vwap_deviation = (last_price - vwap) / vwap if vwap > 0 else 0.0

        # Volume rate (per second in last 30s)
        win_30 = buf.slice_by_age(30)
        if len(win_30) >= 2:
            vols_30 = win_30.volumes()
            vol_delta = float(vols_30[-1] - vols_30[0])
            volume_rate = vol_delta / 30.0
        else:
            volume_rate = 0.0

        # Buy/sell ratio
        buy_q = float(tick.get("buy_quantity", 0) or 0)
        sell_q = float(tick.get("sell_quantity", 0) or 0)
        buy_sell_ratio = buy_q / sell_q if sell_q > 0 else (1.0 if buy_q > 0 else 0.0)

        # RSI(14)
        rsi = self._compute_rsi(prices, period=14)

        # Bollinger Bands(20, 2)
        bb_upper, bb_lower, bb_width = self._compute_bb(prices, period=20, std_mult=2.0)

        # OI change rate
        oi_arr = buf.oi()
        if len(oi_arr) >= 2 and oi_arr[-1] > 0:
            oi_delta = float(oi_arr[-1] - oi_arr[0])
            ts_delta = float(timestamps[-1] - timestamps[0]) if timestamps[-1] != timestamps[0] else 1.0
            oi_change_rate = oi_delta / ts_delta if ts_delta > 0 else 0.0
        else:
            oi_change_rate = 0.0

        # 30s range
        if len(win_30) >= 2:
            p30 = win_30.prices()
            high_30s = float(p30.max())
            low_30s = float(p30.min())
            range_30s = high_30s - low_30s
        else:
            high_30s = last_price
            low_30s = last_price
            range_30s = 0.0

        # IV approximation (options only)
        iv = float("nan")
        atm_dist = float("nan")
        instrument_type = tick.get("instrument_type", "")
        if instrument_type in ("CE", "PE"):
            strike = float(tick.get("strike", 0.0) or 0.0)
            # Find underlying spot
            name = tick.get("name", "")
            spot_map = {"BANKNIFTY": "BANKNIFTY", "NIFTY": "NIFTY"}
            spot = None
            for k, v in spot_map.items():
                if k in name:
                    spot = self._get_spot(k)
                    break

            if spot and spot > 0 and strike > 0:
                expiry_str = tick.get("expiry", "")
                try:
                    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    T = (expiry_date - datetime.now(timezone.utc).date()).days / 365.0
                    if T > 0:
                        iv = approx_iv(last_price, spot, strike, T, option_type=instrument_type)
                except Exception:
                    pass
                atm_dist = abs(last_price - strike) / strike if strike > 0 else float("nan")

        # PCR
        pcr = self._compute_pcr(tick.get("name", ""))

        return {
            "ts": datetime.now(timezone.utc),
            "instrument_token": token,
            "trading_symbol": symbol,
            "bid_ask_spread": bid_ask_spread,
            "mid_price": mid_price,
            "order_book_imbalance": obi_l1,
            "depth_imbalance_l3": obi_l3,
            "price_change_1s": price_change_1s,
            "price_change_5s": price_change_5s,
            "price_change_30s": price_change_30s,
            "vwap_deviation": vwap_deviation,
            "volume_rate": volume_rate,
            "buy_sell_ratio": buy_sell_ratio,
            "oi_change_rate": oi_change_rate,
            "high_30s": high_30s,
            "low_30s": low_30s,
            "range_30s": range_30s,
            "rsi_14": rsi,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width": bb_width,
            "iv_approx": iv if not math.isnan(iv) else 0.0,
            "pcr_oi": pcr,
            "atm_distance_pct": atm_dist if not math.isnan(atm_dist) else 0.0,
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _get_spot(self, underlying: str) -> Optional[float]:
        """Get current spot price from OI tracker (updated by ingestion layer)."""
        for sym, oi in self._oi_by_symbol.items():
            if underlying in sym and "CE" not in sym and "PE" not in sym:
                return oi  # fallback
        return None

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _compute_bb(
        self, prices: np.ndarray, period: int = 20, std_mult: float = 2.0
    ) -> Tuple[float, float, float]:
        if len(prices) < period:
            p = prices[-1] if len(prices) > 0 else 0.0
            return p, p, 0.0
        window = prices[-period:]
        mean = window.mean()
        std = window.std()
        upper = float(mean + std_mult * std)
        lower = float(mean - std_mult * std)
        width = float((upper - lower) / mean) if mean > 0 else 0.0
        return upper, lower, width

    def _compute_pcr(self, name: str) -> float:
        """Put/Call ratio by OI for the given underlying."""
        put_oi = sum(v for k, v in self._oi_by_symbol.items()
                     if name in k and "PE" in k)
        call_oi = sum(v for k, v in self._oi_by_symbol.items()
                      if name in k and "CE" in k)
        return put_oi / call_oi if call_oi > 0 else 1.0
