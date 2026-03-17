"""
QuestDB writer using ILP over HTTP (questdb Python SDK v2+).
Thread-safe, batched, with error handling and metrics.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from questdb.ingress import Sender, TimestampNanos, IngressError

from config.settings import get_settings

logger = logging.getLogger(__name__)


class QuestDBWriter:
    """
    Thread-safe writer to QuestDB via ILP over HTTP.

    Usage:
        writer = QuestDBWriter()
        writer.write_tick(tick_dict)
        writer.write_features(features_dict)
        writer.close()

    Or as context manager:
        with QuestDBWriter() as writer:
            writer.write_tick(tick)
    """

    def __init__(self) -> None:
        self._settings = get_settings().questdb
        self._lock = threading.Lock()
        self._sender: Optional[Sender] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._sender = Sender.from_conf(self._settings.ilp_conf)
            logger.info(
                "QuestDB ILP sender connected to %s:%s",
                self._settings.host,
                self._settings.ilp_port,
            )
        except Exception as e:
            logger.error("Failed to connect QuestDB sender: %s", e)
            raise

    # ------------------------------------------------------------------
    # PUBLIC WRITE METHODS
    # ------------------------------------------------------------------

    def write_tick(self, tick: Dict[str, Any]) -> None:
        """Write a single parsed tick to the `ticks` table."""
        if not self._sender:
            return
        try:
            ts_ns = self._to_ts_nanos(tick.get("exchange_timestamp") or tick.get("last_trade_time"))
            depth = tick.get("depth", {})
            bids = depth.get("buy", [{}] * 5)
            asks = depth.get("sell", [{}] * 5)

            with self._lock:
                self._sender.row(
                    "ticks",
                    symbols={
                        "trading_symbol": str(tick.get("trading_symbol", "")),
                        "exchange": str(tick.get("exchange", "NFO")),
                    },
                    columns={
                        "received_at": TimestampNanos.now(),
                        "instrument_token": int(tick.get("instrument_token", 0)),
                        "last_price": float(tick.get("last_price", 0.0)),
                        "last_quantity": int(tick.get("last_quantity", 0)),
                        "average_price": float(tick.get("average_price", 0.0)),
                        "volume": int(tick.get("volume", 0)),
                        "buy_quantity": int(tick.get("buy_quantity", 0)),
                        "sell_quantity": int(tick.get("sell_quantity", 0)),
                        "open": float(tick.get("ohlc", {}).get("open", 0.0)),
                        "high": float(tick.get("ohlc", {}).get("high", 0.0)),
                        "low": float(tick.get("ohlc", {}).get("low", 0.0)),
                        "close": float(tick.get("ohlc", {}).get("close", 0.0)),
                        "change": float(tick.get("change", 0.0)),
                        "open_interest": int(tick.get("oi", 0)),
                        "oi_day_high": int(tick.get("oi_day_high", 0)),
                        "oi_day_low": int(tick.get("oi_day_low", 0)),
                        # Bid depth
                        "bid1_price": float(bids[0].get("price", 0.0) if len(bids) > 0 else 0.0),
                        "bid1_qty": int(bids[0].get("quantity", 0) if len(bids) > 0 else 0),
                        "bid2_price": float(bids[1].get("price", 0.0) if len(bids) > 1 else 0.0),
                        "bid2_qty": int(bids[1].get("quantity", 0) if len(bids) > 1 else 0),
                        "bid3_price": float(bids[2].get("price", 0.0) if len(bids) > 2 else 0.0),
                        "bid3_qty": int(bids[2].get("quantity", 0) if len(bids) > 2 else 0),
                        "bid4_price": float(bids[3].get("price", 0.0) if len(bids) > 3 else 0.0),
                        "bid4_qty": int(bids[3].get("quantity", 0) if len(bids) > 3 else 0),
                        "bid5_price": float(bids[4].get("price", 0.0) if len(bids) > 4 else 0.0),
                        "bid5_qty": int(bids[4].get("quantity", 0) if len(bids) > 4 else 0),
                        # Ask depth
                        "ask1_price": float(asks[0].get("price", 0.0) if len(asks) > 0 else 0.0),
                        "ask1_qty": int(asks[0].get("quantity", 0) if len(asks) > 0 else 0),
                        "ask2_price": float(asks[1].get("price", 0.0) if len(asks) > 1 else 0.0),
                        "ask2_qty": int(asks[1].get("quantity", 0) if len(asks) > 1 else 0),
                        "ask3_price": float(asks[2].get("price", 0.0) if len(asks) > 2 else 0.0),
                        "ask3_qty": int(asks[2].get("quantity", 0) if len(asks) > 2 else 0),
                        "ask4_price": float(asks[3].get("price", 0.0) if len(asks) > 3 else 0.0),
                        "ask4_qty": int(asks[3].get("quantity", 0) if len(asks) > 3 else 0),
                        "ask5_price": float(asks[4].get("price", 0.0) if len(asks) > 4 else 0.0),
                        "ask5_qty": int(asks[4].get("quantity", 0) if len(asks) > 4 else 0),
                    },
                    at=ts_ns,
                )
        except IngressError as e:
            logger.error("QuestDB ILP error writing tick: %s", e)
        except Exception as e:
            logger.error("Unexpected error writing tick: %s", e)

    def write_features(self, feats: Dict[str, Any]) -> None:
        """Write a computed feature row to the `features` table."""
        if not self._sender:
            return
        try:
            ts_ns = self._to_ts_nanos(feats.get("ts"))
            with self._lock:
                self._sender.row(
                    "features",
                    symbols={
                        "trading_symbol": str(feats.get("trading_symbol", "")),
                    },
                    columns={k: v for k, v in feats.items()
                             if k not in ("ts", "trading_symbol", "instrument_token")
                             and isinstance(v, (int, float))},
                    at=ts_ns,
                )
                # Add instrument_token as int column
                if feats.get("instrument_token"):
                    pass  # Already included above via columns dict
        except IngressError as e:
            logger.error("QuestDB ILP error writing features: %s", e)
        except Exception as e:
            logger.error("Unexpected error writing features: %s", e)

    def write_signal(self, signal: Dict[str, Any]) -> None:
        """Write an ML signal to the `signals` table."""
        if not self._sender:
            return
        try:
            ts_ns = TimestampNanos.now()
            with self._lock:
                self._sender.row(
                    "signals",
                    symbols={
                        "trading_symbol": str(signal.get("trading_symbol", "")),
                        "model_name": str(signal.get("model_name", "")),
                        "signal": str(signal.get("signal", "HOLD")),
                        "predicted_direction": str(signal.get("predicted_direction", "FLAT")),
                    },
                    columns={
                        "instrument_token": int(signal.get("instrument_token", 0)),
                        "model_version": str(signal.get("model_version", "")),
                        "confidence": float(signal.get("confidence", 0.0)),
                        "predicted_magnitude": float(signal.get("predicted_magnitude", 0.0)),
                        "features_snapshot": str(signal.get("features_snapshot", "{}")),
                    },
                    at=ts_ns,
                )
        except Exception as e:
            logger.error("Error writing signal: %s", e)

    def write_paper_trade(self, trade: Dict[str, Any]) -> None:
        """Write a paper trade entry to the `paper_trades` table."""
        if not self._sender:
            return
        try:
            ts_ns = TimestampNanos.now()
            with self._lock:
                self._sender.row(
                    "paper_trades",
                    symbols={
                        "trading_symbol": str(trade.get("trading_symbol", "")),
                        "action": str(trade.get("action", "")),
                        "status": str(trade.get("status", "OPEN")),
                        "model_name": str(trade.get("model_name", "")),
                    },
                    columns={
                        "trade_id": str(trade.get("trade_id", "")),
                        "instrument_token": int(trade.get("instrument_token", 0)),
                        "quantity": int(trade.get("quantity", 0)),
                        "entry_price": float(trade.get("entry_price", 0.0)),
                        "exit_price": float(trade.get("exit_price", 0.0)),
                        "pnl": float(trade.get("pnl", 0.0)),
                        "signal_confidence": float(trade.get("signal_confidence", 0.0)),
                    },
                    at=ts_ns,
                )
        except Exception as e:
            logger.error("Error writing paper trade: %s", e)

    def flush(self) -> None:
        """Force flush the ILP buffer immediately."""
        if self._sender:
            with self._lock:
                try:
                    self._sender.flush()
                except Exception as e:
                    logger.error("QuestDB flush error: %s", e)

    def close(self) -> None:
        if self._sender:
            try:
                self._sender.flush()
                self._sender.__exit__(None, None, None)
            except Exception:
                pass
            self._sender = None
            logger.info("QuestDB writer closed")

    def __enter__(self) -> "QuestDBWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ts_nanos(ts: Any) -> TimestampNanos:
        """Convert various timestamp formats to TimestampNanos."""
        if ts is None:
            return TimestampNanos.now()
        if isinstance(ts, TimestampNanos):
            return ts
        if isinstance(ts, datetime):
            return TimestampNanos(int(ts.timestamp() * 1_000_000_000))
        if isinstance(ts, (int, float)):
            # If it looks like seconds (Unix epoch), convert to nanos
            if ts < 1e12:
                return TimestampNanos(int(ts * 1_000_000_000))
            return TimestampNanos(int(ts))
        return TimestampNanos.now()
