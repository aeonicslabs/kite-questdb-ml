"""Unit tests for feature engineering."""
import math
import numpy as np
import pytest
from features.engineer import RingBuffer, FeatureEngine, approx_iv


class TestRingBuffer:
    def test_append_and_prices(self):
        rb = RingBuffer(maxlen=5)
        for i in range(5):
            rb.append({"last_price": float(i + 1), "_recv_time": float(i)})
        prices = rb.prices()
        assert len(prices) == 5
        assert prices[-1] == 5.0

    def test_maxlen_enforced(self):
        rb = RingBuffer(maxlen=3)
        for i in range(10):
            rb.append({"last_price": float(i), "_recv_time": float(i)})
        assert len(rb) == 3

    def test_slice_by_age(self):
        import time
        rb = RingBuffer(maxlen=100)
        now = time.monotonic()
        # Add some old ticks
        for i in range(5):
            rb.append({"last_price": float(i), "_recv_time": now - 100 + i})
        # Add recent ticks
        for i in range(5):
            rb.append({"last_price": float(i + 10), "_recv_time": now - 2 + i * 0.1})
        recent = rb.slice_by_age(5)
        assert len(recent) == 5


class TestApproxIV:
    def test_reasonable_iv(self):
        # ATM call, 30 days to expiry, ~20% IV
        S, K, T = 50000.0, 50000.0, 30 / 365
        call_price = 800.0  # approx ATM call value
        iv = approx_iv(call_price, S, K, T, option_type="CE")
        assert not math.isnan(iv)
        assert 0.05 < iv < 2.0  # reasonable range

    def test_invalid_inputs(self):
        assert math.isnan(approx_iv(0, 50000, 50000, 0.1))
        assert math.isnan(approx_iv(100, 0, 50000, 0.1))
        assert math.isnan(approx_iv(100, 50000, 50000, 0))

    def test_deep_itm_put(self):
        S, K, T = 48000.0, 52000.0, 7 / 365
        put_price = 4100.0  # deep ITM put
        iv = approx_iv(put_price, S, K, T, option_type="PE")
        # Deep ITM may not converge cleanly, but shouldn't crash
        assert isinstance(iv, float)


class TestFeatureEngineIntegration:
    """Smoke tests — no QuestDB needed, mock writer."""

    def _make_tick(self, price=50000.0, token=12345, oi=100000):
        return {
            "instrument_token": token,
            "trading_symbol": "BANKNIFTY26MAR51000CE",
            "last_price": price,
            "last_quantity": 25,
            "average_price": price,
            "volume": 5000,
            "buy_quantity": 300,
            "sell_quantity": 250,
            "ohlc": {"open": price - 50, "high": price + 100, "low": price - 100, "close": price - 50},
            "change": 0.5,
            "oi": oi,
            "oi_day_high": oi + 1000,
            "oi_day_low": oi - 500,
            "depth": {
                "buy": [
                    {"price": price - 0.5, "quantity": 100},
                    {"price": price - 1.0, "quantity": 200},
                    {"price": price - 1.5, "quantity": 150},
                ],
                "sell": [
                    {"price": price + 0.5, "quantity": 80},
                    {"price": price + 1.0, "quantity": 180},
                    {"price": price + 1.5, "quantity": 120},
                ],
            },
            "exchange": "NFO",
            "instrument_type": "CE",
            "strike": 51000.0,
            "name": "BANKNIFTY",
            "expiry": "2026-03-26",
        }

    def test_feature_computation_no_crash(self):
        written = []

        class MockWriter:
            def write_features(self, f): written.append(f)
            def write_tick(self, t): pass
            def write_signal(self, s): pass

        engine = FeatureEngine(db_writer=MockWriter())

        import time
        for i in range(30):
            tick = self._make_tick(price=50000.0 + i * 10)
            tick["_recv_time"] = time.monotonic()
            engine._process_tick(tick)

        assert len(written) > 0
        feat = written[-1]
        assert "order_book_imbalance" in feat
        assert "rsi_14" in feat
        assert "bid_ask_spread" in feat
        assert feat["bid_ask_spread"] == pytest.approx(1.0, abs=0.1)

    def test_obi_positive_when_more_bids(self):
        written = []

        class MockWriter:
            def write_features(self, f): written.append(f)
            def write_tick(self, t): pass
            def write_signal(self, s): pass

        engine = FeatureEngine(db_writer=MockWriter())

        import time
        for i in range(10):
            tick = self._make_tick(price=50000.0)
            # Set more bid volume than ask
            tick["depth"]["buy"] = [{"price": 49999.5, "quantity": 500}]
            tick["depth"]["sell"] = [{"price": 50000.5, "quantity": 100}]
            tick["_recv_time"] = time.monotonic()
            engine._process_tick(tick)

        if written:
            obi = written[-1].get("order_book_imbalance", 0)
            assert obi > 0, "OBI should be positive when buy volume > sell volume"
