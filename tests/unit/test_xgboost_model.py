"""Unit tests for XGBoost direction model."""
import numpy as np
import pandas as pd
import pytest
from ml.models.xgboost_model import XGBoostDirectionModel, FEATURE_COLS


def make_dummy_df(n_rows=6000, seed=42):
    rng = np.random.RandomState(seed)
    data = {col: rng.randn(n_rows) for col in FEATURE_COLS}
    # Realistic ranges
    data["rsi_14"] = rng.uniform(20, 80, n_rows)
    data["bid_ask_spread"] = rng.uniform(0.5, 5.0, n_rows)
    data["order_book_imbalance"] = rng.uniform(-1, 1, n_rows)
    # Labels: slightly correlated with OBI
    data["label"] = (data["order_book_imbalance"] + rng.randn(n_rows) * 0.5 > 0).astype(int)
    return pd.DataFrame(data)


class TestXGBoostDirectionModel:
    def test_train_and_predict(self):
        model = XGBoostDirectionModel()
        df = make_dummy_df(6000)
        metrics = model.train(df)
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert metrics["auc_roc"] > 0.4  # above random for correlated data

    def test_predict_returns_valid_signal(self):
        model = XGBoostDirectionModel()
        df = make_dummy_df(6000)
        model.train(df)

        feat = {col: float(np.random.randn()) for col in FEATURE_COLS}
        signal, conf = model.predict(feat)
        assert signal in ("BUY_CE", "BUY_PE", "HOLD")
        assert 0.0 <= conf <= 1.0

    def test_predict_without_training_returns_hold(self):
        model = XGBoostDirectionModel()
        feat = {col: 0.0 for col in FEATURE_COLS}
        signal, conf = model.predict(feat)
        assert signal == "HOLD"
        assert conf == 0.0

    def test_insufficient_data_raises(self):
        model = XGBoostDirectionModel()
        df = make_dummy_df(100)  # too few rows
        with pytest.raises(ValueError, match="Need at least"):
            model.train(df)

    def test_save_and_load(self, tmp_path):
        model = XGBoostDirectionModel()
        df = make_dummy_df(6000)
        model.train(df)
        path = model.save(str(tmp_path))

        model2 = XGBoostDirectionModel()
        model2.load(path)
        assert model2.is_trained

        feat = {col: float(np.random.randn()) for col in FEATURE_COLS}
        signal, conf = model2.predict(feat)
        assert signal in ("BUY_CE", "BUY_PE", "HOLD")
