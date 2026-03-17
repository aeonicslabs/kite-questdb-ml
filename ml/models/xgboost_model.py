"""
XGBoost direction classifier.

Target: will the option price move UP (1) or DOWN (0) within
        the next `prediction_horizon_seconds` (default 60s)?

Features: all columns from the `features` table.

Training:
  - Pulls feature + forward-label pairs from QuestDB
  - Trains XGBoost classifier
  - Saves model artifact + feature importances

Inference:
  - Accepts a single feature dict
  - Returns (signal, confidence) tuple
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from config.settings import get_settings

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "bid_ask_spread",
    "mid_price",
    "order_book_imbalance",
    "depth_imbalance_l3",
    "price_change_1s",
    "price_change_5s",
    "price_change_30s",
    "vwap_deviation",
    "volume_rate",
    "buy_sell_ratio",
    "oi_change_rate",
    "high_30s",
    "low_30s",
    "range_30s",
    "rsi_14",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "iv_approx",
    "pcr_oi",
    "atm_distance_pct",
]


class XGBoostDirectionModel:
    """
    XGBoost binary classifier: UP (1) or DOWN/FLAT (0).

    Usage:
        model = XGBoostDirectionModel()
        model.train(features_df)
        signal, confidence = model.predict(feature_dict)
        model.save("models/saved/xgb_v1.pkl")
        model.load("models/saved/xgb_v1.pkl")
    """

    def __init__(self) -> None:
        self._settings = get_settings().ml
        self._model: Optional[xgb.XGBClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._version: str = "untrained"
        self._feature_importance: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train on a DataFrame with feature columns + 'label' column.

        Args:
            df: DataFrame with FEATURE_COLS + 'label' (0 or 1)

        Returns:
            dict with training metrics
        """
        cfg = self._settings
        logger.info("Training XGBoost on %d rows...", len(df))

        # Validate
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if "label" not in df.columns:
            raise ValueError("DataFrame must have a 'label' column")

        # Clean
        df = df[FEATURE_COLS + ["label"]].dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < cfg.min_train_rows:
            raise ValueError(
                f"Need at least {cfg.min_train_rows} rows, got {len(df)}"
            )

        X = df[FEATURE_COLS].values
        y = df["label"].values.astype(int)

        # Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False  # no shuffle for time-series
        )

        # Class balance check
        pos_rate = y_train.mean()
        logger.info("Label distribution: UP=%.2f%% DOWN=%.2f%%", pos_rate * 100, (1 - pos_rate) * 100)
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

        self._model = xgb.XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",           # fastest, works on Apple Silicon
            device="cpu",                 # set to "cuda" if GPU available
            eval_metric="logloss",

            random_state=42,
            n_jobs=-1,
        )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Evaluate
        y_pred = self._model.predict(X_test)
        y_prob = self._model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        # Feature importances
        self._feature_importance = dict(
            zip(FEATURE_COLS, self._model.feature_importances_.tolist())
        )
        top_features = sorted(
            self._feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]

        metrics = {
            "accuracy": report["accuracy"],
            "auc_roc": auc,
            "precision_up": report.get("1", {}).get("precision", 0),
            "recall_up": report.get("1", {}).get("recall", 0),
            "f1_up": report.get("1", {}).get("f1-score", 0),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "top_features": top_features,
        }

        logger.info(
            "XGBoost trained: accuracy=%.3f AUC=%.3f",
            metrics["accuracy"], metrics["auc_roc"],
        )
        logger.info("Top features: %s", top_features)
        self._version = f"xgb_v{int(pd.Timestamp.now().timestamp())}"

        return metrics

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def predict(self, feature_dict: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predict direction from a single feature dict.

        Returns:
            (signal, confidence) where signal is 'BUY_CE', 'BUY_PE', or 'HOLD'
        """
        if self._model is None or self._scaler is None:
            return "HOLD", 0.0

        try:
            x = np.array(
                [float(feature_dict.get(col, 0.0) or 0.0) for col in FEATURE_COLS],
                dtype=np.float64,
            ).reshape(1, -1)

            # Replace inf/nan
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            x_scaled = self._scaler.transform(x)
            proba = self._model.predict_proba(x_scaled)[0]
            confidence = float(max(proba))
            direction = int(np.argmax(proba))

            threshold = self._settings.signal_confidence_threshold
            if confidence < threshold:
                return "HOLD", confidence

            signal = "BUY_CE" if direction == 1 else "BUY_PE"
            return signal, confidence

        except Exception as e:
            logger.error("XGBoost predict error: %s", e)
            return "HOLD", 0.0

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        if self._model is None:
            raise RuntimeError("Model not trained yet")
        cfg = self._settings
        save_dir = Path(path or cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{self._version}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "scaler": self._scaler,
                    "version": self._version,
                    "feature_cols": FEATURE_COLS,
                    "feature_importance": self._feature_importance,
                },
                f,
            )
        logger.info("XGBoost model saved to %s", filepath)
        return str(filepath)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._version = data["version"]
        self._feature_importance = data.get("feature_importance", {})
        logger.info("XGBoost model loaded from %s (version=%s)", path, self._version)

    @property
    def version(self) -> str:
        return self._version

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def feature_importance(self) -> Dict[str, float]:
        return self._feature_importance.copy()
