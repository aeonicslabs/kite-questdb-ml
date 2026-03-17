"""
Live Inference Engine.

Receives feature dicts from the feature engine callback,
runs them through loaded models, emits signals to QuestDB.

Ensemble logic:
  - XGBoost and LSTM must AGREE on direction for a signal to be emitted
  - Final confidence = average of both confidences
  - If models disagree: HOLD

Signal deduplication:
  - Don't emit the same signal for the same instrument within `cooldown_seconds`
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.settings import get_settings
from ml.models.xgboost_model import XGBoostDirectionModel, FEATURE_COLS
from ml.models.lstm_model import LSTMDirectionModel
from storage.questdb_writer import QuestDBWriter

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Live inference: feature dict → signal → QuestDB.

    Usage:
        engine = InferenceEngine(db_writer)
        engine.load_models()  # loads latest from model_dir

        # Called by FeatureEngine callback
        engine.on_features(feature_dict)
    """

    COOLDOWN_SECONDS = 30  # Min seconds between signals for same instrument

    def __init__(self, db_writer: QuestDBWriter) -> None:
        self._db_writer = db_writer
        self._settings = get_settings()

        self._xgb: Optional[XGBoostDirectionModel] = None
        self._lstm: Optional[LSTMDirectionModel] = None

        # Per-instrument sequence buffers for LSTM
        self._seq_buffers: Dict[int, list] = defaultdict(list)
        self._seq_len = self._settings.ml.lstm_seq_len

        # Signal cooldown tracking
        self._last_signal_time: Dict[int, float] = {}

        self._signals_emitted = 0

    def load_models(self, model_dir: Optional[str] = None) -> None:
        """Load the most recently saved models from model_dir."""
        d = Path(model_dir or self._settings.ml.model_dir)
        if not d.exists():
            logger.warning("Model dir %s does not exist — running without models", d)
            return

        # Load latest XGBoost
        xgb_files = sorted(d.glob("xgb_v*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if xgb_files:
            self._xgb = XGBoostDirectionModel()
            self._xgb.load(str(xgb_files[0]))
            logger.info("Loaded XGBoost: %s", xgb_files[0].name)
        else:
            logger.warning("No XGBoost models found in %s", d)

        # Load latest LSTM
        lstm_files = sorted(d.glob("lstm_v*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if lstm_files:
            self._lstm = LSTMDirectionModel()
            self._lstm.load(str(lstm_files[0]))
            logger.info("Loaded LSTM: %s", lstm_files[0].name)
        else:
            logger.info("No LSTM models found — will use XGBoost only")

    def on_features(self, feature_dict: Dict[str, Any]) -> None:
        """
        Callback invoked by FeatureEngine on each computed feature row.
        Runs inference and emits signal if confident.
        """
        token = feature_dict.get("instrument_token")
        if not token:
            return

        # Cooldown check
        now = time.monotonic()
        last = self._last_signal_time.get(token, 0)
        if now - last < self.COOLDOWN_SECONDS:
            return

        # Update sequence buffer for LSTM
        feat_vec = np.array(
            [float(feature_dict.get(col, 0.0) or 0.0) for col in FEATURE_COLS],
            dtype=np.float64,
        )
        self._seq_buffers[token].append(feat_vec)
        if len(self._seq_buffers[token]) > self._seq_len:
            self._seq_buffers[token].pop(0)

        signal, confidence = self._ensemble_predict(feature_dict, token)

        threshold = self._settings.ml.signal_confidence_threshold
        if signal == "HOLD" or confidence < threshold:
            return

        # Emit signal
        self._last_signal_time[token] = now
        self._signals_emitted += 1

        top_feats = self._top_features(feature_dict)
        signal_row = {
            "instrument_token": token,
            "trading_symbol": feature_dict.get("trading_symbol", ""),
            "signal": signal,
            "confidence": confidence,
            "predicted_direction": "UP" if "CE" in signal else "DOWN",
            "predicted_magnitude": 0.0,
            "model_name": "ensemble_xgb_lstm" if self._lstm else "xgboost",
            "model_version": getattr(self._xgb, "version", "unknown"),
            "features_snapshot": str(top_feats),
        }
        self._db_writer.write_signal(signal_row)

        logger.info(
            "Signal emitted: %s %s conf=%.3f (total signals: %d)",
            signal, feature_dict.get("trading_symbol", token),
            confidence, self._signals_emitted,
        )

    def _ensemble_predict(
        self, feature_dict: Dict[str, Any], token: int
    ) -> Tuple[str, float]:
        """Run XGBoost + LSTM and ensemble the results."""
        xgb_signal, xgb_conf = "HOLD", 0.0
        lstm_signal, lstm_conf = "HOLD", 0.0

        if self._xgb and self._xgb.is_trained:
            xgb_signal, xgb_conf = self._xgb.predict(feature_dict)

        if self._lstm and self._lstm.is_trained:
            seq_buf = self._seq_buffers.get(token, [])
            if len(seq_buf) >= 5:
                seq_arr = np.array(seq_buf)
                lstm_signal, lstm_conf = self._lstm.predict(seq_arr)

        # Ensemble: both must agree (or use XGBoost-only if no LSTM)
        if not (self._lstm and self._lstm.is_trained):
            return xgb_signal, xgb_conf

        if xgb_signal == "HOLD" and lstm_signal == "HOLD":
            return "HOLD", 0.0

        if xgb_signal == lstm_signal and xgb_signal != "HOLD":
            avg_conf = (xgb_conf + lstm_conf) / 2.0
            return xgb_signal, avg_conf

        # Disagreement → HOLD
        return "HOLD", 0.0

    def _top_features(self, feature_dict: Dict, top_n: int = 5) -> Dict:
        """Return top N most impactful features by XGBoost importance."""
        if not (self._xgb and self._xgb.feature_importance):
            return {}
        importance = self._xgb.feature_importance
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {k: float(feature_dict.get(k, 0)) for k, _ in top}

    @property
    def signals_emitted(self) -> int:
        return self._signals_emitted

    def update_models(
        self,
        xgb_model: Optional[XGBoostDirectionModel] = None,
        lstm_model: Optional[LSTMDirectionModel] = None,
    ) -> None:
        """Hot-swap models without restarting the engine."""
        if xgb_model:
            self._xgb = xgb_model
            logger.info("XGBoost model hot-swapped: %s", xgb_model.version)
        if lstm_model:
            self._lstm = lstm_model
            logger.info("LSTM model hot-swapped: %s", lstm_model.version)
