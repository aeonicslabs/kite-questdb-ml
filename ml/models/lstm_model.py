"""
LSTM sequence model for direction prediction.

Architecture: 2-layer LSTM → Dropout → Linear → Sigmoid

Input: sequence of `seq_len` feature vectors (one per tick)
Output: probability of UP move in next `prediction_horizon_seconds`

Optimized for Apple Silicon (MPS backend) with CPU fallback.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config.settings import get_settings
from ml.models.xgboost_model import FEATURE_COLS

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Returns MPS (Apple Silicon), CUDA, or CPU — in that order."""
    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal) backend")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Using CUDA backend")
        return torch.device("cuda")
    logger.info("Using CPU backend")
    return torch.device("cpu")


# ------------------------------------------------------------------
# MODEL ARCHITECTURE
# ------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last timestep
        return self.sigmoid(self.fc(out)).squeeze(-1)


# ------------------------------------------------------------------
# TRAINING WRAPPER
# ------------------------------------------------------------------

class LSTMDirectionModel:
    """
    LSTM sequence model wrapper.

    Usage:
        model = LSTMDirectionModel()
        model.train_from_sequences(X_seq, y)  # X_seq: (N, seq_len, features)
        signal, conf = model.predict(feature_sequence)  # feature_sequence: (seq_len, features)
    """

    def __init__(self) -> None:
        self._settings = get_settings().ml
        self._device = get_device()
        self._net: Optional[LSTMClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._version: str = "untrained"
        self._seq_len = self._settings.lstm_seq_len

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def train_from_sequences(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 256
    ) -> Dict[str, Any]:
        """
        Args:
            X: shape (N, seq_len, num_features)
            y: shape (N,) binary labels
        """
        cfg = self._settings
        logger.info("Training LSTM on %d sequences, device=%s", len(X), self._device)

        # Normalize: reshape to 2D, scale, reshape back
        N, seq, feat = X.shape
        X_flat = X.reshape(-1, feat)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_flat).reshape(N, seq, feat)

        # Train/val split (no shuffle — time series)
        split = int(0.8 * N)
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y[:split], y[split:]

        # Tensors
        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.float32)

        train_ds = TensorDataset(Xt, yt)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

        self._net = LSTMClassifier(
            input_size=feat,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            dropout=cfg.lstm_dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )
        criterion = nn.BCELoss()

        best_val_auc = 0.0
        best_state = None

        for epoch in range(epochs):
            self._net.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_pred = self._net(Xv.to(self._device)).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_pred)
            scheduler.step(1 - val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f val_auc=%.4f",
                    epoch + 1, epochs, train_loss / len(train_dl), val_auc,
                )

        if best_state:
            self._net.load_state_dict(best_state)
            logger.info("Loaded best model (val_auc=%.4f)", best_val_auc)

        self._version = f"lstm_v{int(torch.tensor(0).item())}"
        import time; self._version = f"lstm_v{int(time.time())}"

        return {"val_auc": best_val_auc, "epochs": epochs}

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def predict(self, feature_sequence: np.ndarray) -> Tuple[str, float]:
        """
        Args:
            feature_sequence: shape (seq_len, num_features)

        Returns:
            (signal, confidence)
        """
        if self._net is None or self._scaler is None:
            return "HOLD", 0.0

        try:
            seq = feature_sequence
            if seq.shape[0] < self._seq_len:
                # Pad with zeros
                pad = np.zeros((self._seq_len - seq.shape[0], seq.shape[1]))
                seq = np.vstack([pad, seq])
            elif seq.shape[0] > self._seq_len:
                seq = seq[-self._seq_len:]

            # Scale
            flat = seq.reshape(-1, seq.shape[-1])
            scaled = self._scaler.transform(flat).reshape(1, self._seq_len, -1)

            x_t = torch.tensor(scaled, dtype=torch.float32).to(self._device)
            self._net.eval()
            with torch.no_grad():
                prob_up = float(self._net(x_t).cpu().item())

            threshold = self._settings.signal_confidence_threshold
            if abs(prob_up - 0.5) < (threshold - 0.5):
                return "HOLD", prob_up

            signal = "BUY_CE" if prob_up >= 0.5 else "BUY_PE"
            confidence = prob_up if prob_up >= 0.5 else 1.0 - prob_up
            return signal, confidence

        except Exception as e:
            logger.error("LSTM predict error: %s", e)
            return "HOLD", 0.0

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        if self._net is None:
            raise RuntimeError("Model not trained")
        cfg = self._settings
        save_dir = Path(path or cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{self._version}.pt"
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "scaler": self._scaler,
                "version": self._version,
                "seq_len": self._seq_len,
                "net_config": {
                    "input_size": len(FEATURE_COLS),
                    "hidden_size": self._settings.lstm_hidden_size,
                    "num_layers": self._settings.lstm_num_layers,
                    "dropout": self._settings.lstm_dropout,
                },
            },
            filepath,
        )
        logger.info("LSTM model saved to %s", filepath)
        return str(filepath)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self._device)
        cfg = data["net_config"]
        self._net = LSTMClassifier(**cfg).to(self._device)
        self._net.load_state_dict(data["state_dict"])
        self._net.eval()
        self._scaler = data["scaler"]
        self._version = data["version"]
        self._seq_len = data["seq_len"]
        logger.info("LSTM model loaded from %s", path)

    @property
    def is_trained(self) -> bool:
        return self._net is not None
