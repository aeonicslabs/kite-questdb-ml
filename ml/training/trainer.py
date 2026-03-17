"""
Model Trainer — pulls data from QuestDB, generates labels, trains models.

Label generation:
  - For each feature row at time T, look forward `horizon` seconds
  - If price at T+horizon > price at T + threshold: label = 1 (UP)
  - Else: label = 0 (DOWN/FLAT)
  - threshold = 0.3% of current price (configurable)

Training schedule:
  - Called once at market open (warm start from yesterday's data)
  - Called every 30 minutes during market hours (incremental)
  - Can be triggered manually via CLI

MLflow tracking:
  - Each training run logged with params, metrics, model artifact
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import psycopg2

from config.settings import get_settings
from ml.models.xgboost_model import XGBoostDirectionModel, FEATURE_COLS
from ml.models.lstm_model import LSTMDirectionModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Pulls feature data from QuestDB, generates labels, trains and saves models.

    Usage:
        trainer = ModelTrainer()
        metrics = trainer.train_all(symbol="BANKNIFTY26MAR25000CE", lookback_hours=4)
    """

    # Minimum % move to count as UP
    LABEL_THRESHOLD_PCT = 0.003  # 0.3%

    def __init__(self) -> None:
        self._settings = get_settings()
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        mlflow.set_tracking_uri(self._settings.ml.mlflow_tracking_uri)
        mlflow.set_experiment(self._settings.ml.mlflow_experiment_name)

    # ------------------------------------------------------------------
    # DATA FETCHING
    # ------------------------------------------------------------------

    def fetch_features(
        self,
        symbol: Optional[str] = None,
        lookback_hours: float = 4.0,
    ) -> pd.DataFrame:
        """
        Fetch feature rows from QuestDB for the given symbol and time window.
        """
        cfg = self._settings.questdb
        since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S")

        where_clauses = [f"ts > '{since_str}'"]
        if symbol:
            where_clauses.append(f"trading_symbol = '{symbol}'")
        where = " AND ".join(where_clauses)

        query = f"""
        SELECT *
        FROM features
        WHERE {where}
        ORDER BY ts ASC
        """

        try:
            conn = psycopg2.connect(cfg.pg_dsn)
            df = pd.read_sql(query, conn)
            conn.close()
            logger.info("Fetched %d feature rows for %s", len(df), symbol or "all")
            return df
        except Exception as e:
            logger.error("Failed to fetch features from QuestDB: %s", e)
            return pd.DataFrame()

    def fetch_ticks_for_labels(
        self,
        symbol: str,
        lookback_hours: float = 4.0,
    ) -> pd.DataFrame:
        """Fetch raw price ticks for forward-label generation."""
        cfg = self._settings.questdb
        since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S")

        query = f"""
        SELECT ts, instrument_token, trading_symbol, last_price, open_interest
        FROM ticks
        WHERE ts > '{since_str}'
          AND trading_symbol = '{symbol}'
        ORDER BY ts ASC
        """

        try:
            conn = psycopg2.connect(cfg.pg_dsn)
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error("Failed to fetch ticks: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # LABEL GENERATION
    # ------------------------------------------------------------------

    def generate_labels(
        self,
        features_df: pd.DataFrame,
        ticks_df: pd.DataFrame,
        horizon_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Merge features with forward-looking price labels.

        For each feature row at time T:
          - Find price at T + horizon
          - label = 1 if (future_price - current_price) / current_price > threshold
          - label = 0 otherwise

        Returns features_df with 'label' column added.
        """
        cfg = self._settings.ml
        horizon = horizon_seconds or cfg.prediction_horizon_seconds

        if features_df.empty or ticks_df.empty:
            logger.warning("Empty dataframes — cannot generate labels")
            return pd.DataFrame()

        features_df = features_df.copy()
        ticks_df = ticks_df.copy()

        # Ensure timestamps are datetime
        features_df["ts"] = pd.to_datetime(features_df["ts"], utc=True)
        ticks_df["ts"] = pd.to_datetime(ticks_df["ts"], utc=True)

        labels = []
        for _, row in features_df.iterrows():
            t = row["ts"]
            future_t = t + pd.Timedelta(seconds=horizon)

            # Find price closest to future_t
            future_ticks = ticks_df[
                (ticks_df["ts"] >= future_t) &
                (ticks_df["ts"] <= future_t + pd.Timedelta(seconds=30))
            ]

            if future_ticks.empty:
                labels.append(np.nan)
                continue

            current_price = float(row.get("mid_price", 0) or 0)
            future_price = float(future_ticks.iloc[0]["last_price"])

            if current_price <= 0:
                labels.append(np.nan)
                continue

            pct_change = (future_price - current_price) / current_price
            label = 1 if pct_change > self.LABEL_THRESHOLD_PCT else 0
            labels.append(label)

        features_df["label"] = labels
        features_df = features_df.dropna(subset=["label"])
        features_df["label"] = features_df["label"].astype(int)

        up_pct = features_df["label"].mean() * 100
        logger.info(
            "Generated %d labels: UP=%.1f%% DOWN=%.1f%%",
            len(features_df), up_pct, 100 - up_pct,
        )
        return features_df

    # ------------------------------------------------------------------
    # TRAINING ORCHESTRATION
    # ------------------------------------------------------------------

    def train_xgboost(
        self,
        symbol: str,
        lookback_hours: float = 4.0,
    ) -> Tuple[XGBoostDirectionModel, Dict[str, Any]]:
        """Train XGBoost model for a given symbol."""
        features_df = self.fetch_features(symbol, lookback_hours)
        ticks_df = self.fetch_ticks_for_labels(symbol, lookback_hours)
        labeled_df = self.generate_labels(features_df, ticks_df)

        if labeled_df.empty:
            raise ValueError(f"No training data for {symbol}")

        model = XGBoostDirectionModel()
        with mlflow.start_run(run_name=f"xgb_{symbol}_{int(time.time())}"):
            cfg = self._settings.ml
            mlflow.log_params({
                "symbol": symbol,
                "n_estimators": cfg.xgb_n_estimators,
                "max_depth": cfg.xgb_max_depth,
                "learning_rate": cfg.xgb_learning_rate,
                "horizon_seconds": cfg.prediction_horizon_seconds,
                "lookback_hours": lookback_hours,
                "train_rows": len(labeled_df),
            })

            metrics = model.train(labeled_df)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            path = model.save()
            mlflow.log_artifact(path)

        return model, metrics

    def train_lstm(
        self,
        symbol: str,
        lookback_hours: float = 4.0,
    ) -> Tuple[LSTMDirectionModel, Dict[str, Any]]:
        """Train LSTM model for a given symbol."""
        features_df = self.fetch_features(symbol, lookback_hours)
        ticks_df = self.fetch_ticks_for_labels(symbol, lookback_hours)
        labeled_df = self.generate_labels(features_df, ticks_df)

        if labeled_df.empty or len(labeled_df) < 200:
            raise ValueError(f"Insufficient data for LSTM: {len(labeled_df)} rows")

        cfg = self._settings.ml
        seq_len = cfg.lstm_seq_len

        # Build sequences
        feature_vals = labeled_df[FEATURE_COLS].fillna(0).values
        labels = labeled_df["label"].values

        X_seqs, y_seqs = [], []
        for i in range(seq_len, len(feature_vals)):
            X_seqs.append(feature_vals[i - seq_len:i])
            y_seqs.append(labels[i])

        X_arr = np.array(X_seqs)
        y_arr = np.array(y_seqs)

        model = LSTMDirectionModel()
        with mlflow.start_run(run_name=f"lstm_{symbol}_{int(time.time())}"):
            mlflow.log_params({
                "symbol": symbol,
                "seq_len": seq_len,
                "hidden_size": cfg.lstm_hidden_size,
                "num_layers": cfg.lstm_num_layers,
                "horizon_seconds": cfg.prediction_horizon_seconds,
                "train_sequences": len(X_arr),
            })

            metrics = model.train_from_sequences(X_arr, y_arr)
            mlflow.log_metrics(metrics)

            path = model.save()
            mlflow.log_artifact(path)

        return model, metrics

    def train_all(
        self,
        symbols: Optional[list] = None,
        lookback_hours: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Train both XGBoost and LSTM for a list of symbols.
        Falls back gracefully if not enough data.
        """
        results = {}
        syms = symbols or ["BANKNIFTY", "NIFTY"]  # defaults to index-level

        for sym in syms:
            results[sym] = {}
            try:
                _, xgb_metrics = self.train_xgboost(sym, lookback_hours)
                results[sym]["xgboost"] = xgb_metrics
            except Exception as e:
                logger.error("XGBoost training failed for %s: %s", sym, e)
                results[sym]["xgboost"] = {"error": str(e)}

            try:
                _, lstm_metrics = self.train_lstm(sym, lookback_hours)
                results[sym]["lstm"] = lstm_metrics
            except Exception as e:
                logger.warning("LSTM training failed for %s (may need more data): %s", sym, e)
                results[sym]["lstm"] = {"error": str(e)}

        return results
