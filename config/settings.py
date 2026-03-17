"""
Central configuration using Pydantic v2 — type-safe, validated, env-driven.
All secrets come from environment variables (never hardcoded).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of this file's directory (config/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = str(_PROJECT_ROOT / ".env")


class KiteSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KITE_", env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    api_key: str = Field(..., description="Kite Connect API key")
    api_secret: str = Field(..., description="Kite Connect API secret")
    access_token: Optional[str] = Field(None, description="Access token (set after login)")
    request_token: Optional[str] = Field(None, description="Request token from redirect URL")

    # WebSocket config
    ws_reconnect_max_tries: int = Field(50, description="Max reconnect attempts")
    ws_reconnect_max_delay: int = Field(30, description="Max delay between reconnects (s)")

    # Instruments
    # Comma-separated list of exchange:tradingsymbol e.g. "NSE:NIFTY 50,NSE:NIFTY BANK"
    index_instruments: str = Field(
        "NSE:NIFTY 50,NSE:NIFTY BANK",
        description="Index instruments to subscribe",
    )
    # How many weekly option strikes above/below ATM to subscribe per expiry
    option_strikes_range: int = Field(10, description="Strikes above/below ATM")
    # Number of upcoming expiries to subscribe (weekly)
    option_expiries: int = Field(2, description="Upcoming expiries to subscribe")


class QuestDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUESTDB_", extra="ignore")

    host: str = Field("localhost", description="QuestDB host")
    ilp_port: int = Field(9000, description="ILP over HTTP port")
    pg_port: int = Field(8812, description="Postgres wire port")
    http_port: int = Field(9000, description="HTTP API port")
    username: str = Field("admin", description="QuestDB username")
    password: str = Field("quest", description="QuestDB password")

    # Ingestion tuning
    auto_flush_rows: int = Field(1000, description="Rows before auto-flush")
    auto_flush_interval_ms: int = Field(500, description="Interval auto-flush in ms")

    @property
    def ilp_conf(self) -> str:
        return (
            f"http::addr={self.host}:{self.ilp_port};"
            f"username={self.username};"
            f"password={self.password};"
            f"auto_flush_rows={self.auto_flush_rows};"
            f"auto_flush_interval={self.auto_flush_interval_ms};"
        )

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.pg_port}/main"
        )


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ML_", extra="ignore")

    # Training
    feature_window_seconds: int = Field(30, description="Rolling feature window (s)")
    prediction_horizon_seconds: int = Field(60, description="Target horizon for labels (s)")
    min_train_rows: int = Field(5000, description="Minimum rows before training starts")
    retrain_interval_minutes: int = Field(30, description="Retrain every N minutes during market")

    # Model
    xgb_n_estimators: int = Field(200)
    xgb_max_depth: int = Field(6)
    xgb_learning_rate: float = Field(0.05)

    lstm_seq_len: int = Field(60, description="LSTM input sequence length (ticks)")
    lstm_hidden_size: int = Field(128)
    lstm_num_layers: int = Field(2)
    lstm_dropout: float = Field(0.2)

    # Thresholds
    signal_confidence_threshold: float = Field(0.65, description="Min probability to act on signal")

    # MLflow
    mlflow_tracking_uri: str = Field("sqlite:///mlflow.db")
    mlflow_experiment_name: str = Field("kite-options-ml")

    # Paths
    model_dir: str = Field("models/saved", description="Directory for saved models")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    env: str = Field("development", description="development | production")
    log_level: str = Field("INFO")
    paper_trading: bool = Field(True, description="If True, simulate orders only")
    market_open_time: str = Field("09:15", description="IST market open HH:MM")
    market_close_time: str = Field("15:30", description="IST market close HH:MM")
    dashboard_port: int = Field(8000)


class Settings(BaseSettings):
    """Root settings — composes all sub-settings."""
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kite: KiteSettings = Field(default_factory=KiteSettings)
    questdb: QuestDBSettings = Field(default_factory=QuestDBSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
