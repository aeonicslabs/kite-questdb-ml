"""
Test configuration — sets required environment variables so unit tests
run without real Kite API credentials.
"""
import os
import pytest
from functools import lru_cache

# Inject stub credentials before any imports touch settings
os.environ.setdefault("KITE_API_KEY", "test_api_key")
os.environ.setdefault("KITE_API_SECRET", "test_api_secret")
os.environ.setdefault("KITE_ACCESS_TOKEN", "test_access_token")
os.environ.setdefault("QUESTDB_HOST", "localhost")
os.environ.setdefault("APP_PAPER_TRADING", "true")
os.environ.setdefault("ML_MODEL_DIR", "/tmp/test_models")
os.environ.setdefault("ML_MLFLOW_TRACKING_URI", "sqlite:////tmp/test_mlflow.db")


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the lru_cache on get_settings between tests."""
    from config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
