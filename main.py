"""
Main entry point for the Kite ML Trading Pipeline.

Startup sequence:
  1. Load config from .env
  2. Authenticate with Kite
  3. Initialize QuestDB (run schema if needed)
  4. Build instrument universe
  5. Start IngestionPipeline (WebSocket → QuestDB)
  6. Start FeatureEngine (ticks → features → QuestDB)
  7. Load ML models (if available) → InferenceEngine
  8. Schedule model retraining every 30 minutes
  9. Block until Ctrl+C

Usage:
    python main.py

For first-time login:
    python -m ingestion.kite_auth
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from config.settings import get_settings
from ingestion.kite_auth import get_authenticated_kite
from ingestion.instrument_manager import InstrumentManager
from ingestion.kite_ws import IngestionPipeline
from storage.questdb_writer import QuestDBWriter
from features.engineer import FeatureEngine
from ml.inference.engine import InferenceEngine
from ml.training.trainer import ModelTrainer
from scripts.init_db import init_questdb

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("kiteconnect").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ------------------------------------------------------------------
# MAIN ASYNC PIPELINE
# ------------------------------------------------------------------

async def main() -> None:
    cfg = get_settings()

    logger.info("=" * 60)
    logger.info("Kite ML Trading Pipeline")
    logger.info("Environment: %s | Paper trading: %s", cfg.app.env, cfg.app.paper_trading)
    logger.info("=" * 60)

    # 1. Authenticate Kite
    logger.info("Authenticating with Kite Connect...")
    try:
        kite = get_authenticated_kite()
    except RuntimeError as e:
        logger.critical(str(e))
        sys.exit(1)

    # 2. Initialize QuestDB schema
    logger.info("Initializing QuestDB schema...")
    try:
        init_questdb()
    except Exception as e:
        logger.error("QuestDB init failed: %s", e)
        logger.error("Make sure QuestDB is running: docker compose up -d questdb")
        sys.exit(1)

    # 3. Build instrument universe
    logger.info("Building instrument universe...")
    instrument_mgr = InstrumentManager(kite)
    try:
        instrument_mgr.refresh()
    except Exception as e:
        logger.critical("Failed to load instruments: %s", e)
        sys.exit(1)

    # 4. Initialize writers and engines
    db_writer = QuestDBWriter()
    feature_queue: asyncio.Queue = asyncio.Queue(maxsize=100_000)
    inference_engine = InferenceEngine(db_writer)
    inference_engine.load_models()

    feature_engine = FeatureEngine(
        db_writer=db_writer,
        on_features=inference_engine.on_features,
    )

    # 5. Start ingestion pipeline
    pipeline = IngestionPipeline(
        api_key=cfg.kite.api_key,
        access_token=cfg.kite.access_token,
        instrument_manager=instrument_mgr,
        db_writer=db_writer,
        feature_queue=feature_queue,
    )
    await pipeline.start()

    # 6. Start feature engine task
    feature_task = asyncio.create_task(
        feature_engine.run(feature_queue), name="feature_engine"
    )

    # 7. Start periodic model retraining
    retrain_task = asyncio.create_task(
        periodic_retrain(inference_engine, instrument_mgr),
        name="retrain_scheduler",
    )

    # 8. Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _shutdown():
        logger.info("Shutdown signal received...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    logger.info("Pipeline running. Press Ctrl+C to stop.")

    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down gracefully...")
        feature_task.cancel()
        retrain_task.cancel()
        await pipeline.stop()
        db_writer.flush()
        db_writer.close()
        logger.info("Shutdown complete.")


async def periodic_retrain(
    inference_engine: InferenceEngine,
    instrument_mgr: InstrumentManager,
) -> None:
    """
    Retrain models every 30 minutes during market hours.
    """
    cfg = get_settings().ml
    interval = cfg.retrain_interval_minutes * 60

    # Wait initial period before first retrain (let data accumulate)
    await asyncio.sleep(interval)

    trainer = ModelTrainer()

    while True:
        try:
            logger.info("Starting scheduled model retraining...")
            # Get a sample of subscribed symbols
            meta = instrument_mgr.get_all_meta()
            # Pick ATM CE and PE for BANKNIFTY and NIFTY
            symbols = []
            for token, m in meta.items():
                if m.get("instrument_type") in ("CE", "PE"):
                    symbols.append(m["trading_symbol"])
                if len(symbols) >= 4:
                    break

            if not symbols:
                logger.warning("No symbols available for retraining")
            else:
                results = trainer.train_all(symbols=symbols, lookback_hours=4.0)
                logger.info("Retraining results: %s", results)

                # Hot-swap models in inference engine
                from ml.models.xgboost_model import XGBoostDirectionModel
                model_dir = cfg.model_dir
                new_xgb_files = sorted(
                    Path(model_dir).glob("xgb_v*.pkl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if new_xgb_files:
                    new_xgb = XGBoostDirectionModel()
                    new_xgb.load(str(new_xgb_files[0]))
                    inference_engine.update_models(xgb_model=new_xgb)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Retraining failed: %s", e)

        await asyncio.sleep(interval)


if __name__ == "__main__":
    asyncio.run(main())
