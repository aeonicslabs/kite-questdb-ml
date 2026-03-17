"""
Kite WebSocket Ingestion Layer.

Architecture:
  - KiteTickerWorker runs KiteTicker in a background thread (kiteconnect's own event loop)
  - On each tick batch, it enqueues to an asyncio.Queue for downstream consumers
  - Downstream: QuestDB writer + feature engine run as asyncio tasks

Key design decisions:
  - Never block the WebSocket callback — just enqueue and return immediately
  - One persistent QuestDB writer (ILP sender) shared across ticks
  - Instrument meta lookup for symbol enrichment
  - Automatic reconnect is handled by KiteTicker itself (ws_reconnect_*)
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from kiteconnect import KiteTicker

from config.settings import get_settings
from ingestion.instrument_manager import InstrumentManager
from storage.questdb_writer import QuestDBWriter

logger = logging.getLogger(__name__)


class KiteTickerWorker:
    """
    Wraps KiteTicker in a background thread.
    Tick batches are pushed to an asyncio queue for async consumers.

    Usage:
        worker = KiteTickerWorker(
            api_key="...", access_token="...",
            instrument_manager=mgr,
            tick_queue=asyncio.Queue(),
        )
        worker.start()
        # ... consume from tick_queue ...
        worker.stop()
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        instrument_manager: InstrumentManager,
        tick_queue: "asyncio.Queue[List[Dict]]",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._instrument_manager = instrument_manager
        self._tick_queue = tick_queue
        self._loop = loop or asyncio.get_event_loop()
        self._settings = get_settings().kite

        self._kws: Optional[KiteTicker] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ticks_received = 0
        self._last_tick_time: Optional[float] = None

    def start(self) -> None:
        """Start the WebSocket in a background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_ws, name="KiteTickerThread", daemon=True
        )
        self._thread.start()
        logger.info("KiteTickerWorker started")

    def stop(self) -> None:
        """Gracefully stop the WebSocket."""
        self._running = False
        if self._kws:
            try:
                self._kws.stop()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(
            "KiteTickerWorker stopped. Total ticks received: %d",
            self._ticks_received,
        )

    @property
    def is_connected(self) -> bool:
        return self._kws is not None and self._kws.is_connected()

    @property
    def ticks_received(self) -> int:
        return self._ticks_received

    # ------------------------------------------------------------------
    # PRIVATE — WebSocket callbacks (run in KiteTicker's thread)
    # ------------------------------------------------------------------

    def _run_ws(self) -> None:
        """Initialize KiteTicker and connect. Runs in a dedicated thread."""
        cfg = self._settings

        self._kws = KiteTicker(
            api_key=self._api_key,
            access_token=self._access_token,
            reconnect=True,
            reconnect_max_tries=cfg.ws_reconnect_max_tries,
            reconnect_max_delay=cfg.ws_reconnect_max_delay,
        )

        self._kws.on_ticks = self._on_ticks
        self._kws.on_connect = self._on_connect
        self._kws.on_close = self._on_close
        self._kws.on_error = self._on_error
        self._kws.on_reconnect = self._on_reconnect
        self._kws.on_noreconnect = self._on_noreconnect

        logger.info("Connecting to Kite WebSocket...")
        self._kws.connect(threaded=False)  # blocks until stop() is called

    def _on_connect(self, ws, response) -> None:
        tokens = self._instrument_manager.get_subscription_tokens()
        if not tokens:
            logger.warning("No tokens to subscribe — instrument manager may not be initialized")
            return

        logger.info("WebSocket connected. Subscribing %d tokens in FULL mode...", len(tokens))
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        logger.info("Subscription complete.")

    def _on_ticks(self, ws, ticks: List[Dict]) -> None:
        """Called on each tick batch. Must not block."""
        if not ticks:
            return

        self._ticks_received += len(ticks)
        self._last_tick_time = time.monotonic()

        # Enrich ticks with metadata
        enriched = []
        for tick in ticks:
            token = tick.get("instrument_token")
            meta = self._instrument_manager.get_instrument_meta(token)
            if meta:
                tick["trading_symbol"] = meta["trading_symbol"]
                tick["exchange"] = meta["exchange"]
                tick["lot_size"] = meta["lot_size"]
                tick["strike"] = meta["strike"]
                tick["instrument_type"] = meta["instrument_type"]
                tick["name"] = meta["name"]
                tick["expiry"] = meta["expiry"]
            else:
                tick["trading_symbol"] = tick.get("trading_symbol", str(token))
                tick["exchange"] = "NFO"

            # Update spot price if this is an index tick
            sym = tick.get("trading_symbol", "")
            if "NIFTY BANK" in sym or "NIFTY BANK" in str(token):
                self._instrument_manager.update_spot("BANKNIFTY", tick.get("last_price", 0))
            elif "NIFTY 50" in sym or "NIFTY 50" in str(token):
                self._instrument_manager.update_spot("NIFTY", tick.get("last_price", 0))

            enriched.append(tick)

        # Thread-safe enqueue to asyncio queue
        try:
            asyncio.run_coroutine_threadsafe(
                self._tick_queue.put(enriched), self._loop
            )
        except Exception as e:
            logger.error("Failed to enqueue ticks: %s", e)

    def _on_close(self, ws, code, reason) -> None:
        logger.warning("WebSocket closed: code=%s reason=%s", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        logger.error("WebSocket error: code=%s reason=%s", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.info("WebSocket reconnecting (attempt %d)...", attempts_count)

    def _on_noreconnect(self, ws) -> None:
        logger.critical(
            "WebSocket exhausted all reconnect attempts. Manual restart required."
        )
        self._running = False


# ------------------------------------------------------------------
# ASYNC PIPELINE ORCHESTRATOR
# ------------------------------------------------------------------

class IngestionPipeline:
    """
    Async orchestrator for the full ingestion pipeline.

    Starts:
      - KiteTickerWorker (background thread, feeds tick_queue)
      - tick_to_db_task: drains tick_queue → QuestDB writer
      - feature_dispatch_task: routes ticks to feature engine queue

    Usage:
        pipeline = IngestionPipeline(kite_auth, instrument_manager)
        await pipeline.start()
        await pipeline.run_forever()   # blocks until Ctrl+C
        await pipeline.stop()
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        instrument_manager: InstrumentManager,
        db_writer: QuestDBWriter,
        feature_queue: Optional["asyncio.Queue[List[Dict]]"] = None,
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._instrument_manager = instrument_manager
        self._db_writer = db_writer
        self._feature_queue = feature_queue

        self._tick_queue: asyncio.Queue[List[Dict]] = asyncio.Queue(maxsize=50_000)
        self._loop = asyncio.get_event_loop()
        self._ticker_worker: Optional[KiteTickerWorker] = None
        self._tasks: list = []

        self._ticks_written = 0
        self._start_time: Optional[float] = None

    async def start(self) -> None:
        self._start_time = time.monotonic()

        # Start WebSocket worker thread
        self._ticker_worker = KiteTickerWorker(
            api_key=self._api_key,
            access_token=self._access_token,
            instrument_manager=self._instrument_manager,
            tick_queue=self._tick_queue,
            loop=self._loop,
        )
        self._ticker_worker.start()

        # Start async drain tasks
        self._tasks = [
            asyncio.create_task(self._tick_consumer(), name="tick_consumer"),
            asyncio.create_task(self._stats_logger(), name="stats_logger"),
        ]
        logger.info("IngestionPipeline started")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._ticker_worker:
            self._ticker_worker.stop()
        self._db_writer.close()
        logger.info("IngestionPipeline stopped. Total written: %d", self._ticks_written)

    async def run_forever(self) -> None:
        """Block until all tasks complete (or Ctrl+C)."""
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def _tick_consumer(self) -> None:
        """Drain tick_queue → QuestDB + optional feature_queue."""
        while True:
            try:
                ticks: List[Dict] = await self._tick_queue.get()
                for tick in ticks:
                    self._db_writer.write_tick(tick)
                    self._ticks_written += 1

                    # Forward to feature engine if queue provided
                    if self._feature_queue and not self._feature_queue.full():
                        await self._feature_queue.put([tick])

                self._tick_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in tick consumer: %s", e)

    async def _stats_logger(self) -> None:
        """Log ingestion stats every 30 seconds."""
        while True:
            try:
                await asyncio.sleep(30)
                elapsed = time.monotonic() - (self._start_time or time.monotonic())
                rate = self._ticks_written / max(elapsed, 1)
                queue_size = self._tick_queue.qsize()
                logger.info(
                    "Ingestion stats: written=%d rate=%.1f/s queue_backlog=%d ws_connected=%s",
                    self._ticks_written,
                    rate,
                    queue_size,
                    self._ticker_worker.is_connected if self._ticker_worker else False,
                )
            except asyncio.CancelledError:
                break
