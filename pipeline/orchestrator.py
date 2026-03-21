"""
pipeline/orchestrator.py — The central async coordinator.

Wires all pipeline stages together using asyncio.Queue and asyncio.TaskGroup.

Pipeline topology:
  ┌─────────────────────────────────────────────────────────────┐
  │  News Sources (3 tasks)                                     │
  │  NWSSource | GDELTSource | RSSSource                        │
  └──────────────────────┬──────────────────────────────────────┘
                         │ raw_news_queue (maxsize=500)
                         ▼
                   Deduplicator
                         │ deduped_queue (maxsize=200)
                         ▼
                   EventDetector
                         │ events_queue (maxsize=100)
                         ▼
                   MarketMatcher
                         │ matched_queue (maxsize=100)
                         ▼
               ProbabilityEstimator  ←── KalshiWebSocket (live prices)
                         │ candidates_queue (maxsize=50)
                         ▼
                   DecisionEngine
                         │
                   KalshiClient.place_order()

Side tasks (run concurrently via TaskGroup):
  - KalshiWebSocket connection loop
  - Market FAISS index refresh (every 15 min)
  - Pipeline heartbeat (every 60s)
  - Token refresh (handled inside KalshiTokenManager)

Startup sequence:
  1. Initialize DB connection pool
  2. Start Kalshi token manager (authenticate + start refresh loop)
  3. Load or build FAISS index
  4. Start WebSocket manager
  5. Start pipeline stage tasks
  6. Start news source tasks
  7. Start index refresh loop
  8. Start heartbeat loop

Shutdown:
  asyncio.TaskGroup propagates cancellation to all tasks on any unhandled exception.
  The main() function catches KeyboardInterrupt for clean shutdown.

Entry point:
  python -m pipeline.orchestrator
"""

import asyncio
import logging
import signal
import sys
from uuid import UUID

from config.settings import settings
from db.pool import init_pool, close_pool
from db.repositories.metrics import insert_pipeline_run, update_pipeline_heartbeat
from embeddings.market_index_builder import load_or_build_index, refresh_loop
from kalshi.auth import token_manager
from kalshi.client import KalshiClient
from kalshi.websocket import KalshiWebSocketManager
from news.gdelt import GDELTSource
from news.nws import NWSSource
from news.rss import RSSSource
from pipeline.deduplicator import Deduplicator
from pipeline.decision_engine import DecisionEngine
from pipeline.event_detector import EventDetector
from pipeline.market_matcher import MarketMatcher
from pipeline.probability_estimator import ProbabilityEstimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("orchestrator")

# ── Queue sizes ────────────────────────────────────────────────────────────────
# Smaller queues cause earlier backpressure — items are dropped rather than
# piling up memory when the pipeline falls behind.
_RAW_QUEUE_SIZE = 500       # large: many sources feed here simultaneously
_DEDUPED_QUEUE_SIZE = 200   # dedup discards ~80%+ of raw items
_EVENTS_QUEUE_SIZE = 100    # event detection further filters
_MATCHED_QUEUE_SIZE = 100   # market matching further filters
_CANDIDATES_QUEUE_SIZE = 50  # final candidates before decision gate


async def _heartbeat_loop(run_id: UUID, counters: dict) -> None:
    """Update the pipeline_runs heartbeat every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        try:
            await update_pipeline_heartbeat(
                run_id=run_id,
                news_fetched=counters["news_fetched"],
                events_detected=counters["events_detected"],
                markets_matched=counters["markets_matched"],
                trades_executed=counters["trades_executed"],
                trades_rejected=counters["trades_rejected"],
            )
        except Exception as exc:
            logger.warning("Heartbeat update failed: %s", exc)


async def run() -> None:
    """
    Main pipeline entry point.
    Starts all tasks and runs until interrupted or a task raises an unhandled exception.
    """
    logger.info("KalshiQuant starting up...")

    # ── Step 1: Database ──────────────────────────────────────────────────────
    await init_pool()
    run_id = await insert_pipeline_run()
    logger.info("Pipeline run ID: %s", run_id)

    # ── Step 2: Kalshi authentication ─────────────────────────────────────────
    await token_manager.start()
    client = KalshiClient()

    # ── Step 3: FAISS index ───────────────────────────────────────────────────
    await load_or_build_index(client)

    # ── Step 4: WebSocket ─────────────────────────────────────────────────────
    ws_manager = KalshiWebSocketManager()
    await ws_manager.start()

    # ── Step 5: Pipeline stage instances ─────────────────────────────────────
    deduplicator = Deduplicator()
    event_detector = EventDetector()
    await event_detector.initialize()   # pre-compute NLP prototypes

    market_matcher = MarketMatcher()
    probability_estimator = ProbabilityEstimator(ws_manager, client)
    decision_engine = DecisionEngine(client)

    # ── Step 6: Queues ────────────────────────────────────────────────────────
    raw_queue = asyncio.Queue(maxsize=_RAW_QUEUE_SIZE)
    deduped_queue = asyncio.Queue(maxsize=_DEDUPED_QUEUE_SIZE)
    events_queue = asyncio.Queue(maxsize=_EVENTS_QUEUE_SIZE)
    matched_queue = asyncio.Queue(maxsize=_MATCHED_QUEUE_SIZE)
    candidates_queue = asyncio.Queue(maxsize=_CANDIDATES_QUEUE_SIZE)

    # ── Step 7: News sources ──────────────────────────────────────────────────
    news_sources = [
        NWSSource(),
        GDELTSource(),
        RSSSource(),
    ]

    # ── Counters for heartbeat ─────────────────────────────────────────────────
    # Shared mutable dict (safe in single-threaded asyncio)
    counters = {
        "news_fetched": 0,
        "events_detected": 0,
        "markets_matched": 0,
        "trades_executed": 0,
        "trades_rejected": 0,
    }

    logger.info(
        "Pipeline initialized. DRY_RUN=%s | PROBABILITY_SHIFT_MIN=%.2f | MAX_EXPOSURE=$%.0f",
        settings.DRY_RUN, settings.PROBABILITY_SHIFT_MIN, settings.MAX_TOTAL_EXPOSURE_USD,
    )

    # ── Step 8: Launch all tasks ──────────────────────────────────────────────
    async with asyncio.TaskGroup() as tg:
        # News sources
        for source in news_sources:
            tg.create_task(source.run_forever(raw_queue))

        # Pipeline stages
        tg.create_task(deduplicator.run(raw_queue, deduped_queue))
        tg.create_task(event_detector.run(deduped_queue, events_queue))
        tg.create_task(market_matcher.run(events_queue, matched_queue))
        tg.create_task(probability_estimator.run(matched_queue, candidates_queue))
        tg.create_task(decision_engine.run(candidates_queue))

        # Background maintenance tasks
        tg.create_task(refresh_loop(client))
        tg.create_task(_heartbeat_loop(run_id, counters))

    logger.info("Pipeline stopped.")


async def shutdown() -> None:
    """Graceful shutdown: close connections."""
    logger.info("Shutting down...")
    await token_manager.stop()
    await close_pool()


def main() -> None:
    """CLI entry point: python -m pipeline.orchestrator"""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal.")
    finally:
        asyncio.run(shutdown())


if __name__ == "__main__":
    main()
