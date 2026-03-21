"""
scripts/simulate_event.py — Inject a synthetic news event for end-to-end testing.

Bypasses the news sources and injects a NewsItem directly into the pipeline's
DB layer, then runs it through event detection, market matching, probability
estimation, and decision logic.

This lets you test the full pipeline without waiting for a real news event.

Usage:
    # Basic: inject a headline and watch the dashboard
    python scripts/simulate_event.py --headline "Hurricane Category 5 makes landfall in Florida"

    # Specify source (default: nws)
    python scripts/simulate_event.py --headline "Fed raises rates by 75 basis points" --source gnews

    # Dry run (default): won't place real orders
    python scripts/simulate_event.py --headline "..." --dry-run

    # Live run (USE WITH CAUTION — places real Kalshi orders)
    python scripts/simulate_event.py --headline "..." --live

    # See top market matches without running full pipeline
    python scripts/simulate_event.py --headline "..." --match-only
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("simulate_event")


async def main(args: argparse.Namespace) -> None:
    from db.pool import init_pool, close_pool
    from kalshi.auth import token_manager
    from kalshi.client import KalshiClient
    from kalshi.websocket import KalshiWebSocketManager
    from embeddings.market_index_builder import load_or_build_index
    from embeddings.encoder import encoder
    from embeddings.index import faiss_index
    from pipeline.event_detector import EventDetector
    from pipeline.market_matcher import MarketMatcher, MatchedEvent
    from pipeline.probability_estimator import ProbabilityEstimator
    from pipeline.decision_engine import DecisionEngine
    from pipeline.deduplicator import Deduplicator
    from news.base import NewsItem
    import config.settings as cfg_module

    # Apply dry-run override
    if not args.live:
        os.environ["DRY_RUN"] = "true"
        # Reload settings
        import importlib
        import config.settings
        importlib.reload(config.settings)

    await init_pool()
    await token_manager.start()
    client = KalshiClient()
    await load_or_build_index(client)

    if args.match_only:
        # Just show market matches and exit
        query_embedding = encoder.encode_sync(args.headline)
        results = await faiss_index.search(query_embedding, k=10)
        print(f"\nTop market matches for: '{args.headline}'\n")
        for score, meta in results:
            print(f"  {score:.3f}  {meta['ticker']:<30}  {meta['title']}")
        await client.close()
        await token_manager.stop()
        await close_pool()
        return

    # ── Full pipeline simulation ───────────────────────────────────────────────
    ws_manager = KalshiWebSocketManager()
    await ws_manager.start()

    event_detector = EventDetector()
    await event_detector.initialize()

    deduplicator = Deduplicator()
    market_matcher = MarketMatcher()
    probability_estimator = ProbabilityEstimator(ws_manager, client)
    decision_engine = DecisionEngine(client)

    # Create a synthetic NewsItem
    import uuid
    item = NewsItem(
        source=args.source,
        source_id=f"simulate-{uuid.uuid4().hex[:8]}",
        headline=args.headline,
        published_at=datetime.utcnow(),
        fetched_at=datetime.utcnow(),
        body=None,
        url=None,
    )

    logger.info("Simulating pipeline for headline: %s", args.headline)

    # Run through pipeline stages using queues
    raw_queue = asyncio.Queue()
    deduped_queue = asyncio.Queue()
    events_queue = asyncio.Queue()
    matched_queue = asyncio.Queue()
    candidates_queue = asyncio.Queue()

    # Inject item
    await raw_queue.put(item)

    # Stage 1: Dedup
    await deduplicator._process(item, deduped_queue)
    if deduped_queue.empty():
        logger.info("Item was deduplicated (already in DB from a previous run).")
        await client.close()
        await token_manager.stop()
        await close_pool()
        return

    deduped_item = await deduped_queue.get()

    # Stage 2: Event detection
    from pipeline.event_detector import DetectedEvent
    await event_detector._process(deduped_item, events_queue)
    if events_queue.empty():
        logger.info("Item was filtered out by event detection.")
        await client.close()
        await token_manager.stop()
        await close_pool()
        return

    detected: DetectedEvent = await events_queue.get()
    logger.info(
        "Event detected: score=%.3f (keyword=%.3f, nlp=%.3f)",
        detected.event_score, detected.keyword_score, detected.nlp_score,
    )

    # Stage 3: Market matching
    await market_matcher._process(detected, matched_queue)
    if matched_queue.empty():
        logger.info("No market matches found above similarity threshold.")
        await client.close()
        await token_manager.stop()
        await close_pool()
        return

    matched: MatchedEvent = await matched_queue.get()
    logger.info("Found %d market match(es):", len(matched.matches))
    for m in matched.matches:
        logger.info("  %.3f  %s  —  %s", m.similarity_score, m.ticker, m.title)

    # Stage 4: Probability estimation
    await probability_estimator._process(matched, candidates_queue)
    if candidates_queue.empty():
        logger.info("No trade candidates generated (prices unavailable?).")
        await client.close()
        await token_manager.stop()
        await close_pool()
        return

    # Stage 5: Decision engine
    while not candidates_queue.empty():
        candidate = await candidates_queue.get()
        logger.info(
            "Candidate: %s  p_market=%.3f  p_est=%.3f  edge=%+.3f  confidence=%.3f",
            candidate.ticker, candidate.p_market, candidate.p_estimated,
            candidate.edge, candidate.confidence,
        )
        await decision_engine._evaluate(candidate)

    logger.info("Simulation complete. Check the dashboard for results.")

    await client.close()
    await token_manager.stop()
    await close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject a synthetic news event into the KalshiQuant pipeline.")
    parser.add_argument("--headline", required=True, help="News headline to simulate.")
    parser.add_argument("--source", default="nws", help="Source label (default: nws).")
    parser.add_argument("--match-only", action="store_true", help="Only show market matches, skip trading logic.")
    parser.add_argument("--live", action="store_true", help="Place real orders (default is dry-run).")
    args = parser.parse_args()
    asyncio.run(main(args))
