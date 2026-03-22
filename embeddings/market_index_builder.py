"""
embeddings/market_index_builder.py — Builds and periodically refreshes the FAISS market index.

At startup:
  1. Try to load a saved index from disk (fast path, ~100ms)
  2. If no saved index, fetch all open markets from Kalshi and build a new one

Every MARKET_INDEX_REFRESH_INTERVAL seconds (15 min by default):
  - Fetch all open markets from Kalshi API
  - Re-encode their titles/descriptions
  - Atomically replace the FAISS index (under asyncio.Lock in FAISSIndex)
  - Save to disk

Also updates the market_embeddings table in PostgreSQL so the dashboard
can resolve FAISS index positions back to market tickers.

Embedding text strategy:
  Each market is embedded using a concatenation of its most informative fields:
    "{title}. {subtitle}. Category: {category}. Tags: {tags}"
  This gives the model sufficient context to match news headlines to relevant markets.
"""

import asyncio
import json
import logging
import re
import numpy as np
from collections import Counter
from datetime import datetime
from db.pool import get_pool
from config.settings import settings
from kalshi.client import KalshiClient
from kalshi.models import Market
from embeddings.encoder import encoder
from embeddings.index import faiss_index

# Title patterns that indicate a ranked/multi-outcome market rather than binary yes/no.
# These markets require knowing WHICH outcome the news supports — not handled by the pipeline.
_MULTI_OUTCOME_PATTERNS = re.compile(
    r"^(top\s+\d|which\s+|who\s+will\s+win|ranked\s+|most\s+likely\s+to\s+)",
    re.IGNORECASE,
)

# If this many markets share the same event_ticker, treat the whole group as
# multi-outcome (ranked/selection event) and exclude all of them.
_MULTI_OUTCOME_EVENT_THRESHOLD = 5

logger = logging.getLogger(__name__)


def _build_embedding_text(market: Market) -> str:
    """Construct the text to embed for a market."""
    parts = [market.title]
    if market.subtitle:
        parts.append(market.subtitle)
    return ". ".join(parts)


def _build_metadata(market: Market, faiss_id: int) -> dict:
    """Build the metadata dict stored alongside each FAISS vector."""
    return {
        "ticker": market.ticker,
        "title": market.title,
        "subtitle": market.subtitle,
        "close_time": market.close_time.isoformat() if market.close_time else None,
        "faiss_id": faiss_id,
        "embedding_text": _build_embedding_text(market),
    }


async def build_index(client: KalshiClient) -> None:
    """
    Fetch all open markets, encode them, build the FAISS index, and save to disk.
    Also upserts metadata into the market_embeddings PostgreSQL table.
    """
    logger.info("Building FAISS market index...")
    markets = await client.get_all_open_markets()

    if not markets:
        logger.warning("No open markets returned from Kalshi. Index not built.")
        return

    # Filter out multi-leg parlay markets — bundled sports parlays with no
    # meaningful single title for semantic matching against news headlines.
    markets = [m for m in markets if not m.is_parlay]
    logger.info("After parlay filter: %d markets.", len(markets))

    # Filter out markets with no price data — these are inactive or not yet
    # launched and cannot be traded. Keeping only markets with a bid or ask
    # dramatically reduces the index to actively priced markets.
    markets = [m for m in markets if m.yes_bid_dollars is not None or m.yes_ask_dollars is not None]
    logger.info("After liquidity filter: %d priced markets.", len(markets))

    # Filter out multi-outcome (ranked/selection) markets — these require knowing
    # WHICH specific outcome a news headline supports, which the current pipeline
    # cannot determine. Two detection strategies:
    #   1. Title pattern match (e.g. "Top 3 AI companies...", "Who will win...")
    #   2. Event-ticker grouping: if ≥5 markets share an event_ticker, the whole
    #      group is a ranked selection event (e.g. "rank the top 5 teams")
    event_ticker_counts = Counter(m.event_ticker for m in markets if m.event_ticker)
    multi_outcome_event_tickers = {
        ticker for ticker, count in event_ticker_counts.items()
        if count >= _MULTI_OUTCOME_EVENT_THRESHOLD
    }
    before_mo = len(markets)
    markets = [
        m for m in markets
        if not _MULTI_OUTCOME_PATTERNS.match(m.title or "")
        and m.event_ticker not in multi_outcome_event_tickers
    ]
    logger.info(
        "After multi-outcome filter: %d markets (removed %d).",
        len(markets), before_mo - len(markets),
    )

    if not markets:
        logger.warning("No markets remain after filtering. Index not built.")
        return

    texts = [_build_embedding_text(m) for m in markets]
    logger.info("Encoding %d markets...", len(markets))
    embeddings = await encoder.encode_batch(texts)

    metadata = [_build_metadata(m, i) for i, m in enumerate(markets)]

    # Atomically replace the FAISS index (safe even if searches are running)
    await faiss_index.atomic_replace(embeddings, metadata)
    faiss_index.save()

    # Persist metadata to PostgreSQL for dashboard resolution
    await _upsert_market_embeddings(metadata)

    logger.info("FAISS index built and saved: %d markets.", len(markets))


async def _upsert_market_embeddings(metadata: list[dict]) -> None:
    """Upsert market metadata into the market_embeddings table."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Use a transaction for the bulk upsert
        async with conn.transaction():
            for m in metadata:
                close_time = None
                if m["close_time"]:
                    try:
                        close_time = datetime.fromisoformat(m["close_time"])
                    except ValueError:
                        pass

                await conn.execute(
                    """
                    INSERT INTO market_embeddings
                        (ticker, title, subtitle, category, tags,
                         close_time, faiss_index_id, embedding_text, last_indexed)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())
                    ON CONFLICT (ticker) DO UPDATE SET
                        title = EXCLUDED.title,
                        subtitle = EXCLUDED.subtitle,
                        category = EXCLUDED.category,
                        tags = EXCLUDED.tags,
                        close_time = EXCLUDED.close_time,
                        faiss_index_id = EXCLUDED.faiss_index_id,
                        embedding_text = EXCLUDED.embedding_text,
                        last_indexed = NOW()
                    """,
                    m["ticker"], m["title"], m.get("subtitle"),
                    m.get("category"),
                    m.get("tags") or [],
                    close_time,
                    m["faiss_id"],
                    m["embedding_text"],
                )


async def load_or_build_index(client: KalshiClient) -> None:
    """
    Load a saved index from disk if available (fast startup), otherwise build one.
    Called once at pipeline startup before any news is processed.
    """
    if faiss_index.load():
        logger.info("FAISS index loaded from disk (%d markets). Scheduling background refresh.", faiss_index.total_markets)
    else:
        logger.info("No saved FAISS index found. Building from scratch...")
        await build_index(client)


async def refresh_loop(client: KalshiClient) -> None:
    """
    Background task: rebuild the FAISS index every MARKET_INDEX_REFRESH_INTERVAL seconds.

    This ensures the index stays current as new Kalshi markets open and old ones close.
    The atomic replace in FAISSIndex means live searches are never blocked for more than
    the time it takes to swap the index pointer (~microseconds).
    """
    while True:
        await asyncio.sleep(settings.MARKET_INDEX_REFRESH_INTERVAL)
        try:
            logger.info("Refreshing FAISS market index...")
            await build_index(client)
        except Exception as exc:
            logger.error("FAISS index refresh failed: %s. Will retry at next interval.", exc)
