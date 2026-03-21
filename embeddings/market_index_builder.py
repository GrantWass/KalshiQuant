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
import numpy as np
from datetime import datetime
from db.pool import get_pool
from config.settings import settings
from kalshi.client import KalshiClient
from kalshi.models import Market
from embeddings.encoder import encoder
from embeddings.index import faiss_index

logger = logging.getLogger(__name__)


def _build_embedding_text(market: Market) -> str:
    """
    Construct the text to embed for a market.
    More context = better semantic matching. Include all informative fields.
    """
    parts = [market.title]
    if market.subtitle:
        parts.append(market.subtitle)
    if market.category:
        parts.append(f"Category: {market.category}")
    if market.tags:
        parts.append(f"Tags: {', '.join(market.tags)}")
    return ". ".join(parts)


def _build_metadata(market: Market, faiss_id: int) -> dict:
    """Build the metadata dict stored alongside each FAISS vector."""
    return {
        "ticker": market.ticker,
        "title": market.title,
        "subtitle": market.subtitle,
        "category": market.category,
        "tags": market.tags,
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
