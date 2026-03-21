"""
pipeline/market_matcher.py — FAISS-based market matching.

For each detected event, finds the most semantically similar open Kalshi markets
using the FAISS vector index built by embeddings/market_index_builder.py.

Process:
  1. Encode the news headline using the same model as the market index (all-MiniLM-L6-v2)
  2. Search the FAISS index for top-k similar markets
  3. Filter results by SIMILARITY_MIN_SCORE
  4. Write match results to DB (market_matches table)
  5. Forward matched events downstream for probability estimation

Why cosine similarity?
  Cosine similarity measures the angle between two embedding vectors, capturing
  semantic relevance independent of text length. A headline about "hurricane warning
  Florida" will have high cosine similarity to a market titled "Will a hurricane
  make landfall in Florida this month?" even though the exact words differ.
"""

import asyncio
import logging
from dataclasses import dataclass

from config.settings import settings
from db.repositories.metrics import insert_market_match
from embeddings.encoder import encoder
from embeddings.index import faiss_index
from pipeline.event_detector import DetectedEvent

logger = logging.getLogger(__name__)


@dataclass
class MarketMatch:
    """A single (market, similarity_score) match for a detected event."""
    ticker: str
    title: str
    category: str | None
    close_time: str | None    # ISO 8601 string or None
    similarity_score: float
    embedding_text: str       # the text that was embedded for this market (for audit)


@dataclass
class MatchedEvent:
    """A detected event with its list of matching markets."""
    detected: DetectedEvent
    matches: list[MarketMatch]


class MarketMatcher:
    """Matches detected events to Kalshi markets via FAISS similarity search."""

    async def run(
        self,
        events_queue: asyncio.Queue,
        matched_queue: asyncio.Queue,
    ) -> None:
        """
        Consume DetectedEvents, find matching markets, produce MatchedEvents.
        Items with no matching markets (below similarity threshold) are dropped.
        """
        while True:
            event: DetectedEvent = await events_queue.get()
            try:
                await self._process(event, matched_queue)
            except Exception as exc:
                logger.error("MarketMatcher error: %s", exc)
            finally:
                events_queue.task_done()

    async def _process(self, event: DetectedEvent, matched_queue: asyncio.Queue) -> None:
        """Encode the headline, search FAISS, and forward if matches found."""
        item = event.item

        # Encode the headline (synchronous ~8ms — acceptable in async context)
        query_embedding = encoder.encode_sync(item.headline)

        # Search FAISS for top-k similar markets
        results = await faiss_index.search(query_embedding, k=settings.SIMILARITY_TOP_K)

        if not results:
            # No markets matched above the similarity threshold — drop item
            logger.debug("No market matches for: %s", item.headline[:60])
            return

        # Build MarketMatch objects from FAISS results
        matches = []
        for score, meta in results:
            match = MarketMatch(
                ticker=meta["ticker"],
                title=meta["title"],
                category=meta.get("category"),
                close_time=meta.get("close_time"),
                similarity_score=score,
                embedding_text=meta.get("embedding_text", ""),
            )
            matches.append(match)

            # Persist to DB for dashboard visibility
            if item.db_id:
                await insert_market_match(
                    news_event_id=item.db_id,
                    market_ticker=match.ticker,
                    market_title=match.title,
                    market_category=match.category,
                    similarity_score=score,
                )

        # Stamp match timestamp
        if item.trace:
            item.trace.stamp_matched()

        logger.info(
            "Matched %d markets for: %s (best: %s %.3f)",
            len(matches), item.headline[:50],
            matches[0].ticker, matches[0].similarity_score,
        )

        matched_event = MatchedEvent(detected=event, matches=matches)
        try:
            matched_queue.put_nowait(matched_event)
        except asyncio.QueueFull:
            logger.warning("Matched queue full; dropping matched event.")
