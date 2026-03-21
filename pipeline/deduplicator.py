"""
pipeline/deduplicator.py — Duplicate news item suppression.

Two-level deduplication:
  1. In-memory LRU set: O(1) check for recently seen (source, source_id) pairs.
     Bounded by DEDUP_MAX_MEMORY_ENTRIES to prevent unbounded memory growth.
     Uses collections.OrderedDict as an LRU eviction structure.

  2. DB check: for items older than the in-memory window, query news_events
     table on the (source, source_id) unique index.
     Only performed if the item passes the in-memory check (rare for old items).

Why two levels?
  The same GDELT file can be partially re-processed across polling intervals.
  The same RSS item appears in multiple feed fetches within the same minute.
  The in-memory set handles 99%+ of duplicates with zero DB queries.

Items that are duplicates are simply dropped (not written to DB) since they
contribute no new information. The original DB entry is unchanged.
"""

import asyncio
import logging
from collections import OrderedDict

from config.settings import settings
from db.repositories.news import insert_news_event
from news.base import NewsItem

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Stateful deduplicator that consumes from a raw queue and writes to a deduped queue.
    """

    def __init__(self) -> None:
        # LRU cache: OrderedDict where key=(source, source_id), value=True
        # Oldest entries are evicted when max size is reached.
        self._seen: OrderedDict[tuple[str, str], bool] = OrderedDict()
        self._max_size = settings.DEDUP_MAX_MEMORY_ENTRIES
        self._dropped_count = 0

    def _is_duplicate_memory(self, source: str, source_id: str) -> bool:
        """Check in-memory LRU cache. Updates access order (LRU behavior)."""
        key = (source, source_id)
        if key in self._seen:
            # Move to end (most recently used)
            self._seen.move_to_end(key)
            return True
        return False

    def _mark_seen(self, source: str, source_id: str) -> None:
        """Add to LRU cache, evicting the oldest entry if at capacity."""
        key = (source, source_id)
        self._seen[key] = True
        self._seen.move_to_end(key)
        if len(self._seen) > self._max_size:
            self._seen.popitem(last=False)  # evict oldest

    async def run(
        self,
        raw_queue: asyncio.Queue,
        deduped_queue: asyncio.Queue,
    ) -> None:
        """
        Consume from raw_queue, filter duplicates, produce to deduped_queue.
        Runs until task is cancelled.
        """
        while True:
            item: NewsItem = await raw_queue.get()
            try:
                await self._process(item, deduped_queue)
            except Exception as exc:
                logger.error("Deduplicator error for %s/%s: %s", item.source, item.source_id, exc)
            finally:
                raw_queue.task_done()

    async def _process(self, item: NewsItem, deduped_queue: asyncio.Queue) -> None:
        """Process a single item: check for duplicates, then enqueue or drop."""
        # Fast path: in-memory check
        if self._is_duplicate_memory(item.source, item.source_id):
            self._dropped_count += 1
            return

        # Mark as seen in memory before the DB write to handle concurrent duplicates
        self._mark_seen(item.source, item.source_id)

        # Write to DB (INSERT ... ON CONFLICT DO NOTHING handles race conditions)
        db_id = await insert_news_event(
            source=item.source,
            source_id=item.source_id,
            headline=item.headline,
            published_at=item.published_at,
            fetched_at=item.fetched_at,
            body=item.body,
            url=item.url,
            raw_payload=item.raw_payload,
        )

        if db_id is None:
            # INSERT returned nothing → (source, source_id) already in DB
            self._dropped_count += 1
            return

        # Attach DB ID and stamp the dedup timestamp
        item.db_id = db_id
        if item.trace:
            item.trace.news_event_id = db_id
            item.trace.stamp_deduped()

        try:
            deduped_queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("Deduped queue full; dropping item %s/%s", item.source, item.source_id)
