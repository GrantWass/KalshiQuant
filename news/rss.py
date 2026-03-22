"""
news/rss.py — RSS feed ingestion (AP News, BBC).

Polls multiple RSS feeds concurrently and yields NewsItem objects.
Articles typically appear 15-30 minutes after publication, so this source
is best suited for less time-sensitive markets (elections, economics).

Libraries:
  - feedparser: parses RSS/Atom, handles malformed feeds gracefully
  - aiohttp: async HTTP for concurrent feed fetches

Source IDs for deduplication:
  RSS entries have a `link` or `id` field we use as the source_id.
  Feedparser normalizes this across RSS 1.0/2.0/Atom.
"""

import hashlib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import AsyncIterator

import aiohttp
import feedparser

from config.settings import settings
from news.base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

# Free RSS feeds — no authentication, no rate limits
# Add more feeds here or in .env to expand coverage
_FEEDS = [
    ("rss_bbc", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("rss_bbc", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("rss_bbc", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("rss_bbc", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
    ("rss_bbc", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ("rss_npr", "https://www.npr.org/rss/rss.php?id=1001"),   # NPR News
    ("rss_npr", "https://www.npr.org/rss/rss.php?id=1003"),   # NPR Politics
]


class RSSSource(NewsSource):
    """Polls multiple RSS feeds concurrently every RSS_POLL_INTERVAL seconds."""

    def __init__(self) -> None:
        super().__init__(poll_interval_seconds=settings.RSS_POLL_INTERVAL)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "KalshiQuant/1.0 RSS reader"}
            )
        return self._session

    async def fetch(self) -> AsyncIterator[NewsItem]:
        """Fetch all RSS feeds concurrently and yield items."""
        import asyncio

        session = await self._get_session()

        # Fetch all feeds concurrently
        tasks = [self._fetch_feed(session, source, url) for source, url in _FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("RSS feed error: %s", result)
                continue
            for item in result:
                yield item

    async def _fetch_feed(
        self, session: aiohttp.ClientSession, source: str, url: str
    ) -> list[NewsItem]:
        """Fetch a single RSS feed URL and parse its entries."""
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.warning("RSS feed %s returned %d", url, resp.status)
                    return []
                content = await resp.text()
        except aiohttp.ClientError as exc:
            logger.warning("RSS feed %s failed: %s", url, exc)
            return []

        feed = feedparser.parse(content)
        items = []
        for entry in feed.entries:
            item = self._parse_entry(source, entry)
            if item:
                items.append(item)
        return items

    def _parse_entry(self, source: str, entry) -> NewsItem | None:
        """Parse a feedparser entry into a NewsItem."""
        # Use the entry ID or link as the dedup key
        source_id = (
            getattr(entry, "id", None)
            or getattr(entry, "link", None)
        )
        if not source_id:
            return None
        # Truncate to fit DB column
        source_id = source_id[:512]

        headline = getattr(entry, "title", None) or ""
        if not headline:
            return None

        url = getattr(entry, "link", None)

        # Body: prefer summary, fall back to content
        body = None
        if hasattr(entry, "summary"):
            body = entry.summary[:2000]  # truncate to keep DB rows manageable
        elif hasattr(entry, "content") and entry.content:
            body = entry.content[0].get("value", "")[:2000]

        published_at = _parse_rss_date(entry) or datetime.utcnow()

        return NewsItem(
            source=source,
            source_id=source_id,
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            body=body,
            url=url,
            raw_payload={"feed_source": source},
        )


def _parse_rss_date(entry) -> datetime | None:
    """Extract and parse the publication date from a feedparser entry."""
    # feedparser normalizes dates into published_parsed (time.struct_time)
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            import time
            ts = time.mktime(entry.published_parsed)
            return datetime.utcfromtimestamp(ts)
        except (OverflowError, ValueError):
            pass

    # Fallback: parse the raw published string
    if hasattr(entry, "published") and entry.published:
        try:
            dt = parsedate_to_datetime(entry.published)
            return dt.replace(tzinfo=None)
        except Exception:
            pass

    return None
