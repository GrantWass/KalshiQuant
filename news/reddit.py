"""
news/reddit.py — Reddit signal ingestion via the public JSON API.

Reddit's self-service OAuth app registration was discontinued in Nov 2025.
The public .json endpoint still works without credentials for read-only access.

Polls hot posts from Kalshi-relevant subreddits. The key requirements for
the public API are a descriptive User-Agent and staying under ~1 req/sec.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

from config.settings import settings
from news.base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

_SUBREDDITS = [
    "politics",
    "economics",
    "investing",
    "worldnews",
    "wallstreetbets",
    "geopolitics",
]

# Hot posts have had hours to accumulate votes — higher bar makes sense.
# New posts are brand new — lower bar since score hasn't had time to grow.
_MIN_SCORE_HOT = 100
_MIN_SCORE_NEW = 10

# Seconds between each API request — unauthenticated Reddit rate limit is ~1 req/2s.
# 12 requests per cycle (6 subreddits × hot + new) × 2s = 24s total, well within limits.
_REQUEST_DELAY = 2.5


class RedditSource(NewsSource):
    """Polls hot posts from financial/political subreddits via Reddit public JSON API."""

    def __init__(self) -> None:
        super().__init__(poll_interval_seconds=settings.REDDIT_POLL_INTERVAL)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    # Reddit public API requires a specific User-Agent format
                    "User-Agent": "python:kalshiquant:v1.0 (by /u/kalshiquant)",
                    "Accept": "application/json",
                }
            )
        return self._session

    async def fetch(self) -> AsyncIterator[NewsItem]:
        session = await self._get_session()
        for subreddit in _SUBREDDITS:
            for feed in ("hot", "new"):
                min_score = _MIN_SCORE_HOT if feed == "hot" else _MIN_SCORE_NEW
                items = await self._fetch_subreddit(session, subreddit, feed, min_score)
                for item in items:
                    yield item
                await asyncio.sleep(_REQUEST_DELAY)

    async def _fetch_subreddit(
        self, session: aiohttp.ClientSession, subreddit: str, feed: str, min_score: int
    ) -> list[NewsItem]:
        url = f"https://www.reddit.com/r/{subreddit}/{feed}.json?limit=25"
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 429:
                    logger.warning("Reddit rate limited on r/%s/%s", subreddit, feed)
                    return []
                if resp.status == 403:
                    logger.warning("Reddit r/%s returned 403 — subreddit may be private", subreddit)
                    return []
                if resp.status != 200:
                    logger.warning("Reddit r/%s/%s returned %d", subreddit, feed, resp.status)
                    return []
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning("Reddit r/%s/%s fetch failed: %s", subreddit, feed, exc)
            return []

        items = []
        for child in data.get("data", {}).get("children", []):
            item = self._parse_post(subreddit, feed, min_score, child.get("data", {}))
            if item is not None:
                items.append(item)
        return items

    def _clean_title(self, title: str) -> str:
        title = re.sub(r"^\s*\[[^\]]{1,30}\]\s*", "", title)
        title = re.sub(r"\s*\[[^\]]{1,30}\]\s*$", "", title)
        return title.strip()

    def _parse_post(self, subreddit: str, feed: str, min_score: int, post: dict) -> NewsItem | None:
        post_id = post.get("id")
        if not post_id:
            return None
        if post.get("score", 0) < min_score:
            return None
        if post.get("is_self"):
            return None

        headline = self._clean_title(post.get("title", ""))
        if not headline:
            return None

        source_id = f"reddit_{post_id}"
        published_at = datetime.fromtimestamp(
            post.get("created_utc", 0), tz=timezone.utc
        ).replace(tzinfo=None)

        return NewsItem(
            source=f"reddit_{subreddit}",
            source_id=source_id,
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            url=post.get("url"),
            raw_payload={
                "subreddit": subreddit,
                "feed": feed,          # "hot" or "new" — indicates recency vs popularity
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "post_id": post_id,
            },
        )
