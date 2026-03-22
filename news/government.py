"""
news/government.py — Government API signal ingestion.

Three sub-sources, all free:

1. FRED (Federal Reserve Economic Data)
   - Monitors key economic series for new data releases
   - Generates a headline when CPI, unemployment, GDP, or Fed rate data drops
   - Requires FRED_API_KEY (free at fred.stlouisfed.org/docs/api/api_key.html)

2. SEC EDGAR
   - Polls recent 8-K filings (earnings surprises, major corporate events)
   - No authentication required
   - Generates headlines for filings from S&P 500 companies

3. Congress.gov
   - Polls recently updated bills in the House and Senate
   - Generates headlines for bills that passed a vote or were signed into law
   - Requires CONGRESS_API_KEY (free at api.congress.gov)

Each sub-source is polled on GOVERNMENT_POLL_INTERVAL (default 5 minutes).
Sources with missing API keys are skipped silently at startup.
"""

import logging
import re as _re
import time as _time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import aiohttp
import feedparser

from config.settings import settings
from news.base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

# ── FRED configuration ────────────────────────────────────────────────────────
# Key economic series to monitor. When FRED posts a new observation, we emit
# a headline. These are the series that most directly move Kalshi markets.
_FRED_SERIES = {
    "CPIAUCSL":  "CPI Inflation Data Released",
    "UNRATE":    "Unemployment Rate Data Released",
    "GDP":       "GDP Growth Data Released",
    "FEDFUNDS":  "Federal Funds Rate Updated",
    "DGS10":     "10-Year Treasury Yield Updated",
    "ICSA":      "Weekly Jobless Claims Released",
}

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── SEC EDGAR configuration ───────────────────────────────────────────────────
_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22%22&dateRange=custom&startdt={date}&forms=8-K&_source=hits.hits._source.period_of_report,hits.hits._source.entity_name,hits.hits._source.file_date,hits.hits._source.period_of_report&hits.hits.total.value=true"

# ── Congress.gov configuration ────────────────────────────────────────────────
_CONGRESS_BASE = "https://api.congress.gov/v3/bill"

# Bill actions that are worth generating a headline for
_SIGNIFICANT_ACTIONS = {
    "Became Public Law",
    "Signed by President",
    "Passed Senate",
    "Passed House",
    "Cloture Motion Presented",
    "Failed of Passage",
    "Vetoed by President",
}


class GovernmentSource(NewsSource):
    """Polls FRED, SEC EDGAR, and Congress.gov for market-moving government data."""

    def __init__(self) -> None:
        super().__init__(poll_interval_seconds=settings.GOVERNMENT_POLL_INTERVAL)
        self._session: aiohttp.ClientSession | None = None
        # Track last-seen observation dates per FRED series to avoid re-emitting
        self._fred_last_seen: dict[str, str] = {}
        # Track last-seen SEC filing accession numbers
        self._edgar_seen: set[str] = set()
        # Track last-seen Congress bill numbers
        self._congress_seen: set[str] = set()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    # SEC requires a descriptive User-Agent with contact info
                    "User-Agent": "KalshiQuant automated-trading-system contact@kalshiquant.com",
                    "Accept-Encoding": "gzip, deflate",
                }
            )
        return self._session

    async def fetch(self) -> AsyncIterator[NewsItem]:
        session = await self._get_session()

        if settings.FRED_API_KEY:
            async for item in self._fetch_fred(session):
                yield item
        else:
            logger.debug("FRED_API_KEY not set — skipping FRED source")

        async for item in self._fetch_edgar(session):
            yield item

        if settings.CONGRESS_API_KEY:
            async for item in self._fetch_congress(session):
                yield item
        else:
            logger.debug("CONGRESS_API_KEY not set — skipping Congress.gov source")

    # ── FRED ──────────────────────────────────────────────────────────────────

    async def _fetch_fred(self, session: aiohttp.ClientSession) -> AsyncIterator[NewsItem]:
        for series_id, label in _FRED_SERIES.items():
            item = await self._fetch_fred_series(session, series_id, label)
            if item:
                yield item

    async def _fetch_fred_series(
        self, session: aiohttp.ClientSession, series_id: str, label: str
    ) -> NewsItem | None:
        params = {
            "series_id":      series_id,
            "api_key":        settings.FRED_API_KEY,
            "file_type":      "json",
            "sort_order":     "desc",
            "limit":          1,
            "observation_start": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
        }
        try:
            async with session.get(
                _FRED_BASE, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning("FRED %s returned %d", series_id, resp.status)
                    return None
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning("FRED %s fetch failed: %s", series_id, exc)
            return None

        observations = data.get("observations", [])
        if not observations:
            return None

        latest = observations[0]
        obs_date = latest.get("date", "")
        value = latest.get("value", "")

        # Only emit if this is a new observation we haven't seen before
        if self._fred_last_seen.get(series_id) == obs_date:
            return None
        self._fred_last_seen[series_id] = obs_date

        # Skip placeholder values FRED uses when data isn't yet available
        if value == ".":
            return None

        headline = f"{label}: {value} (as of {obs_date})"
        source_id = f"fred_{series_id}_{obs_date}"

        try:
            published_at = datetime.strptime(obs_date, "%Y-%m-%d")
        except ValueError:
            published_at = datetime.utcnow()

        return NewsItem(
            source="fred",
            source_id=source_id,
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            url=f"https://fred.stlouisfed.org/series/{series_id}",
            raw_payload={"series_id": series_id, "value": value, "date": obs_date},
        )

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────

    async def _fetch_edgar(self, session: aiohttp.ClientSession) -> AsyncIterator[NewsItem]:
        # Official EDGAR RSS feed — the supported mechanism for recent filing ingestion
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            "?action=getcurrent&type=8-K&dateb=&owner=include&count=40&output=atom"
        )
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning("SEC EDGAR RSS returned %d", resp.status)
                    return
                content = await resp.text()
        except aiohttp.ClientError as exc:
            logger.warning("SEC EDGAR RSS fetch failed: %s", exc)
            return

        feed = feedparser.parse(content)
        for entry in feed.entries:
            item = self._parse_edgar_entry(entry)
            if item:
                yield item

    def _parse_edgar_entry(self, entry) -> NewsItem | None:
        link = getattr(entry, "link", "") or ""
        if not link or link in self._edgar_seen:
            return None
        self._edgar_seen.add(link)

        # Title format from EDGAR RSS: "8-K - COMPANY NAME"
        raw_title = getattr(entry, "title", "") or ""
        if not raw_title:
            return None

        # Strip leading "8-K - " or "8-K/A - " prefix
        clean_title = _re.sub(r"^8-K[/A]*\s*-\s*", "", raw_title).strip()
        entity = clean_title.split(" (")[0].strip() if " (" in clean_title else clean_title
        form = "8-K"
        if "8-K" in raw_title:
            form = "8-K/A" if "8-K/A" in raw_title else "8-K"

        headline = f"{entity} filed {form} with SEC"
        source_id = f"edgar_{link}"

        published_at = datetime.utcnow()
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published_at = datetime.utcfromtimestamp(_time.mktime(entry.published_parsed))
            except (OverflowError, ValueError):
                pass

        return NewsItem(
            source="sec_edgar",
            source_id=source_id[:512],
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            url=link,
            raw_payload={"entity": entity, "form": form, "raw_title": raw_title},
        )

    # ── Congress.gov ──────────────────────────────────────────────────────────

    async def _fetch_congress(self, session: aiohttp.ClientSession) -> AsyncIterator[NewsItem]:
        params = {
            "api_key":   settings.CONGRESS_API_KEY,
            "sort":      "updateDate+desc",
            "limit":     20,
            "format":    "json",
        }
        try:
            async with session.get(
                _CONGRESS_BASE, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning("Congress.gov returned %d", resp.status)
                    return
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning("Congress.gov fetch failed: %s", exc)
            return

        for bill in data.get("bills", []):
            item = self._parse_bill(bill)
            if item:
                yield item

    def _parse_bill(self, bill: dict) -> NewsItem | None:
        bill_number = f"{bill.get('type', '')}{bill.get('number', '')}"
        congress = bill.get("congress", "")
        unique_id = f"{congress}_{bill_number}"

        if unique_id in self._congress_seen:
            return None

        latest_action = bill.get("latestAction", {})
        action_text = latest_action.get("text", "")

        # Only emit for significant legislative actions
        if not any(sig in action_text for sig in _SIGNIFICANT_ACTIONS):
            return None

        self._congress_seen.add(unique_id)

        title = bill.get("title", "Untitled Bill")
        headline = f"{bill_number} ({congress}th Congress): {action_text} — {title[:80]}"
        source_id = f"congress_{unique_id}"

        action_date_str = latest_action.get("actionDate", "")
        try:
            published_at = datetime.strptime(action_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            published_at = datetime.utcnow()

        url = bill.get("url", "")

        return NewsItem(
            source="congress",
            source_id=source_id,
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            url=url,
            raw_payload={
                "bill_number": bill_number,
                "congress": congress,
                "action": action_text,
                "title": title,
            },
        )
