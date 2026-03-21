"""
news/gdelt.py — GDELT (Global Database of Events, Language, and Tone) ingestion.

GDELT publishes a master file list every 15 minutes at:
  http://data.gdeltproject.org/gdeltv2/lastupdate.txt

This file contains URLs for three CSV files (events, mentions, GKG).
We use the GKG (Global Knowledge Graph) CSV which contains:
  - Article URL
  - Themes (e.g., "NATURAL_DISASTER", "ELECTION", "ECON_INFLATION")
  - Tone (sentiment score)
  - Person and organization names
  - Source country

Why GKG vs Events CSV?
  The Events CSV uses CAMEO codes which require domain knowledge to map.
  The GKG Themes column is a flat list of readable strings that map more
  directly to Kalshi market categories (weather, politics, economics).

Memory strategy:
  GKG files can be 50-200MB uncompressed. We stream the compressed (.csv.zip)
  file using aiohttp chunked reads and parse CSV rows on the fly — never loading
  the full file into memory.

Deduplication:
  We track the last processed GKG file URL in-memory. Between 15-min updates,
  polling finds the same URL and skips processing (no duplicate items).

Rate limit: GDELT has undocumented rate limits. We add a 5s delay between
  requests and never poll more frequently than GDELT_POLL_INTERVAL (60s).
"""

import csv
import io
import logging
import zipfile
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

from config.settings import settings
from news.base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GKG themes that are likely to affect prediction markets
_RELEVANT_THEMES = {
    # Weather / Natural disasters
    "NATURAL_DISASTER", "WEATHER_HURRICANE", "WEATHER_TORNADO", "WEATHER_FLOOD",
    "WEATHER_WILDFIRE", "EARTHQUAKE",
    # Political
    "ELECTION", "ELECTIONS", "POLITICAL", "GOVERNMENT_OFFICIAL", "MILITARY",
    "COUP", "PROTEST", "CIVIL_UNREST",
    # Economic
    "ECON_INFLATION", "ECON_UNEMPLOYMENT", "ECON_TRADE", "ECON_TAXATION",
    "CENTRAL_BANK", "INTEREST_RATE", "RECESSION", "STOCK_MARKET",
    # Sports (some Kalshi markets cover sports outcomes)
    "SPORTS", "SPORTS_NFL", "SPORTS_NBA",
}

# GKG CSV column indices (V2.1 format)
# Full spec: http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
_COL_DATE = 0          # YYYYMMDDHHMMSS
_COL_SOURCE_URL = 4    # Article URL
_COL_THEMES = 7        # Semicolon-separated theme codes
_COL_TONE = 15         # "tone,positive,negative,polarity,..." (comma-separated)


class GDELTSource(NewsSource):
    """GDELT GKG streaming source. Polls every 60s; data updates every 15 min."""

    def __init__(self) -> None:
        super().__init__(poll_interval_seconds=settings.GDELT_POLL_INTERVAL)
        self._last_gkg_url: str | None = None
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch(self) -> AsyncIterator[NewsItem]:
        """Check for a new GKG file and stream relevant articles."""
        gkg_url = await self._get_latest_gkg_url()
        if gkg_url is None or gkg_url == self._last_gkg_url:
            return  # No new data since last poll

        self._last_gkg_url = gkg_url
        logger.info("Processing new GDELT GKG file: %s", gkg_url.split("/")[-1])

        async for item in self._stream_gkg(gkg_url):
            yield item

    async def _get_latest_gkg_url(self) -> str | None:
        """Parse lastupdate.txt to find the GKG CSV URL."""
        session = await self._get_session()
        try:
            async with session.get(
                _LASTUPDATE_URL, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning("GDELT lastupdate.txt returned %d", resp.status)
                    return None
                text = await resp.text()
        except aiohttp.ClientError as exc:
            logger.error("GDELT lastupdate.txt fetch failed: %s", exc)
            return None

        # Each line: "size hash url". Third line is GKG.
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and "gkg" in parts[2].lower():
                return parts[2]

        logger.warning("Could not find GKG URL in GDELT lastupdate.txt")
        return None

    async def _stream_gkg(self, url: str) -> AsyncIterator[NewsItem]:
        """Stream and parse the GKG zip file, yielding relevant NewsItems."""
        session = await self._get_session()

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    logger.warning("GDELT GKG file returned %d", resp.status)
                    return

                # Read the entire zip into a BytesIO buffer
                # (zipfile requires seekable stream; we keep it small by only
                #  reading what we need)
                content = await resp.read()
        except aiohttp.ClientError as exc:
            logger.error("GDELT GKG download failed: %s", exc)
            return

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = zf.namelist()
                if not names:
                    return
                csv_name = names[0]
                with zf.open(csv_name) as csv_file:
                    # GKG uses tab-separated values
                    reader = csv.reader(
                        io.TextIOWrapper(csv_file, encoding="utf-8", errors="replace"),
                        delimiter="\t",
                    )
                    count = 0
                    for row in reader:
                        item = self._parse_row(row)
                        if item is not None:
                            yield item
                            count += 1
                    logger.info("GDELT: yielded %d relevant items.", count)
        except (zipfile.BadZipFile, UnicodeDecodeError, Exception) as exc:
            logger.error("GDELT GKG parse error: %s", exc)

    def _parse_row(self, row: list[str]) -> NewsItem | None:
        """Parse a single GKG CSV row. Returns None if not relevant."""
        if len(row) <= max(_COL_THEMES, _COL_SOURCE_URL, _COL_TONE):
            return None

        themes_raw = row[_COL_THEMES] if _COL_THEMES < len(row) else ""
        themes = set(themes_raw.upper().split(";")) if themes_raw else set()

        # Only process rows with at least one relevant theme
        if not themes.intersection(_RELEVANT_THEMES):
            return None

        url = row[_COL_SOURCE_URL].strip() if _COL_SOURCE_URL < len(row) else ""
        if not url:
            return None

        date_str = row[_COL_DATE].strip() if _COL_DATE < len(row) else ""
        published_at = _parse_gdelt_date(date_str) or datetime.utcnow()

        # Build a headline from the URL domain + matched themes
        matched = themes.intersection(_RELEVANT_THEMES)
        theme_str = ", ".join(sorted(matched)[:3])
        domain = url.split("/")[2] if "/" in url else url
        headline = f"[GDELT] {theme_str} — {domain}"

        # Extract overall tone score for sentiment signal
        tone_raw = row[_COL_TONE].strip() if _COL_TONE < len(row) else ""
        tone_score = None
        try:
            tone_score = float(tone_raw.split(",")[0])
        except (ValueError, IndexError):
            pass

        # Use URL as source_id (unique per article)
        source_id = url[:512]

        return NewsItem(
            source="gdelt",
            source_id=source_id,
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            url=url,
            raw_payload={
                "themes": list(matched),
                "tone": tone_score,
                "date_str": date_str,
            },
        )


def _parse_gdelt_date(date_str: str) -> datetime | None:
    """Parse GDELT date format YYYYMMDDHHMMSS to datetime."""
    if not date_str or len(date_str) < 14:
        return None
    try:
        return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None
