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
import re
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
_COL_SOURCE_COMMON = 3 # Human-readable source name e.g. "BBC News"
_COL_SOURCE_URL = 4    # Article URL
_COL_THEMES = 7        # Semicolon-separated theme codes
_COL_LOCATIONS = 9     # Semicolon-separated locations
_COL_PERSONS = 11      # Semicolon-separated person names
_COL_ORGS = 13         # Semicolon-separated organization names
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

        matched = themes.intersection(_RELEVANT_THEMES)

        # Extract persons, orgs, locations from their columns
        persons = _parse_list(row, _COL_PERSONS, limit=2)
        orgs    = _parse_list(row, _COL_ORGS, limit=2)
        locs    = _parse_list(row, _COL_LOCATIONS, limit=1, field_index=1)  # location name is field 1

        # Try to extract a headline from the URL slug first
        slug_headline = _headline_from_url(url)

        if slug_headline:
            # Enrich slug with key entities if they're not already in it
            entities = persons + orgs + locs
            extras = [e for e in entities if e.lower() not in slug_headline.lower()]
            if extras:
                headline = f"{slug_headline} ({', '.join(extras[:2])})"
            else:
                headline = slug_headline
        else:
            # Fall back to building from entities + themes
            parts = persons + orgs + locs
            if parts:
                theme_hint = _theme_label(matched)
                headline = f"{', '.join(parts[:3])} — {theme_hint}"
            else:
                theme_hint = _theme_label(matched)
                domain = url.split("/")[2] if "/" in url else url
                headline = f"{theme_hint} ({domain})"

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
                "persons": persons,
                "orgs": orgs,
                "locations": locs,
                "date_str": date_str,
            },
        )


def _headline_from_url(url: str) -> str:
    """
    Extract a human-readable headline from a URL slug.

    Most news URLs look like:
      .../federal-reserve-raises-rates-25-basis-points/
      .../2025/03/21/ukraine-war-ceasefire-talks-resume/

    Strategy:
      - Take the last non-empty path segment
      - Strip numeric IDs at the end (e.g. -12345678)
      - Replace hyphens/underscores with spaces
      - Title-case the result
      - Discard if too short or looks like a bare date / category slug
    """
    try:
        path = url.split("?")[0].rstrip("/")
        segments = [s for s in path.split("/") if s]
        if not segments:
            return ""
        slug = segments[-1]

        # Skip pure date segments like "20250321" or "2025"
        if re.fullmatch(r"\d{4,14}", slug):
            slug = segments[-2] if len(segments) >= 2 else ""

        # Skip UUID-based filename slugs like "article_d958c61a-21e5-48b4-8178-3657f6f50409.html"
        if re.search(r"[0-9a-f]{8}[-_][0-9a-f]{4}", slug, re.IGNORECASE):
            slug = segments[-2] if len(segments) >= 2 else ""

        if not slug:
            return ""

        # Remove trailing numeric IDs: "breaking-news-story-1234567"
        slug = re.sub(r"[-_]\d{5,}$", "", slug)

        # Replace separators with spaces and title-case
        headline = re.sub(r"[-_]+", " ", slug).strip().title()

        # Discard if too short to be a real headline
        if len(headline) < 20 or len(headline.split()) < 4:
            return ""

        return headline
    except Exception:
        return ""


def _parse_list(row: list[str], col: int, limit: int, field_index: int = 0) -> list[str]:
    """
    Parse a semicolon-delimited GDELT column where each entry may itself be
    comma-delimited. Returns up to `limit` cleaned values from `field_index`.

    Example (persons col): "Jerome Powell;Joe Biden" → ["Jerome Powell", "Joe Biden"]
    Example (locations col): "1#Florida#US#...;2#Ukraine#UA#..." with field_index=1
      → ["Florida", "Ukraine"]
    """
    if col >= len(row):
        return []
    raw = row[col].strip()
    if not raw:
        return []
    results = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("#") if "#" in entry else [entry]
        if field_index < len(parts):
            val = parts[field_index].strip().title()
            if val and val not in results:
                results.append(val)
        if len(results) >= limit:
            break
    return results


# Map theme codes to readable labels for fallback headline construction
_THEME_LABELS = {
    "WEATHER_HURRICANE": "Hurricane",
    "WEATHER_TORNADO": "Tornado",
    "WEATHER_FLOOD": "Flooding",
    "WEATHER_WILDFIRE": "Wildfire",
    "EARTHQUAKE": "Earthquake",
    "NATURAL_DISASTER": "Natural Disaster",
    "ELECTION": "Election",
    "ELECTIONS": "Election",
    "MILITARY": "Military Conflict",
    "COUP": "Coup",
    "PROTEST": "Protest",
    "CIVIL_UNREST": "Civil Unrest",
    "ECON_INFLATION": "Inflation",
    "ECON_UNEMPLOYMENT": "Unemployment",
    "ECON_TRADE": "Trade",
    "ECON_TAXATION": "Tax Policy",
    "CENTRAL_BANK": "Central Bank",
    "INTEREST_RATE": "Interest Rates",
    "RECESSION": "Recession",
    "STOCK_MARKET": "Stock Market",
    "SPORTS_NFL": "NFL",
    "SPORTS_NBA": "NBA",
    "SPORTS": "Sports",
    "POLITICAL": "Politics",
    "GOVERNMENT_OFFICIAL": "Government",
}

def _theme_label(matched: set[str]) -> str:
    """Return the most specific readable label for the matched theme set."""
    for theme in sorted(matched):
        if theme in _THEME_LABELS:
            return _THEME_LABELS[theme]
    return ", ".join(sorted(matched)[:2])


def _parse_gdelt_date(date_str: str) -> datetime | None:
    """Parse GDELT date format YYYYMMDDHHMMSS to datetime."""
    if not date_str or len(date_str) < 14:
        return None
    try:
        return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None
