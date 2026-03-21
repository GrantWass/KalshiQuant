"""
news/nws.py — National Weather Service (NWS) alert ingestion.

Source: https://api.weather.gov/alerts/active
  - Free, no authentication required
  - Returns real-time active weather alerts as GeoJSON
  - Polled every 30 seconds (NWS_POLL_INTERVAL)
  - Best latency of all sources — alerts appear within seconds of issuance

Relevant alert types for Kalshi markets (hurricane, tornado, winter weather markets):
  "Hurricane Warning", "Hurricane Watch",
  "Tornado Warning", "Tornado Watch",
  "Blizzard Warning", "Winter Storm Warning",
  "Extreme Wind Warning", "Tropical Storm Warning"

Items are deduped by (source, source_id) where source_id = alert["id"] from the API.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

from config.settings import settings
from news.base import NewsItem, NewsSource

logger = logging.getLogger(__name__)

# NWS API endpoint — no auth required, no rate limit documented
_NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"

# Alert event types we care about (affects prediction markets)
_RELEVANT_EVENT_TYPES = {
    "Hurricane Warning",
    "Hurricane Watch",
    "Tornado Warning",
    "Tornado Watch",
    "Blizzard Warning",
    "Winter Storm Warning",
    "Winter Storm Watch",
    "Extreme Wind Warning",
    "Tropical Storm Warning",
    "Tropical Storm Watch",
    "Tsunami Warning",
    "Earthquake Warning",
    "Flash Flood Emergency",
    "Tornado Emergency",
}


class NWSSource(NewsSource):
    """National Weather Service real-time alert source."""

    def __init__(self) -> None:
        super().__init__(poll_interval_seconds=settings.NWS_POLL_INTERVAL)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    # NWS requires a User-Agent to identify the application
                    "User-Agent": "KalshiQuant/1.0 (automated trading system; contact@example.com)",
                    "Accept": "application/geo+json",
                }
            )
        return self._session

    async def fetch(self) -> AsyncIterator[NewsItem]:
        """Fetch active NWS alerts and yield relevant ones as NewsItems."""
        session = await self._get_session()
        params = {
            "status": "actual",
            "message_type": "alert",
            "urgency": "Immediate,Expected",
        }

        try:
            async with session.get(
                _NWS_ALERTS_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning("NWS API returned %d", resp.status)
                    return

                data = await resp.json()
                features = data.get("features", [])

        except aiohttp.ClientError as exc:
            logger.error("NWS fetch failed: %s", exc)
            return

        for feature in features:
            item = self._parse_feature(feature)
            if item is not None:
                yield item

    def _parse_feature(self, feature: dict) -> NewsItem | None:
        """Parse a GeoJSON feature into a NewsItem. Returns None if not relevant."""
        props = feature.get("properties", {})
        event_type = props.get("event", "")

        # Only process alerts for event types that can move prediction markets
        if event_type not in _RELEVANT_EVENT_TYPES:
            return None

        alert_id = feature.get("id", "")
        if not alert_id:
            return None

        # Build a descriptive headline from the alert properties
        area = props.get("areaDesc", "")
        severity = props.get("severity", "")
        headline = props.get("headline") or f"{event_type} — {area}"

        # Body: combine description and instruction fields
        description = props.get("description", "") or ""
        instruction = props.get("instruction", "") or ""
        body = "\n\n".join(filter(None, [description, instruction])) or None

        # Parse effective time (when the alert was issued)
        effective_str = props.get("effective") or props.get("onset") or props.get("sent")
        published_at = _parse_nws_time(effective_str) or datetime.now(timezone.utc)

        # Use alert URL as the link (NWS alerts have stable URLs)
        url = f"https://api.weather.gov/alerts/{alert_id.split('/')[-1]}" if alert_id else None

        return NewsItem(
            source="nws",
            source_id=alert_id,
            headline=headline,
            published_at=published_at,
            fetched_at=datetime.utcnow(),
            body=body,
            url=url,
            raw_payload={
                "id": alert_id,
                "event": event_type,
                "severity": severity,
                "area": area,
                "effective": effective_str,
            },
        )


def _parse_nws_time(time_str: str | None) -> datetime | None:
    """Parse NWS ISO 8601 timestamp to datetime (UTC)."""
    if not time_str:
        return None
    try:
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)  # store as naive UTC
    except (ValueError, AttributeError):
        return None
