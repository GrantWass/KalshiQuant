"""
db/repositories/news.py — NewsEvent database operations.
"""

import json
from uuid import UUID
import asyncpg
from db.pool import get_pool


async def insert_news_event(
    source: str,
    source_id: str,
    headline: str,
    published_at,
    fetched_at,
    body: str | None = None,
    url: str | None = None,
    event_score: float | None = None,
    keyword_score: float | None = None,
    nlp_score: float | None = None,
    filtered_out: bool = False,
    filter_reason: str | None = None,
    raw_payload: dict | None = None,
) -> UUID | None:
    """Insert a news event. Returns the new UUID or None on duplicate."""
    pool = await get_pool()
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO news_events
                (source, source_id, headline, body, url, published_at, fetched_at,
                 event_score, keyword_score, nlp_score, filtered_out, filter_reason, raw_payload)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (source, source_id) DO NOTHING
            RETURNING id
            """,
            source, source_id, headline, body, url,
            published_at, fetched_at,
            event_score, keyword_score, nlp_score,
            filtered_out, filter_reason,
            json.dumps(raw_payload or {}),
        )
        return row["id"] if row else None
    except asyncpg.PostgresError:
        return None


async def update_news_event_scores(
    event_id: UUID,
    event_score: float,
    keyword_score: float,
    nlp_score: float,
    filtered_out: bool,
    filter_reason: str | None,
) -> None:
    """Update scores after event detection (item was inserted before scoring)."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE news_events
        SET event_score=$2, keyword_score=$3, nlp_score=$4,
            filtered_out=$5, filter_reason=$6
        WHERE id=$1
        """,
        event_id, event_score, keyword_score, nlp_score, filtered_out, filter_reason,
    )


async def fetch_recent_news(limit: int = 100) -> list[dict]:
    """Fetch the most recent news events for the dashboard."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, source, headline, url, published_at, fetched_at,
               event_score, keyword_score, nlp_score, filtered_out, filter_reason
        FROM news_events
        ORDER BY fetched_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]
