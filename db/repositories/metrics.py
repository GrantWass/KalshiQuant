"""
db/repositories/metrics.py — Pipeline latency metrics.
"""

from datetime import datetime
from uuid import UUID
from db.pool import get_pool


def _ms(t_start: datetime | None, t_end: datetime | None) -> float | None:
    """Compute milliseconds between two timestamps. Returns None if either is None."""
    if t_start is None or t_end is None:
        return None
    return (t_end - t_start).total_seconds() * 1000


async def insert_latency_metric(
    source: str,
    t_fetched: datetime,
    t_deduped: datetime | None = None,
    t_detected: datetime | None = None,
    t_matched: datetime | None = None,
    t_estimated: datetime | None = None,
    t_decided: datetime | None = None,
    t_executed: datetime | None = None,
    news_event_id: UUID | None = None,
) -> None:
    """Insert a pipeline trace record with computed stage durations."""
    pool = await get_pool()

    # Compute end timestamp for total (prefer executed, fall back to decided)
    t_end = t_executed or t_decided

    await pool.execute(
        """
        INSERT INTO latency_metrics
            (news_event_id, source,
             t_fetched, t_deduped, t_detected, t_matched, t_estimated, t_decided, t_executed,
             ms_dedup, ms_detect, ms_match, ms_estimate, ms_decide, ms_execute, ms_total)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
        """,
        news_event_id, source,
        t_fetched, t_deduped, t_detected, t_matched, t_estimated, t_decided, t_executed,
        _ms(t_fetched, t_deduped),
        _ms(t_deduped, t_detected),
        _ms(t_detected, t_matched),
        _ms(t_matched, t_estimated),
        _ms(t_estimated, t_decided),
        _ms(t_decided, t_executed),
        _ms(t_fetched, t_end),
    )


async def fetch_recent_latency(limit: int = 200) -> list[dict]:
    """Fetch recent latency records for the dashboard."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT source, t_fetched, ms_dedup, ms_detect, ms_match, ms_estimate,
               ms_decide, ms_execute, ms_total
        FROM latency_metrics
        ORDER BY t_fetched DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]


async def fetch_latency_percentiles() -> dict:
    """Fetch P50/P95/P99 total latency for the dashboard summary."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ms_total) AS p50,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ms_total) AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ms_total) AS p99
        FROM latency_metrics
        WHERE t_fetched >= NOW() - INTERVAL '1 hour'
          AND ms_total IS NOT NULL
        """
    )
    return dict(row) if row else {"p50": None, "p95": None, "p99": None}


async def insert_market_match(
    news_event_id: UUID,
    market_ticker: str,
    market_title: str,
    market_category: str | None,
    similarity_score: float,
    below_threshold: bool = False,
) -> None:
    """Record a FAISS market match result."""
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO market_matches
            (news_event_id, market_ticker, market_title, market_category, similarity_score, below_threshold)
        VALUES ($1,$2,$3,$4,$5,$6)
        """,
        news_event_id, market_ticker, market_title, market_category, similarity_score, below_threshold,
    )


async def fetch_recent_matches(limit: int = 100) -> list[dict]:
    """Fetch recent market match results for the dashboard."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT mm.news_event_id, ne.headline, ne.source,
               mm.market_ticker, mm.market_title, mm.market_category,
               mm.similarity_score, mm.matched_at
        FROM market_matches mm
        JOIN news_events ne ON ne.id = mm.news_event_id
        ORDER BY mm.matched_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]


async def update_pipeline_heartbeat(
    run_id: UUID,
    news_fetched: int,
    events_detected: int,
    markets_matched: int,
    trades_executed: int,
    trades_rejected: int,
) -> None:
    """Update the pipeline heartbeat record."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE pipeline_runs
        SET last_heartbeat=$2, news_fetched=$3, events_detected=$4,
            markets_matched=$5, trades_executed=$6, trades_rejected=$7
        WHERE id=$1
        """,
        run_id, datetime.utcnow(),
        news_fetched, events_detected, markets_matched, trades_executed, trades_rejected,
    )


async def insert_pipeline_run() -> UUID:
    """Create a new pipeline run record at startup."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "INSERT INTO pipeline_runs DEFAULT VALUES RETURNING id"
    )
    return row["id"]
