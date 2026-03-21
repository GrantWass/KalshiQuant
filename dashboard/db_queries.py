"""
dashboard/db_queries.py — All database read queries for the Streamlit dashboard.

Uses synchronous psycopg2 (not asyncpg) because Streamlit's execution model
is synchronous. All queries are read-only — the dashboard never writes to DB.

The dashboard reads exclusively from PostgreSQL and never touches the trading
pipeline's in-memory state, ensuring zero coupling between display and execution.
"""

import os
import pandas as pd
import psycopg2
import psycopg2.extras
from functools import lru_cache

# Read DATABASE_URL from environment (same as the pipeline)
_DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://kalshi:kalshi@localhost:5432/kalshiquant")


def _get_connection():
    """Create a new psycopg2 connection. Called per query (no pooling needed for dashboard)."""
    return psycopg2.connect(_DATABASE_URL)


def _query_df(sql: str, params=None) -> pd.DataFrame:
    """Execute a SQL query and return a pandas DataFrame. Returns empty DataFrame on error."""
    try:
        conn = _get_connection()
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── News Feed ─────────────────────────────────────────────────────────────────

def get_recent_news(limit: int = 200) -> pd.DataFrame:
    return _query_df(
        """
        SELECT
            fetched_at AT TIME ZONE 'UTC' AS fetched_at,
            source,
            headline,
            url,
            ROUND(event_score::numeric, 3)   AS event_score,
            ROUND(keyword_score::numeric, 3) AS keyword_score,
            ROUND(nlp_score::numeric, 3)     AS nlp_score,
            filtered_out,
            filter_reason
        FROM news_events
        ORDER BY fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_news_source_counts() -> pd.DataFrame:
    return _query_df(
        """
        SELECT source,
               COUNT(*) AS total,
               SUM(CASE WHEN filtered_out THEN 0 ELSE 1 END) AS passed,
               SUM(CASE WHEN filtered_out THEN 1 ELSE 0 END) AS filtered
        FROM news_events
        WHERE fetched_at >= NOW() - INTERVAL '24 hours'
        GROUP BY source
        ORDER BY total DESC
        """
    )


# ── Event Detection ───────────────────────────────────────────────────────────

def get_score_distribution() -> pd.DataFrame:
    """Return event scores bucketed into 0.1-width bins for histogram."""
    return _query_df(
        """
        SELECT
            ROUND((event_score / 0.1)::numeric, 0) * 0.1 AS score_bucket,
            COUNT(*) AS count
        FROM news_events
        WHERE event_score IS NOT NULL
          AND fetched_at >= NOW() - INTERVAL '24 hours'
        GROUP BY score_bucket
        ORDER BY score_bucket
        """
    )


def get_detection_detail(limit: int = 100) -> pd.DataFrame:
    """Detailed scores for items that were scored (not duped before detection)."""
    return _query_df(
        """
        SELECT
            fetched_at AT TIME ZONE 'UTC' AS fetched_at,
            source,
            headline,
            ROUND(keyword_score::numeric, 3) AS keyword_score,
            ROUND(nlp_score::numeric, 3)     AS nlp_score,
            ROUND(event_score::numeric, 3)   AS event_score,
            filtered_out,
            filter_reason
        FROM news_events
        WHERE event_score IS NOT NULL
        ORDER BY fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )


# ── Market Matches ────────────────────────────────────────────────────────────

def get_recent_matches(limit: int = 100) -> pd.DataFrame:
    return _query_df(
        """
        SELECT
            mm.matched_at AT TIME ZONE 'UTC' AS matched_at,
            ne.source,
            ne.headline,
            mm.market_ticker,
            mm.market_title,
            mm.market_category,
            ROUND(mm.similarity_score::numeric, 3) AS similarity_score
        FROM market_matches mm
        JOIN news_events ne ON ne.id = mm.news_event_id
        ORDER BY mm.matched_at DESC
        LIMIT %s
        """,
        (limit,),
    )


# ── Trade Decisions ───────────────────────────────────────────────────────────

def get_recent_decisions(limit: int = 200) -> pd.DataFrame:
    return _query_df(
        """
        SELECT
            decided_at AT TIME ZONE 'UTC' AS decided_at,
            action,
            market_ticker,
            market_title,
            side,
            contracts,
            price_cents,
            ROUND(edge::numeric, 3)        AS edge,
            ROUND(p_market::numeric, 3)    AS p_market,
            ROUND(p_estimated::numeric, 3) AS p_estimated,
            ROUND(confidence::numeric, 3)  AS confidence,
            rejection_reasons,
            kalshi_order_id
        FROM trade_decisions
        ORDER BY decided_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_daily_decision_summary() -> dict:
    """Counts of executed vs rejected decisions today."""
    df = _query_df(
        """
        SELECT
            COUNT(*) FILTER (WHERE action = 'EXECUTE') AS executed,
            COUNT(*) FILTER (WHERE action = 'REJECT')  AS rejected
        FROM trade_decisions
        WHERE decided_at >= CURRENT_DATE
        """
    )
    if df.empty:
        return {"executed": 0, "rejected": 0}
    return df.iloc[0].to_dict()


def get_rejection_reason_counts() -> pd.DataFrame:
    """Unnest rejection reasons to count which gates fire most often."""
    return _query_df(
        """
        SELECT
            UNNEST(rejection_reasons) AS reason,
            COUNT(*) AS count
        FROM trade_decisions
        WHERE action = 'REJECT'
          AND decided_at >= NOW() - INTERVAL '24 hours'
        GROUP BY reason
        ORDER BY count DESC
        LIMIT 20
        """
    )


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions() -> pd.DataFrame:
    return _query_df(
        """
        SELECT
            market_ticker,
            market_title,
            side,
            contracts,
            ROUND(avg_price_cents::numeric / 100, 2) AS avg_price,
            ROUND(current_price_cents::numeric / 100, 2) AS current_price,
            ROUND(unrealized_pnl_cents::numeric / 100, 2) AS unrealized_pnl,
            ROUND(realized_pnl_cents::numeric / 100, 2) AS realized_pnl,
            last_updated AT TIME ZONE 'UTC' AS last_updated
        FROM positions
        ORDER BY last_updated DESC
        """
    )


def get_portfolio_summary() -> dict:
    df = _query_df(
        """
        SELECT
            COALESCE(SUM(contracts * avg_price_cents / 100.0), 0)        AS total_exposure_usd,
            COALESCE(SUM(unrealized_pnl_cents / 100.0), 0)               AS total_unrealized_pnl,
            COALESCE(SUM(realized_pnl_cents / 100.0), 0)                 AS total_realized_pnl,
            COUNT(*)                                                       AS open_positions
        FROM positions
        """
    )
    if df.empty:
        return {"total_exposure_usd": 0, "total_unrealized_pnl": 0, "total_realized_pnl": 0, "open_positions": 0}
    return df.iloc[0].to_dict()


# ── Latency Metrics ───────────────────────────────────────────────────────────

def get_recent_latency(limit: int = 200) -> pd.DataFrame:
    return _query_df(
        """
        SELECT
            t_fetched AT TIME ZONE 'UTC' AS t_fetched,
            source,
            ROUND(ms_dedup::numeric, 1)     AS ms_dedup,
            ROUND(ms_detect::numeric, 1)    AS ms_detect,
            ROUND(ms_match::numeric, 1)     AS ms_match,
            ROUND(ms_estimate::numeric, 1)  AS ms_estimate,
            ROUND(ms_decide::numeric, 1)    AS ms_decide,
            ROUND(ms_execute::numeric, 1)   AS ms_execute,
            ROUND(ms_total::numeric, 1)     AS ms_total
        FROM latency_metrics
        WHERE ms_total IS NOT NULL
        ORDER BY t_fetched DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_latency_percentiles() -> dict:
    df = _query_df(
        """
        SELECT
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ms_total)::numeric, 0) AS p50,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ms_total)::numeric, 0) AS p95,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ms_total)::numeric, 0) AS p99
        FROM latency_metrics
        WHERE t_fetched >= NOW() - INTERVAL '1 hour'
          AND ms_total IS NOT NULL
        """
    )
    if df.empty or df.iloc[0].isnull().all():
        return {"p50": None, "p95": None, "p99": None}
    return df.iloc[0].to_dict()


def get_stage_averages() -> pd.DataFrame:
    """Average time per pipeline stage for the bar chart."""
    return _query_df(
        """
        SELECT
            'Dedup'    AS stage, ROUND(AVG(ms_dedup)::numeric, 1)    AS avg_ms FROM latency_metrics WHERE t_fetched >= NOW() - INTERVAL '1 hour'
        UNION ALL
        SELECT 'Detect',   ROUND(AVG(ms_detect)::numeric, 1)   FROM latency_metrics WHERE t_fetched >= NOW() - INTERVAL '1 hour'
        UNION ALL
        SELECT 'Match',    ROUND(AVG(ms_match)::numeric, 1)    FROM latency_metrics WHERE t_fetched >= NOW() - INTERVAL '1 hour'
        UNION ALL
        SELECT 'Estimate', ROUND(AVG(ms_estimate)::numeric, 1) FROM latency_metrics WHERE t_fetched >= NOW() - INTERVAL '1 hour'
        UNION ALL
        SELECT 'Decide',   ROUND(AVG(ms_decide)::numeric, 1)   FROM latency_metrics WHERE t_fetched >= NOW() - INTERVAL '1 hour'
        UNION ALL
        SELECT 'Execute',  ROUND(AVG(ms_execute)::numeric, 1)  FROM latency_metrics WHERE t_fetched >= NOW() - INTERVAL '1 hour'
        """
    )


def get_pipeline_health() -> dict:
    """Check if the pipeline is alive (last heartbeat < 2 minutes ago)."""
    df = _query_df(
        """
        SELECT
            healthy,
            last_heartbeat AT TIME ZONE 'UTC' AS last_heartbeat,
            news_fetched, events_detected, markets_matched,
            trades_executed, trades_rejected
        FROM pipeline_runs
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    if df.empty:
        return {"healthy": False, "last_heartbeat": None}
    return df.iloc[0].to_dict()
