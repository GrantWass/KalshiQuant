"""
dashboard/db_queries.py — All database read queries for the Streamlit dashboard.

Uses synchronous psycopg2 (not asyncpg) because Streamlit's execution model
is synchronous. All queries are read-only — the dashboard never writes to DB.

The dashboard reads exclusively from PostgreSQL and never touches the trading
pipeline's in-memory state, ensuring zero coupling between display and execution.
"""

import os
import warnings
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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
            id,
            fetched_at AT TIME ZONE 'UTC' AS fetched_at,
            source,
            headline,
            body,
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
        SELECT
            source,
            COUNT(*)                                                      AS total,
            SUM(CASE WHEN NOT filtered_out THEN 1 ELSE 0 END)            AS passed,
            SUM(CASE WHEN filtered_out THEN 1 ELSE 0 END)                AS filtered,
            ROUND(
                100.0 * SUM(CASE WHEN NOT filtered_out THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 1
            )                                                             AS pass_rate
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
            id,
            fetched_at AT TIME ZONE 'UTC' AS fetched_at,
            source,
            headline,
            body,
            url,
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

def get_events_with_matches(limit: int = 100) -> pd.DataFrame:
    """One row per news event that reached the market matcher, with aggregate match stats."""
    return _query_df(
        """
        SELECT
            ne.id                                               AS news_event_id,
            ne.fetched_at AT TIME ZONE 'UTC'                   AS fetched_at,
            ne.source,
            ne.headline,
            ne.url,
            ROUND(ne.event_score::numeric, 3)                  AS event_score,
            COUNT(mm.id)                                        AS match_count,
            ROUND(MAX(mm.similarity_score)::numeric, 3)        AS best_similarity,
            ROUND(AVG(mm.similarity_score)::numeric, 3)        AS avg_similarity
        FROM news_events ne
        JOIN market_matches mm ON mm.news_event_id = ne.id
        WHERE ne.filtered_out = false
          AND mm.below_threshold = false
        GROUP BY ne.id, ne.fetched_at, ne.source, ne.headline, ne.url, ne.event_score
        ORDER BY ne.fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_matches_for_event(news_event_id: str) -> pd.DataFrame:
    """All market matches for a specific news event, sorted by similarity descending."""
    return _query_df(
        """
        SELECT
            mm.market_ticker,
            mm.market_title,
            ROUND(mm.similarity_score::numeric, 3)  AS similarity_score,
            td.action                                AS decision_action,
            td.side                                  AS decision_side,
            td.contracts                             AS decision_contracts,
            td.price_cents                           AS decision_price_cents,
            ROUND(td.edge::numeric, 3)               AS decision_edge,
            ROUND(td.confidence::numeric, 3)         AS decision_confidence,
            td.rejection_reasons
        FROM market_matches mm
        LEFT JOIN trade_decisions td ON td.market_ticker = mm.market_ticker
            AND td.decided_at BETWEEN mm.matched_at AND mm.matched_at + INTERVAL '30 seconds'
        WHERE mm.news_event_id = %s
        ORDER BY mm.similarity_score DESC
        """,
        (news_event_id,),
    )


def get_near_miss_events(limit: int = 100) -> pd.DataFrame:
    """Events that passed detection but had no matches above the similarity threshold."""
    return _query_df(
        """
        SELECT
            ne.id                                               AS news_event_id,
            ne.fetched_at AT TIME ZONE 'UTC'                   AS fetched_at,
            ne.source,
            ne.headline,
            ne.url,
            ROUND(ne.event_score::numeric, 3)                  AS event_score,
            ROUND(MAX(mm.similarity_score)::numeric, 3)        AS best_similarity
        FROM news_events ne
        JOIN market_matches mm ON mm.news_event_id = ne.id AND mm.below_threshold = true
        WHERE ne.filtered_out = false
          AND ne.id NOT IN (
              SELECT DISTINCT news_event_id FROM market_matches WHERE below_threshold = false
          )
        GROUP BY ne.id, ne.fetched_at, ne.source, ne.headline, ne.url, ne.event_score
        ORDER BY ne.fetched_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_near_misses_for_event(news_event_id: str) -> pd.DataFrame:
    """Top near-miss markets for an event that didn't pass the similarity threshold."""
    return _query_df(
        """
        SELECT
            mm.market_ticker,
            mm.market_title,
            ROUND(mm.similarity_score::numeric, 3) AS similarity_score
        FROM market_matches mm
        WHERE mm.news_event_id = %s
          AND mm.below_threshold = true
        ORDER BY mm.similarity_score DESC
        """,
        (news_event_id,),
    )


def get_recent_matches(limit: int = 100) -> pd.DataFrame:
    return _query_df(
        """
        SELECT
            mm.id,
            mm.matched_at AT TIME ZONE 'UTC' AS matched_at,
            ne.source,
            ne.headline,
            ne.body,
            ne.url,
            mm.market_ticker,
            mm.market_title,
            ROUND(mm.similarity_score::numeric, 3) AS similarity_score,
            td.action          AS decision_action,
            td.side            AS decision_side,
            td.contracts       AS decision_contracts,
            td.price_cents     AS decision_price_cents,
            ROUND(td.edge::numeric, 3)        AS decision_edge,
            ROUND(td.confidence::numeric, 3)  AS decision_confidence,
            td.rejection_reasons
        FROM market_matches mm
        JOIN news_events ne ON ne.id = mm.news_event_id
        LEFT JOIN trade_decisions td ON td.market_ticker = mm.market_ticker
            AND td.decided_at BETWEEN mm.matched_at AND mm.matched_at + INTERVAL '30 seconds'
        ORDER BY mm.matched_at DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_pipeline_funnel() -> dict:
    """Today's counts at each pipeline stage for the overview page."""
    df = _query_df(
        """
        SELECT
            COUNT(*)                                                    AS news_total,
            COUNT(*) FILTER (WHERE NOT filtered_out AND event_score IS NOT NULL) AS events_passed,
            COUNT(*) FILTER (WHERE filtered_out)                        AS news_filtered
        FROM news_events
        WHERE fetched_at >= CURRENT_DATE
        """
    )
    matches = _query_df(
        "SELECT COUNT(*) AS matches FROM market_matches WHERE matched_at >= CURRENT_DATE"
    )
    decisions = _query_df(
        """
        SELECT
            COUNT(*) FILTER (WHERE action = 'EXECUTE') AS executed,
            COUNT(*) FILTER (WHERE action = 'REJECT')  AS rejected
        FROM trade_decisions WHERE decided_at >= CURRENT_DATE
        """
    )
    result = {}
    if not df.empty:
        result.update(df.iloc[0].to_dict())
    if not matches.empty:
        result["matches"] = int(matches.iloc[0]["matches"])
    if not decisions.empty:
        result.update(decisions.iloc[0].to_dict())
    return result


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
