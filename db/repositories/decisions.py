"""
db/repositories/decisions.py — TradeDecision database operations.
"""

from uuid import UUID
import asyncpg
from db.pool import get_pool


async def insert_trade_decision(
    news_event_id: UUID,
    market_ticker: str,
    market_title: str,
    action: str,           # 'EXECUTE' or 'REJECT'
    edge: float,
    p_market: float,
    p_estimated: float,
    confidence: float,
    side: str | None = None,
    contracts: int | None = None,
    price_cents: int | None = None,
    rejection_reasons: list[str] | None = None,
    kalshi_order_id: str | None = None,
    kalshi_status: str | None = None,
) -> UUID:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO trade_decisions
            (news_event_id, market_ticker, market_title, action, side, contracts,
             price_cents, edge, p_market, p_estimated, confidence,
             rejection_reasons, kalshi_order_id, kalshi_status)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
        RETURNING id
        """,
        news_event_id, market_ticker, market_title, action, side, contracts,
        price_cents, edge, p_market, p_estimated, confidence,
        rejection_reasons, kalshi_order_id, kalshi_status,
    )
    return row["id"]


async def fetch_recent_decisions(limit: int = 200) -> list[dict]:
    """Fetch recent decisions for the dashboard."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT id, market_ticker, market_title, action, side, contracts, price_cents,
               edge, p_market, p_estimated, confidence, rejection_reasons,
               kalshi_order_id, decided_at
        FROM trade_decisions
        ORDER BY decided_at DESC
        LIMIT $1
        """,
        limit,
    )
    return [dict(r) for r in rows]


async def fetch_daily_counts() -> dict:
    """Return today's executed and rejected counts for the dashboard summary."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE action = 'EXECUTE') AS executed,
            COUNT(*) FILTER (WHERE action = 'REJECT')  AS rejected
        FROM trade_decisions
        WHERE decided_at >= CURRENT_DATE
        """
    )
    return dict(row)
