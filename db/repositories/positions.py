"""
db/repositories/positions.py — Position tracking and P&L.
"""

from db.pool import get_pool


async def upsert_position(
    market_ticker: str,
    market_title: str,
    side: str,
    contracts: int,
    avg_price_cents: float,
) -> None:
    """Insert or update a position. Called after each successful order."""
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO positions (market_ticker, market_title, side, contracts, avg_price_cents)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (market_ticker) DO UPDATE SET
            side = EXCLUDED.side,
            contracts = positions.contracts + EXCLUDED.contracts,
            avg_price_cents = (
                (positions.avg_price_cents * positions.contracts +
                 EXCLUDED.avg_price_cents * EXCLUDED.contracts)
                / NULLIF(positions.contracts + EXCLUDED.contracts, 0)
            ),
            last_updated = NOW()
        """,
        market_ticker, market_title, side, contracts, avg_price_cents,
    )


async def update_current_price(market_ticker: str, current_price_cents: float) -> None:
    """Update the mark-to-market price from the WebSocket feed."""
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE positions
        SET current_price_cents = $2,
            unrealized_pnl_cents = ($2 - avg_price_cents) * contracts,
            last_updated = NOW()
        WHERE market_ticker = $1
        """,
        market_ticker, current_price_cents,
    )


async def get_position_contracts(market_ticker: str) -> int:
    """Return number of contracts held in a market (0 if no position)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT contracts FROM positions WHERE market_ticker = $1",
        market_ticker,
    )
    return row["contracts"] if row else 0


async def get_total_exposure_usd() -> float:
    """Return total portfolio exposure: sum(contracts × avg_price_cents / 100)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT COALESCE(SUM(contracts * avg_price_cents / 100.0), 0) AS total FROM positions"
    )
    return float(row["total"])


async def fetch_all_positions() -> list[dict]:
    """Fetch all positions for the dashboard."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT market_ticker, market_title, side, contracts, avg_price_cents,
               current_price_cents, unrealized_pnl_cents, realized_pnl_cents, last_updated
        FROM positions
        ORDER BY last_updated DESC
        """
    )
    return [dict(r) for r in rows]
