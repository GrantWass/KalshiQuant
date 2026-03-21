"""
db/pool.py — asyncpg connection pool singleton.

Usage:
    from db.pool import get_pool, init_pool, close_pool

    await init_pool()             # call once at startup
    pool = await get_pool()       # get the shared pool
    async with pool.acquire() as conn:
        await conn.fetch("SELECT 1")
    await close_pool()            # call once at shutdown
"""

import asyncpg
from config.settings import settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    """Create the module-level connection pool. Call once at application startup."""
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        command_timeout=30,
    )
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool, initializing it if needed."""
    global _pool
    if _pool is None:
        await init_pool()
    return _pool


async def close_pool() -> None:
    """Gracefully close all connections. Call once at application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
