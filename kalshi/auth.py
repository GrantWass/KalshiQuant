"""
kalshi/auth.py — Kalshi token manager.

Kalshi auth tokens expire after 30 minutes. This module manages proactive
token refresh to ensure no in-flight trade ever hits an expired token.

Design:
  - KalshiTokenManager is a module-level singleton.
  - A background asyncio task refreshes the token every TOKEN_TTL_SECONDS (1700s = 28 min).
  - All callers use get_valid_token(), which is instant when the token is fresh.
  - On any 401 response from the API, call force_refresh() to re-authenticate immediately.
  - An asyncio.Lock prevents concurrent refresh races (e.g., two requests both
    hitting 401 at the same time would both try to re-authenticate).

Usage:
    token_manager = KalshiTokenManager()
    await token_manager.start()              # start background refresh loop

    token = await token_manager.get_valid_token()
    headers = {"Authorization": f"Bearer {token}"}

    # If a request returns 401:
    await token_manager.force_refresh()
    token = await token_manager.get_valid_token()
"""

import asyncio
import logging
import time
import aiohttp
from config.settings import settings

logger = logging.getLogger(__name__)


class KalshiTokenManager:
    def __init__(self) -> None:
        self._token: str | None = None
        self._refreshed_at: float = 0.0     # monotonic timestamp of last successful refresh
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Authenticate immediately and start the background refresh loop."""
        self._session = aiohttp.ClientSession()
        await self._authenticate()
        asyncio.create_task(self._refresh_loop(), name="kalshi_token_refresh")

    async def stop(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()

    async def get_valid_token(self) -> str:
        """
        Return the current valid token.
        Blocks briefly if a refresh is in progress (asyncio.Lock).
        Raises RuntimeError if no token has been obtained yet.
        """
        async with self._lock:
            if self._token is None:
                raise RuntimeError("Token manager not started — call start() first.")
            return self._token

    async def force_refresh(self) -> None:
        """
        Re-authenticate immediately. Called by the API client on 401 responses.
        The lock prevents a thundering herd if multiple concurrent requests all
        receive 401 at the same time.
        """
        async with self._lock:
            # Another coroutine may have already refreshed by the time we acquire the lock.
            # Only re-authenticate if the token hasn't been refreshed in the last 5 seconds.
            if time.monotonic() - self._refreshed_at > 5.0:
                logger.warning("Forcing token refresh after 401 response.")
                await self._authenticate()

    async def _refresh_loop(self) -> None:
        """Background task: sleep TOKEN_TTL_SECONDS then re-authenticate, forever."""
        while True:
            await asyncio.sleep(settings.KALSHI_TOKEN_TTL_SECONDS)
            try:
                async with self._lock:
                    await self._authenticate()
                logger.info("Kalshi token proactively refreshed.")
            except Exception as exc:
                logger.error("Token refresh failed: %s — will retry in 60s", exc)
                await asyncio.sleep(60)

    async def _authenticate(self) -> None:
        """
        POST /log_in to obtain a new token.
        Must be called while holding self._lock (or before the loop starts).
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = f"{settings.KALSHI_BASE_URL}/log_in"
        payload = {
            "email": settings.KALSHI_EMAIL,
            "password": settings.KALSHI_PASSWORD,
        }

        try:
            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Kalshi login returned {resp.status}: {body}")
                data = await resp.json()
                self._token = data.get("token") or data.get("access_token")
                if not self._token:
                    raise RuntimeError(f"No token in login response: {data}")
                self._refreshed_at = time.monotonic()
                logger.debug("Kalshi token obtained successfully.")
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Kalshi login request failed: {exc}") from exc


# Module-level singleton shared by KalshiClient and KalshiWebSocketManager.
token_manager = KalshiTokenManager()
