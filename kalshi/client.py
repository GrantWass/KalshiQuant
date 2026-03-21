"""
kalshi/client.py — Async Kalshi REST API client.

Handles:
  - Authentication headers (via KalshiTokenManager)
  - Rate limiting (asyncio.Semaphore: max concurrent requests)
  - Automatic retry with exponential backoff on 429 (rate limited) and 503 (server error)
  - Automatic token refresh on 401, then one retry

Usage:
    client = KalshiClient()
    markets = await client.get_markets(status="open")
    order = await client.place_order(OrderRequest(...))
"""

import asyncio
import logging
import aiohttp
from config.settings import settings
from kalshi.auth import token_manager
from kalshi.models import Market, OrderRequest, OrderResponse, Orderbook, Portfolio, Position

logger = logging.getLogger(__name__)

# Maximum number of markets fetched per page from the Kalshi API.
_PAGE_SIZE = 100

# Delays (seconds) for exponential backoff: attempt 0 → 1s, 1 → 2s, 2 → 4s
_RETRY_DELAYS = [1, 2, 4]


class KalshiClient:
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        # Semaphore limits concurrent in-flight requests to avoid rate limiting.
        self._semaphore = asyncio.Semaphore(settings.KALSHI_MAX_CONCURRENT_REQUESTS)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=settings.KALSHI_BASE_URL,
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        retry_on_401: bool = True,
    ) -> dict:
        """
        Make an authenticated API request with retry logic.

        Retry policy:
          - 429 or 503: exponential backoff (up to 3 attempts)
          - 401: force token refresh and retry once
          - Other 4xx: raise immediately (client error, no retry)
        """
        session = await self._get_session()

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)

            token = await token_manager.get_valid_token()
            headers = {"Authorization": f"Bearer {token}"}

            async with self._semaphore:
                try:
                    async with session.request(
                        method, path, params=params, json=json, headers=headers
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()

                        if resp.status == 401 and retry_on_401:
                            logger.warning("Received 401; forcing token refresh.")
                            await token_manager.force_refresh()
                            # Retry once with the new token (set retry_on_401=False to avoid loop)
                            return await self._request(
                                method, path, params=params, json=json, retry_on_401=False
                            )

                        if resp.status in (429, 503):
                            logger.warning(
                                "Received %d from Kalshi API (attempt %d); retrying in %ds",
                                resp.status, attempt + 1, _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)],
                            )
                            continue  # exponential backoff loop

                        body = await resp.text()
                        raise RuntimeError(
                            f"Kalshi API {method} {path} returned {resp.status}: {body}"
                        )

                except aiohttp.ClientError as exc:
                    if attempt < len(_RETRY_DELAYS):
                        logger.warning("Request failed (%s); retrying.", exc)
                        continue
                    raise

        raise RuntimeError(f"Kalshi API {method} {path} failed after all retries.")

    # ── Market Data ────────────────────────────────────────────────────────────

    async def get_markets(
        self,
        status: str = "open",
        cursor: str | None = None,
        limit: int = _PAGE_SIZE,
    ) -> tuple[list[Market], str | None]:
        """
        Fetch a page of markets. Returns (markets, next_cursor).
        Pass next_cursor into the next call to paginate.
        """
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/markets", params=params)
        markets = [Market(**m) for m in data.get("markets", [])]
        next_cursor = data.get("cursor")
        return markets, next_cursor

    async def get_all_open_markets(self) -> list[Market]:
        """Paginate through all open markets and return the full list."""
        all_markets: list[Market] = []
        cursor = None
        while True:
            page, cursor = await self.get_markets(status="open", cursor=cursor)
            all_markets.extend(page)
            if not cursor or not page:
                break
        logger.info("Fetched %d open markets from Kalshi.", len(all_markets))
        return all_markets

    async def get_market(self, ticker: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self._request("GET", f"/markets/{ticker}")
        return Market(**data.get("market", data))

    async def get_orderbook(self, ticker: str) -> Orderbook:
        """Fetch the current order book for a market."""
        data = await self._request("GET", f"/markets/{ticker}/orderbook")
        return Orderbook(ticker=ticker, **data.get("orderbook", data))

    # ── Trading ────────────────────────────────────────────────────────────────

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place a limit order. Returns the order response from Kalshi.
        In DRY_RUN mode, logs the order but does not call the API.
        """
        if settings.DRY_RUN:
            logger.info(
                "[DRY RUN] Would place order: %s %s %d contracts @ %d cents",
                order.side.upper(), order.ticker, order.count, order.yes_price,
            )
            return OrderResponse(
                order_id="dry-run",
                ticker=order.ticker,
                status="dry_run",
                side=order.side,
                action=order.action,
                count=order.count,
                yes_price=order.yes_price,
            )

        data = await self._request(
            "POST",
            "/portfolio/orders",
            json=order.model_dump(),
        )
        return OrderResponse(**data.get("order", data))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successful."""
        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except RuntimeError as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    # ── Portfolio ──────────────────────────────────────────────────────────────

    async def get_portfolio(self) -> Portfolio:
        """Fetch portfolio summary (balance, total value)."""
        data = await self._request("GET", "/portfolio/balance")
        return Portfolio(**data)

    async def get_positions(self) -> list[Position]:
        """Fetch all open positions."""
        data = await self._request("GET", "/portfolio/positions")
        return [Position(**p) for p in data.get("market_positions", [])]
