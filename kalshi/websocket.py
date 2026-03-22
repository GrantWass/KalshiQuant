"""
kalshi/websocket.py — Kalshi WebSocket feed for real-time market price updates.

Maintains a live dict of {ticker: price_cents} updated continuously from
the Kalshi WebSocket feed. The ProbabilityEstimator reads from this dict
for zero-latency price lookups (vs. ~200ms REST roundtrip).

Features:
  - Automatic reconnection with exponential backoff on disconnect
  - Re-subscription to all tickers after reconnect
  - Graceful shutdown via stop()

Usage:
    ws_manager = KalshiWebSocketManager()
    await ws_manager.start()

    # Subscribe to markets you want live prices for
    await ws_manager.subscribe(["TICKER-A", "TICKER-B"])

    # Read prices anywhere in the codebase
    price = ws_manager.get_price("TICKER-A")   # returns int cents or None
"""

import asyncio
import json
import logging
import time
import aiohttp
from config.settings import settings
from kalshi.auth import signer

logger = logging.getLogger(__name__)

# Backoff delays (seconds) for reconnect attempts: 1s, 2s, 4s, 8s, 16s, 30s cap
_BACKOFF = [1, 2, 4, 8, 16, 30]


class KalshiWebSocketManager:
    def __init__(self) -> None:
        # Shared price state: {ticker: last_yes_price_cents}
        # Read by ProbabilityEstimator (lock-free; worst case reads a stale value,
        # which is acceptable — REST fallback handles truly missing prices).
        self._prices: dict[str, int] = {}

        # Tickers we are currently subscribed to (persisted across reconnects)
        self._subscribed: set[str] = set()

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._msg_seq = 0   # incrementing sequence number for WebSocket messages

    async def start(self) -> None:
        """Connect and start the background message loop."""
        # WebSocket disabled until correct URL path is confirmed from Kalshi docs.
        # ProbabilityEstimator falls back to REST for price lookups automatically.
        logger.info("Kalshi WebSocket disabled — using REST fallback for prices.")
        return

    async def stop(self) -> None:
        """Gracefully disconnect."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()

    def get_price(self, ticker: str) -> int | None:
        """
        Return the latest YES price in cents for a ticker, or None if not available.
        Thread-safe: reads from a plain dict (no lock needed for single-process asyncio).
        """
        return self._prices.get(ticker)

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to real-time price updates for a list of market tickers."""
        new_tickers = set(tickers) - self._subscribed
        if not new_tickers:
            return

        self._subscribed.update(new_tickers)

        if self._ws and not self._ws.closed:
            await self._send_subscribe(list(new_tickers))

    async def _send_subscribe(self, tickers: list[str]) -> None:
        """Send a subscription message for the given tickers."""
        if not self._ws or self._ws.closed:
            return
        self._msg_seq += 1
        msg = {
            "id": self._msg_seq,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": tickers,
            },
        }
        await self._ws.send_str(json.dumps(msg))

    async def _connect_loop(self) -> None:
        """
        Main connection loop. Reconnects with exponential backoff on failure.
        Re-subscribes to all previously subscribed tickers after reconnect.
        """
        attempt = 0
        while self._running:
            try:
                await self._connect_and_run()
                attempt = 0  # reset backoff on clean disconnect
            except Exception as exc:
                if not self._running:
                    break
                delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                logger.warning(
                    "WebSocket disconnected (%s); reconnecting in %ds (attempt %d).",
                    exc, delay, attempt + 1,
                )
                attempt += 1
                await asyncio.sleep(delay)

    async def _connect_and_run(self) -> None:
        """Open the WebSocket connection and process messages until disconnect."""
        ws_path = "/trade-api/v2/ws"
        auth = signer.get_auth_headers("GET", ws_path)
        # Pass auth as both headers and query params — Kalshi WS may require either
        url = (
            f"{settings.KALSHI_WS_URL}"
            f"?access_key={auth['KALSHI-ACCESS-KEY']}"
            f"&access_timestamp={auth['KALSHI-ACCESS-TIMESTAMP']}"
            f"&access_signature={auth['KALSHI-ACCESS-SIGNATURE']}"
        )

        async with self._session.ws_connect(url, headers=auth) as ws:
            self._ws = ws
            logger.info("Kalshi WebSocket connected.")

            # Re-subscribe to all tickers (handles reconnect case)
            if self._subscribed:
                await self._send_subscribe(list(self._subscribed))

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._on_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.info("WebSocket closed by server.")
                    break

    async def _on_message(self, raw: str) -> None:
        """Parse an incoming WebSocket message and update the price cache."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")

        # Ticker update: {type: "ticker", msg: {market_ticker: "...", yes_bid: 43, yes_ask: 45, ...}}
        if msg_type == "ticker":
            msg = data.get("msg", {})
            ticker = msg.get("market_ticker")
            if not ticker:
                return

            # Use mid-price (average of best bid and ask) as the reference price.
            yes_bid = msg.get("yes_bid")
            yes_ask = msg.get("yes_ask")
            last_price = msg.get("last_price")

            if yes_bid is not None and yes_ask is not None:
                self._prices[ticker] = (yes_bid + yes_ask) // 2
            elif last_price is not None:
                self._prices[ticker] = last_price

        # Subscription confirmation or error (log only)
        elif msg_type in ("subscribed", "error"):
            logger.debug("WebSocket message: %s", data)
