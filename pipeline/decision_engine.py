"""
pipeline/decision_engine.py — Final trading decision gate.

Evaluates each trade candidate against 5 risk gates. ALL decisions —
both EXECUTE and REJECT — are written to the trade_decisions table
so the dashboard can show exactly why every trade was or was not placed.

The 5 gates (checked in order):

  Gate 1: Minimum edge
    abs(edge) >= PROBABILITY_SHIFT_MIN (default 0.04 = 4 cents)
    If the edge is too small, expected value doesn't cover fees + slippage.

  Gate 2: Confidence
    confidence >= CONFIDENCE_MIN (default 0.60)
    Requires the harmonic mean of event_score × similarity_score × recency
    to be above threshold. Rejects trades where either the news or the
    market match is weakly supported.

  Gate 3: Per-market position limit
    current_contracts < MAX_POSITION_PER_MARKET (default 50)
    Prevents over-concentration in a single market.

  Gate 4: Total portfolio exposure
    total_exposure_usd + order_cost < MAX_TOTAL_EXPOSURE_USD (default $500)
    Hard cap on total dollars at risk across all markets.

  Gate 5: Market close time
    minutes_to_close > MIN_MINUTES_TO_CLOSE (default 30)
    Don't trade markets that close soon — thin orderbook, insufficient time
    to realize the edge, and hard to exit if wrong.

Order sizing (Kelly fraction):
  contracts = int(kelly_fraction * bankroll_contracts * KELLY_FRACTION)
  kelly_fraction = edge / (1 - p_market)  # simplified Kelly for binary bets
  Capped at: MAX_POSITION_PER_MARKET - current_contracts
  Quarter-Kelly (KELLY_FRACTION = 0.25) used for conservative sizing.
"""

import asyncio
import logging
from datetime import datetime

from config.settings import settings
from db.repositories.decisions import insert_trade_decision
from db.repositories.metrics import insert_latency_metric
from db.repositories.positions import (
    get_position_contracts,
    get_total_exposure_usd,
    upsert_position,
)
from kalshi.client import KalshiClient
from kalshi.models import OrderRequest
from pipeline.probability_estimator import TradeCandidate

logger = logging.getLogger(__name__)

# Bankroll in contracts used for Kelly sizing.
# At $500 max exposure and ~$0.50 avg price, this is ~1000 contracts total.
# Quarter-Kelly with this bankroll gives reasonable order sizes.
_BANKROLL_CONTRACTS = 1000


class DecisionEngine:
    """
    Evaluates trade candidates through 5 gates and executes approved orders.
    """

    def __init__(self, kalshi_client: KalshiClient) -> None:
        self._client = kalshi_client

    async def run(self, candidates_queue: asyncio.Queue) -> None:
        """Consume TradeCandidates, evaluate gates, execute or reject."""
        while True:
            candidate: TradeCandidate = await candidates_queue.get()
            try:
                await self._evaluate(candidate)
            except Exception as exc:
                logger.error("DecisionEngine error for %s: %s", candidate.ticker, exc)
            finally:
                candidates_queue.task_done()

    async def _evaluate(self, candidate: TradeCandidate) -> None:
        """
        Run all 5 gates. Record the decision to DB regardless of outcome.
        Execute the order if all gates pass.
        """
        rejection_reasons: list[str] = []

        # ── Gate 1: Minimum edge ───────────────────────────────────────────────
        if abs(candidate.edge) < settings.PROBABILITY_SHIFT_MIN:
            rejection_reasons.append(
                f"Edge {candidate.edge:+.3f} < minimum {settings.PROBABILITY_SHIFT_MIN:.3f}"
            )

        # ── Gate 2: Confidence ────────────────────────────────────────────────
        if candidate.confidence < settings.CONFIDENCE_MIN:
            rejection_reasons.append(
                f"Confidence {candidate.confidence:.3f} < minimum {settings.CONFIDENCE_MIN:.3f}"
            )

        # ── Gate 3: Per-market position limit ────────────────────────────────
        current_contracts = await get_position_contracts(candidate.ticker)
        if current_contracts >= settings.MAX_POSITION_PER_MARKET:
            rejection_reasons.append(
                f"Position limit reached: {current_contracts}/{settings.MAX_POSITION_PER_MARKET} contracts"
            )

        # ── Gate 4: Total portfolio exposure ─────────────────────────────────
        # Compute order cost based on the side we're buying
        price_for_order = candidate.p_market
        order_contracts = self._size_order(candidate, current_contracts)
        order_cost_usd = order_contracts * price_for_order

        total_exposure = await get_total_exposure_usd()
        if total_exposure + order_cost_usd > settings.MAX_TOTAL_EXPOSURE_USD:
            rejection_reasons.append(
                f"Exposure limit: ${total_exposure:.2f} + ${order_cost_usd:.2f} > ${settings.MAX_TOTAL_EXPOSURE_USD:.2f}"
            )

        # ── Gate 5: Market close time ─────────────────────────────────────────
        if candidate.close_time:
            try:
                close_dt = datetime.fromisoformat(candidate.close_time)
                minutes_to_close = (close_dt - datetime.utcnow()).total_seconds() / 60
                if minutes_to_close < settings.MIN_MINUTES_TO_CLOSE:
                    rejection_reasons.append(
                        f"Market closes in {minutes_to_close:.0f} min (min: {settings.MIN_MINUTES_TO_CLOSE})"
                    )
            except (ValueError, TypeError):
                pass  # no close time available, allow the trade

        # ── Decision ─────────────────────────────────────────────────────────
        if rejection_reasons:
            await self._record_rejection(candidate, rejection_reasons)
            return

        if order_contracts <= 0:
            await self._record_rejection(candidate, ["Computed order size is 0 contracts"])
            return

        await self._execute(candidate, order_contracts)

    def _size_order(self, candidate: TradeCandidate, current_contracts: int) -> int:
        """
        Quarter-Kelly order sizing.

        Kelly fraction for a binary bet: f = edge / (1 - p_market)
        Full Kelly is very aggressive; quarter-Kelly (KELLY_FRACTION = 0.25)
        provides conservative sizing with high geometric growth.

        Result is capped at (MAX_POSITION_PER_MARKET - current_contracts)
        to respect the per-market limit.
        """
        p = candidate.p_market
        edge = abs(candidate.edge)

        # Kelly fraction for this bet (simplified: ignores fees)
        kelly = edge / max(1 - p, 0.01)

        # Apply KELLY_FRACTION multiplier (0.25 for quarter-Kelly)
        contracts = int(kelly * _BANKROLL_CONTRACTS * settings.KELLY_FRACTION)

        # Cap at remaining position allowance
        remaining = settings.MAX_POSITION_PER_MARKET - current_contracts
        return max(0, min(contracts, remaining))

    async def _execute(self, candidate: TradeCandidate, contracts: int) -> None:
        """Place the order via Kalshi API and record the execution."""
        # YES side: buy YES at ask price (+1 cent buffer to improve fill probability)
        # NO side:  buy NO which is equivalent to selling YES
        side = candidate.side.lower()  # "yes" or "no"
        price_cents = int(candidate.p_market * 100)

        order = OrderRequest(
            ticker=candidate.ticker,
            side=side,
            action="buy",
            count=contracts,
            yes_price=price_cents,
        )

        try:
            response = await self._client.place_order(order)
            kalshi_order_id = response.order_id
            kalshi_status = response.status
            logger.info(
                "ORDER PLACED: %s %s %d @ %d cents | edge=%.3f | confidence=%.3f | headline: %s",
                side.upper(), candidate.ticker, contracts, price_cents,
                candidate.edge, candidate.confidence, candidate.headline[:60],
            )
        except Exception as exc:
            logger.error("Order placement failed for %s: %s", candidate.ticker, exc)
            await self._record_rejection(candidate, [f"Order placement failed: {exc}"])
            return

        # Stamp execution timestamp
        if candidate.trace:
            candidate.trace.stamp_executed()

        # Persist to trade_decisions
        from uuid import UUID
        news_event_id = UUID(candidate.news_event_id) if candidate.news_event_id else None
        await insert_trade_decision(
            news_event_id=news_event_id,
            market_ticker=candidate.ticker,
            market_title=candidate.title,
            action="EXECUTE",
            edge=candidate.edge,
            p_market=candidate.p_market,
            p_estimated=candidate.p_estimated,
            confidence=candidate.confidence,
            side=candidate.side,
            contracts=contracts,
            price_cents=price_cents,
            kalshi_order_id=kalshi_order_id,
            kalshi_status=kalshi_status,
        )

        # Update position tracking
        await upsert_position(
            market_ticker=candidate.ticker,
            market_title=candidate.title,
            side=candidate.side,
            contracts=contracts,
            avg_price_cents=float(price_cents),
        )

        # Write latency trace
        if candidate.trace:
            await self._write_trace(candidate.trace, news_event_id)

    async def _record_rejection(
        self,
        candidate: TradeCandidate,
        reasons: list[str],
    ) -> None:
        """Record a rejected trade to the DB."""
        from uuid import UUID
        news_event_id = UUID(candidate.news_event_id) if candidate.news_event_id else None

        logger.debug(
            "REJECTED %s | %s | reasons: %s",
            candidate.ticker, candidate.headline[:50], "; ".join(reasons),
        )

        await insert_trade_decision(
            news_event_id=news_event_id,
            market_ticker=candidate.ticker,
            market_title=candidate.title,
            action="REJECT",
            edge=candidate.edge,
            p_market=candidate.p_market,
            p_estimated=candidate.p_estimated,
            confidence=candidate.confidence,
            rejection_reasons=reasons,
        )

        # Stamp decided timestamp and write latency trace even for rejections
        if candidate.trace:
            candidate.trace.stamp_decided()
            await self._write_trace(candidate.trace, news_event_id)

    async def _write_trace(self, trace, news_event_id) -> None:
        """Persist the pipeline trace to the latency_metrics table."""
        try:
            await insert_latency_metric(
                source=trace.source,
                t_fetched=trace.t_fetched,
                t_deduped=trace.t_deduped,
                t_detected=trace.t_detected,
                t_matched=trace.t_matched,
                t_estimated=trace.t_estimated,
                t_decided=trace.t_decided,
                t_executed=trace.t_executed,
                news_event_id=news_event_id,
            )
        except Exception as exc:
            logger.warning("Failed to write latency trace: %s", exc)
