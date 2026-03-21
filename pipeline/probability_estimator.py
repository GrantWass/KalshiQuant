"""
pipeline/probability_estimator.py — Probability shift estimation.

For each (news event, market) pair, estimates how much the market probability
should shift given the news, and computes the resulting trading edge.

Full model documentation is in README.md § Probability Model.

Formula summary:
  P_new = P_market + direction * event_score * similarity_score * MAX_SHIFT
  edge  = P_new - P_market
  confidence = harmonic_mean(event_score, similarity_score) * recency_factor

  direction:      +1 if news increases the probability, -1 if it decreases it
  recency_factor: exp(-age_minutes / RECENCY_HALF_LIFE_MINUTES), clipped [0.2, 1.0]

Sentiment (direction):
  Uses a simple positive/negative keyword approach for speed (~0.01ms).
  Words like "warning", "crisis", "attack" → negative (decreases probability
  of normal conditions but *increases* probability of the risk event).
  Context-aware sentiment would require a heavier model — this is a known
  simplification that can be improved later.

Current price source:
  Primary: KalshiWebSocketManager._prices dict (0ms, updated live)
  Fallback: KalshiClient.get_market() REST call (~200ms)
  If neither is available: skip the trade candidate (log a warning)
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime

from config.settings import settings
from kalshi.websocket import KalshiWebSocketManager
from kalshi.client import KalshiClient
from pipeline.market_matcher import MatchedEvent, MarketMatch

logger = logging.getLogger(__name__)


#TODO: look into using a more sophisticated sentiment model (e.g., finBERT) for direction estimation.
# The current keyword-based approach is fast but may misclassify some headlines.

# ── Sentiment keyword lists ────────────────────────────────────────────────────
# These determine the *direction* of the probability shift.
# "Positive" keywords increase the probability of the event described by the market.
# "Negative" keywords decrease it.
# Context matters: "hurricane warning" increases landfall probability (positive direction
# for a landfall market) but these simple word lists work well for common cases.
_POSITIVE_KEYWORDS = {
    "warning", "watch", "emergency", "alert", "threat", "risk", "danger",
    "imminent", "approaching", "landfall", "strike", "hit", "makes landfall",
    "confirmed", "wins", "victory", "passes", "approved", "signed",
    "raises", "hike", "increase", "surge", "spike", "jump",
    "declares war", "invades", "attacks", "launches",
}

_NEGATIVE_KEYWORDS = {
    "weakens", "dissipates", "downgraded", "cancelled", "lifted",
    "ceasefire", "deal", "agreement", "resolved", "peace",
    "cuts", "reduces", "falls", "drops", "declines",
    "loses", "defeat", "fails", "rejected", "vetoed",
    "miss", "below expectations",
}


@dataclass
class TradeCandidate:
    """
    A market trade candidate produced by the probability estimator.
    Contains all the information needed by the decision engine.
    """
    # Source news item
    news_event_id: str | None
    headline: str
    source: str

    # Market info
    ticker: str
    title: str
    category: str | None
    close_time: str | None

    # Probability analysis
    p_market: float           # current market price (YES probability)
    p_estimated: float        # our estimated new probability
    edge: float               # p_estimated - p_market
    side: str                 # "YES" if edge > 0 (buy YES), "NO" if edge < 0 (buy NO)
    confidence: float         # [0..1] combined confidence score

    # Component scores (for dashboard transparency)
    event_score: float
    similarity_score: float
    recency_factor: float
    direction: int            # +1 or -1

    # Pipeline trace (passed through for latency recording)
    trace: object | None = None


class ProbabilityEstimator:
    """
    Estimates the probability shift for each matched (news, market) pair.
    """

    def __init__(
        self,
        ws_manager: KalshiWebSocketManager,
        kalshi_client: KalshiClient,
    ) -> None:
        self._ws = ws_manager
        self._client = kalshi_client

    async def run(
        self,
        matched_queue: asyncio.Queue,
        candidates_queue: asyncio.Queue,
    ) -> None:
        """
        Consume MatchedEvents, estimate probabilities, produce TradeCandidates.
        """
        while True:
            event: MatchedEvent = await matched_queue.get()
            try:
                await self._process(event, candidates_queue)
            except Exception as exc:
                logger.error("ProbabilityEstimator error: %s", exc)
            finally:
                matched_queue.task_done()

    async def _process(
        self, event: MatchedEvent, candidates_queue: asyncio.Queue
    ) -> None:
        """Process each (news, market) pair and emit trade candidates."""
        item = event.detected.item
        event_score = event.detected.event_score

        # Compute recency factor once per news item
        recency = _recency_factor(item.published_at)

        # Compute sentiment direction once per news item
        direction = _sentiment_direction(item.headline)

        for match in event.matches:
            candidate = await self._estimate_for_market(
                item=item,
                match=match,
                event_score=event_score,
                recency=recency,
                direction=direction,
            )
            if candidate is None:
                continue

            # Stamp estimation timestamp on the trace
            if item.trace:
                item.trace.stamp_estimated()
                candidate.trace = item.trace

            try:
                candidates_queue.put_nowait(candidate)
            except asyncio.QueueFull:
                logger.warning("Candidates queue full.")

    async def _estimate_for_market(
        self,
        item,
        match: MarketMatch,
        event_score: float,
        recency: float,
        direction: int,
    ) -> TradeCandidate | None:
        """
        Estimate probability shift for a single (news, market) pair.

        Returns None if current market price is unavailable.
        """
        # ── Get current market price ───────────────────────────────────────────
        # Primary: WebSocket cache (instant)
        price_cents = self._ws.get_price(match.ticker)

        if price_cents is None:
            # Subscribe to this market so future polls have live prices
            await self._ws.subscribe([match.ticker])

            # Fallback: REST API call (~200ms)
            try:
                market = await self._client.get_market(match.ticker)
                price_cents = market.mid_price_cents
            except Exception as exc:
                logger.warning("Could not get price for %s: %s", match.ticker, exc)
                return None

        if price_cents is None:
            return None

        p_market = price_cents / 100.0

        # ── Compute probability shift ──────────────────────────────────────────
        #
        # base_shift: how much our estimate deviates from current market price.
        #   Scaled by event_score (how market-moving is the news?) and
        #   similarity_score (how relevant is this market to the news?).
        #   Capped by MAX_SHIFT to prevent runaway estimates.
        #
        base_shift = event_score * match.similarity_score * settings.MAX_SHIFT

        #
        # Apply direction: +1 means event supports market outcome, -1 opposes it.
        # Apply recency dampening: older news contributes less.
        #
        p_estimated = p_market + direction * base_shift * recency

        # Clip to valid probability range [0.01, 0.99]
        # (Kalshi prices must be between 1 and 99 cents)
        p_estimated = max(0.01, min(0.99, p_estimated))

        edge = p_estimated - p_market

        # ── Compute confidence ─────────────────────────────────────────────────
        #
        # harmonic_mean penalizes imbalanced scores:
        #   e.g., high event_score but low similarity → low confidence
        #   This prevents trading on loosely related markets.
        #
        hm = _harmonic_mean(event_score, match.similarity_score)
        confidence = hm * recency

        # Determine trade side
        side = "YES" if edge > 0 else "NO"

        return TradeCandidate(
            news_event_id=str(item.db_id) if item.db_id else None,
            headline=item.headline,
            source=item.source,
            ticker=match.ticker,
            title=match.title,
            category=match.category,
            close_time=match.close_time,
            p_market=p_market,
            p_estimated=p_estimated,
            edge=edge,
            side=side,
            confidence=confidence,
            event_score=event_score,
            similarity_score=match.similarity_score,
            recency_factor=recency,
            direction=direction,
            trace=None,  # filled in by caller
        )


# ── Helper functions ───────────────────────────────────────────────────────────

def _recency_factor(published_at: datetime) -> float:
    """
    Exponential decay based on news age.

    recency_factor = exp(-age_minutes / HALF_LIFE), clipped to [0.2, 1.0]

    Rationale:
      - Fresh news (< 1 min): factor ~1.0 — full signal strength
      - 30 min old: factor ~0.50 — half the signal strength (half-life)
      - 2 hours old: factor ~0.20 — floor, still some signal but heavily discounted
      - Older than that: floor at 0.20 — we don't completely ignore it, but barely
    """
    now = datetime.utcnow()
    age_seconds = (now - published_at).total_seconds()
    age_minutes = max(age_seconds / 60, 0)
    factor = math.exp(-age_minutes / settings.RECENCY_HALF_LIFE_MINUTES)
    return max(0.20, min(1.0, factor))


def _sentiment_direction(headline: str) -> int:
    """
    Determine whether the news is positive (+1) or negative (-1) relative
    to the market event probability.

    Simple keyword matching on the lowercased headline.
    Default to +1 (most market-moving events are about something happening,
    which increases the probability of event markets).
    """
    headline_lower = headline.lower()
    pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in headline_lower)
    neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in headline_lower)

    if neg_hits > pos_hits:
        return -1
    return 1  # default: assume event is occurring/intensifying


def _harmonic_mean(a: float, b: float) -> float:
    """
    Harmonic mean of two values in [0..1].
    Returns 0 if either value is 0 (undefined harmonic mean).

    The harmonic mean is lower than arithmetic mean when values differ —
    this penalizes imbalanced scores (e.g., high event score but weak market match).
    """
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)
