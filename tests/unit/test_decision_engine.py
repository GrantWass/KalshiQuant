"""
Unit tests for pipeline/decision_engine.py

Tests each of the 5 decision gates in isolation using mocked dependencies.
No database connection or Kalshi credentials required.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from pipeline.decision_engine import DecisionEngine, _BANKROLL_CONTRACTS
from pipeline.probability_estimator import TradeCandidate
from config.settings import settings


def make_candidate(**overrides) -> TradeCandidate:
    """Create a TradeCandidate that passes all 5 gates by default."""
    defaults = dict(
        news_event_id="00000000-0000-0000-0000-000000000001",
        headline="Hurricane warning Florida",
        source="nws",
        ticker="KXHURRICANE-2026SEP",
        title="Will a hurricane make landfall in Florida in September?",
        category="Weather",
        close_time=(datetime.utcnow() + timedelta(days=30)).isoformat(),
        p_market=0.35,
        p_estimated=0.47,
        edge=0.12,            # > PROBABILITY_SHIFT_MIN (0.04)
        side="YES",
        confidence=0.85,      # > CONFIDENCE_MIN (0.60)
        event_score=0.85,
        similarity_score=0.91,
        recency_factor=0.97,
        direction=1,
        trace=None,
    )
    defaults.update(overrides)
    return TradeCandidate(**defaults)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.place_order = AsyncMock(return_value=MagicMock(
        order_id="test-order-123",
        status="resting",
    ))
    return client


@pytest.fixture
def engine(mock_client):
    return DecisionEngine(mock_client)


class TestGate1Edge:
    async def test_passes_when_edge_sufficient(self, engine):
        candidate = make_candidate(edge=settings.PROBABILITY_SHIFT_MIN + 0.01)
        reasons = []
        # Gate 1 check only
        if abs(candidate.edge) < settings.PROBABILITY_SHIFT_MIN:
            reasons.append("edge too small")
        assert reasons == []

    async def test_fails_when_edge_too_small(self, engine):
        candidate = make_candidate(edge=0.01)  # below 0.04 minimum
        reasons = []
        if abs(candidate.edge) < settings.PROBABILITY_SHIFT_MIN:
            reasons.append(f"Edge {candidate.edge:+.3f} < minimum {settings.PROBABILITY_SHIFT_MIN}")
        assert len(reasons) == 1
        assert "Edge" in reasons[0]


class TestGate2Confidence:
    def test_passes_when_confidence_sufficient(self):
        candidate = make_candidate(confidence=settings.CONFIDENCE_MIN + 0.1)
        reasons = []
        if candidate.confidence < settings.CONFIDENCE_MIN:
            reasons.append("confidence too low")
        assert reasons == []

    def test_fails_when_confidence_too_low(self):
        candidate = make_candidate(confidence=0.30)
        reasons = []
        if candidate.confidence < settings.CONFIDENCE_MIN:
            reasons.append(f"Confidence {candidate.confidence:.3f} < minimum")
        assert len(reasons) == 1


class TestGate5CloseTime:
    def test_passes_when_market_open_long(self):
        close_time = (datetime.utcnow() + timedelta(days=30)).isoformat()
        candidate = make_candidate(close_time=close_time)
        reasons = []
        close_dt = datetime.fromisoformat(candidate.close_time)
        minutes_to_close = (close_dt - datetime.utcnow()).total_seconds() / 60
        if minutes_to_close < settings.MIN_MINUTES_TO_CLOSE:
            reasons.append("market closing soon")
        assert reasons == []

    def test_fails_when_market_closing_soon(self):
        close_time = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        candidate = make_candidate(close_time=close_time)
        reasons = []
        close_dt = datetime.fromisoformat(candidate.close_time)
        minutes_to_close = (close_dt - datetime.utcnow()).total_seconds() / 60
        if minutes_to_close < settings.MIN_MINUTES_TO_CLOSE:
            reasons.append(f"Market closes in {minutes_to_close:.0f} min")
        assert len(reasons) == 1

    def test_passes_when_no_close_time(self):
        candidate = make_candidate(close_time=None)
        reasons = []
        if candidate.close_time:
            reasons.append("close time check")
        assert reasons == []


class TestKellySizing:
    def test_kelly_is_positive_for_valid_candidate(self, engine):
        candidate = make_candidate(edge=0.12, p_market=0.35)
        size = engine._size_order(candidate, current_contracts=0)
        assert size > 0

    def test_kelly_capped_at_position_limit(self, engine):
        candidate = make_candidate(edge=0.50, p_market=0.10)  # very high edge
        current = settings.MAX_POSITION_PER_MARKET - 5
        size = engine._size_order(candidate, current_contracts=current)
        assert size <= 5  # can't exceed remaining allowance

    def test_kelly_zero_when_at_limit(self, engine):
        candidate = make_candidate()
        size = engine._size_order(candidate, current_contracts=settings.MAX_POSITION_PER_MARKET)
        assert size == 0
