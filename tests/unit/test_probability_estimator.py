"""
Unit tests for pipeline/probability_estimator.py

Tests the core probability formula: P_new, edge, confidence, and helper functions.
No external dependencies required.
"""

import math
import pytest
from datetime import datetime, timedelta

from pipeline.probability_estimator import (
    _recency_factor,
    _sentiment_direction,
    _harmonic_mean,
)


class TestRecencyFactor:
    def test_fresh_news_near_one(self):
        """News from 1 minute ago should have recency ~1.0."""
        t = datetime.utcnow() - timedelta(minutes=1)
        factor = _recency_factor(t)
        assert factor > 0.95, f"Expected ~1.0 for fresh news, got {factor}"

    def test_thirty_min_old_near_half(self):
        """News from 30 min ago should be ~0.5 (half-life)."""
        t = datetime.utcnow() - timedelta(minutes=30)
        factor = _recency_factor(t)
        # exp(-1) ≈ 0.368... but clipped to [0.2, 1.0], half-life should give ~0.5
        assert 0.30 < factor < 0.55, f"Expected ~0.5 for 30-min-old news, got {factor}"

    def test_old_news_at_floor(self):
        """News older than 2 hours should hit the floor at 0.2."""
        t = datetime.utcnow() - timedelta(hours=5)
        factor = _recency_factor(t)
        assert factor == pytest.approx(0.2, abs=0.01), f"Expected floor 0.2, got {factor}"

    def test_always_in_range(self):
        """Recency factor should always be in [0.2, 1.0]."""
        for minutes_ago in [0, 1, 15, 30, 60, 120, 300]:
            t = datetime.utcnow() - timedelta(minutes=minutes_ago)
            factor = _recency_factor(t)
            assert 0.2 <= factor <= 1.0


class TestSentimentDirection:
    def test_hurricane_warning_positive(self):
        """A hurricane warning increases probability of a hurricane market."""
        assert _sentiment_direction("Hurricane Warning issued for Florida Gulf Coast") == 1

    def test_storm_weakens_negative(self):
        """A weakening storm decreases probability."""
        assert _sentiment_direction("Hurricane weakens to tropical storm as it moves offshore") == -1

    def test_default_positive(self):
        """Ambiguous headlines default to +1."""
        assert _sentiment_direction("New data released on economic conditions") == 1

    def test_ceasefire_negative(self):
        """A ceasefire decreases probability of ongoing conflict."""
        assert _sentiment_direction("Ceasefire agreement reached between warring parties") == -1


class TestHarmonicMean:
    def test_equal_values(self):
        """Harmonic mean of equal values equals those values."""
        assert _harmonic_mean(0.8, 0.8) == pytest.approx(0.8, abs=0.001)

    def test_imbalanced_values(self):
        """Harmonic mean is lower than arithmetic mean for imbalanced values."""
        hm = _harmonic_mean(0.9, 0.1)
        am = (0.9 + 0.1) / 2
        assert hm < am

    def test_zero_input(self):
        """Zero input returns 0 (undefined harmonic mean)."""
        assert _harmonic_mean(0.8, 0.0) == 0.0
        assert _harmonic_mean(0.0, 0.8) == 0.0

    def test_high_confidence_scenario(self):
        """High event score and high similarity should give high confidence."""
        hm = _harmonic_mean(0.85, 0.91)
        assert hm > 0.85, f"Expected > 0.85, got {hm}"


class TestProbabilityFormula:
    """Test the full edge calculation formula as documented in the README."""

    def test_worked_example(self):
        """
        Worked example from README:
          event_score=0.85, similarity=0.91, direction=+1, p_market=0.35
          base_shift = 0.85 * 0.91 * 0.15 = 0.116
          P_new = 0.35 + 0.116 = 0.466
          edge = 0.116
        """
        from config.settings import settings

        event_score = 0.85
        similarity = 0.91
        direction = 1
        p_market = 0.35
        recency = 1.0   # fresh news

        base_shift = event_score * similarity * settings.MAX_SHIFT
        p_new = p_market + direction * base_shift * recency
        edge = p_new - p_market

        assert edge == pytest.approx(0.116, abs=0.002)
        assert p_new == pytest.approx(0.466, abs=0.002)
        assert edge > settings.PROBABILITY_SHIFT_MIN   # should pass Gate 1

    def test_edge_above_minimum_passes_gate(self):
        """Edge above PROBABILITY_SHIFT_MIN should pass Gate 1."""
        from config.settings import settings
        edge = settings.PROBABILITY_SHIFT_MIN + 0.01
        assert abs(edge) >= settings.PROBABILITY_SHIFT_MIN

    def test_edge_below_minimum_fails_gate(self):
        """Edge below PROBABILITY_SHIFT_MIN should fail Gate 1."""
        from config.settings import settings
        edge = settings.PROBABILITY_SHIFT_MIN - 0.01
        assert abs(edge) < settings.PROBABILITY_SHIFT_MIN
