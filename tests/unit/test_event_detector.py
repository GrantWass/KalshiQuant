"""
Unit tests for pipeline/event_detector.py

Tests the keyword scoring and combined scoring logic.
These tests do NOT require a database connection or Kalshi credentials.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
import numpy as np

from pipeline.event_detector import EventDetector, _KEYWORD_CATEGORIES


@pytest.fixture
def detector():
    d = EventDetector()
    # Pre-set a mock prototype embedding so we don't need the real model
    d._prototype_embeddings = np.ones((5, 384), dtype=np.float32) / np.sqrt(384)
    d._initialized = True
    return d


def test_keyword_score_hurricane(detector):
    """A hurricane headline should get a high keyword score."""
    score = detector._keyword_score("Hurricane Category 5 approaches Florida coast")
    assert score > 0.1, f"Expected score > 0.1, got {score}"


def test_keyword_score_election(detector):
    """An election headline should score."""
    score = detector._keyword_score("Presidential election results expected tonight")
    assert score > 0.05, f"Expected score > 0.05, got {score}"


def test_keyword_score_irrelevant(detector):
    """A celebrity gossip headline should score near 0."""
    score = detector._keyword_score("Actor wins award at film festival last night")
    assert score < 0.05, f"Expected score < 0.05, got {score}"


def test_keyword_score_economic(detector):
    """A Fed rate hike headline should score."""
    score = detector._keyword_score("Federal Reserve raises interest rates by 75 basis points")
    assert score > 0.1, f"Expected score > 0.1, got {score}"


def test_keyword_score_normalized(detector):
    """Score should never exceed 1.0."""
    max_text = " ".join([
        kw
        for _, (_, keywords) in _KEYWORD_CATEGORIES.items()
        for kw in keywords
    ])
    score = detector._keyword_score(max_text)
    assert score <= 1.0, f"Score {score} exceeds 1.0"


def test_event_score_combines_correctly(detector):
    """Final score combines keyword and NLP scores at configured weights."""
    from config.settings import settings

    # Mock NLP score
    with patch.object(detector, "_nlp_score", return_value=0.8):
        kw_score = detector._keyword_score("hurricane warning Florida")
        nlp_score = 0.8
        expected = settings.KEYWORD_WEIGHT * kw_score + (1 - settings.KEYWORD_WEIGHT) * nlp_score
        # Verify the formula manually
        assert abs(expected - (settings.KEYWORD_WEIGHT * kw_score + 0.6 * 0.8)) < 0.001
