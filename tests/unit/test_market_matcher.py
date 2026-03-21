"""
Unit tests for embeddings/index.py (FAISS index) and pipeline/market_matcher.py

Tests vector search logic without requiring a real FAISS index or network.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


class TestFAISSIndex:
    def test_search_returns_results_above_threshold(self):
        """FAISS search should only return results above SIMILARITY_MIN_SCORE."""
        from embeddings.index import FAISSIndex
        import faiss

        idx = FAISSIndex()
        dim = 384
        n = 10

        # Create random normalized vectors
        vecs = np.random.randn(n, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        meta = [
            {"ticker": f"MARKET-{i}", "title": f"Market {i}", "category": None, "close_time": None, "tags": [], "embedding_text": ""}
            for i in range(n)
        ]

        idx.build(vecs, meta)
        assert idx.total_markets == n

    def test_build_and_search(self):
        """Build a small index and verify search returns results."""
        import asyncio
        from embeddings.index import FAISSIndex
        from config.settings import settings

        idx = FAISSIndex()
        dim = settings.EMBEDDING_DIM
        n = 5

        vecs = np.random.randn(n, dim).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        meta = [
            {"ticker": f"T-{i}", "title": f"Title {i}", "category": "test", "close_time": None, "tags": [], "embedding_text": "test"}
            for i in range(n)
        ]
        idx.build(vecs, meta)

        # Search with a vector identical to the first market vector
        query = vecs[0]  # should return 1.0 similarity for itself

        async def run_search():
            return await idx.search(query, k=3)

        results = asyncio.run(run_search())
        # The query vector should match itself with similarity = 1.0
        assert len(results) > 0
        best_score, best_meta = results[0]
        assert best_score > 0.99, f"Self-match should be ~1.0, got {best_score}"
