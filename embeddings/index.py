"""
embeddings/index.py — FAISS vector index manager.

Maintains a FAISS IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
over all open Kalshi markets. Supports:
  - Building from a list of (text, metadata) pairs
  - Searching with a query vector (returns top-k markets with similarity scores)
  - Atomic refresh under asyncio.Lock (safe to call from a background task)
  - Persisting to / loading from disk

Why IndexFlatIP?
  Kalshi has at most a few thousand open markets at any time. Exact flat search
  over this many vectors takes ~1-2ms, making approximate indexes (IVF, HNSW)
  unnecessary. Flat search also requires no training step.

Why inner product on normalized vectors?
  When vectors are L2-normalized, dot product equals cosine similarity.
  This gives a clean [0..1] similarity score that maps directly to semantic relevance.

Usage:
    idx = FAISSIndex()
    idx.build(embeddings, metadata_list)

    scores, metas = idx.search(query_vec, k=5)
    # scores: list of cosine similarities [0..1], descending
    # metas:  list of market metadata dicts (ticker, title, category, close_time)
"""

import asyncio
import json
import logging
import os
import numpy as np
import faiss
from config.settings import settings

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Thread-safe (asyncio) FAISS index with atomic refresh support.

    The refresh lock prevents race conditions between:
      - Live pipeline searches (market_matcher.py)
      - Periodic 15-min index rebuilds (market_index_builder.py)
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] = []         # parallel list: metadata[i] ↔ index vector i
        self._lock = asyncio.Lock()
        self._total = 0

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Build the FAISS index from pre-computed embeddings.

        Args:
            embeddings: float32 array of shape (N, EMBEDDING_DIM), L2-normalized.
            metadata:   list of N dicts, each with keys: ticker, title, category, close_time, tags
        """
        assert embeddings.shape[0] == len(metadata), "embeddings and metadata must have same length"
        assert embeddings.dtype == np.float32, "embeddings must be float32"

        index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)
        index.add(embeddings)

        self._index = index
        self._metadata = metadata
        self._total = len(metadata)
        logger.info("FAISS index built: %d markets indexed.", self._total)

    async def atomic_replace(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Replace the current index atomically under the async lock.
        Called by market_index_builder every 15 minutes.

        All searches that are waiting on the lock will get the NEW index
        immediately after this completes.
        """
        async with self._lock:
            self.build(embeddings, metadata)
        logger.info("FAISS index atomically refreshed: %d markets.", self._total)

    async def search(self, query: np.ndarray, k: int) -> list[tuple[float, dict]]:
        """
        Search for the top-k most similar markets to a query embedding.

        Args:
            query: float32 array of shape (EMBEDDING_DIM,), L2-normalized.
            k:     number of results to return.

        Returns:
            List of (similarity_score, market_metadata) tuples, sorted by score descending.
            similarity_score is cosine similarity in [0..1].
            Only returns markets with score >= SIMILARITY_MIN_SCORE.
        """
        if self._index is None or self._total == 0:
            return []

        # Reshape to (1, dim) for FAISS
        query_2d = query.reshape(1, -1)

        async with self._lock:
            k_actual = min(k, self._total)
            scores_2d, indices_2d = self._index.search(query_2d, k_actual)

        scores = scores_2d[0].tolist()
        indices = indices_2d[0].tolist()

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:   # FAISS returns -1 for unfilled slots
                continue
            if score < settings.SIMILARITY_MIN_SCORE:
                continue
            results.append((float(score), self._metadata[idx]))

        return results

    def save(self) -> None:
        """Persist the index and metadata to disk (for faster startup)."""
        if self._index is None:
            return
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH) or ".", exist_ok=True)
        faiss.write_index(self._index, settings.FAISS_INDEX_PATH)
        with open(settings.FAISS_METADATA_PATH, "w") as f:
            json.dump(self._metadata, f, default=str)
        logger.info("FAISS index saved to %s (%d markets).", settings.FAISS_INDEX_PATH, self._total)

    def load(self) -> bool:
        """
        Load a previously saved index from disk.
        Returns True if successful, False if no saved index exists.
        """
        if not os.path.exists(settings.FAISS_INDEX_PATH):
            return False
        try:
            self._index = faiss.read_index(settings.FAISS_INDEX_PATH)
            with open(settings.FAISS_METADATA_PATH) as f:
                self._metadata = json.load(f)
            self._total = len(self._metadata)
            logger.info("FAISS index loaded from disk: %d markets.", self._total)
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index from disk: %s. Will rebuild.", exc)
            return False

    @property
    def total_markets(self) -> int:
        return self._total


# Module-level singleton shared by MarketMatcher and MarketIndexBuilder.
faiss_index = FAISSIndex()
