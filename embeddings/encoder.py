"""
embeddings/encoder.py — sentence-transformers wrapper.

Encodes text to dense vectors using the all-MiniLM-L6-v2 model (384 dimensions).
The same model is used for:
  - Market embeddings (at index build time)
  - News headline embeddings (at market matching time)
  - Event prototype embeddings (at event detection time)

Using the same model for all three ensures vector spaces are compatible.

Model characteristics (all-MiniLM-L6-v2):
  - 384 dimensions
  - ~22MB model size
  - ~8ms per encode on CPU (single text)
  - ~64ms for a batch of 64 texts on CPU
  - Good semantic understanding for short to medium-length text

To switch to a higher-accuracy model (e.g., all-mpnet-base-v2, 768 dim),
update EMBEDDING_MODEL and EMBEDDING_DIM in config/settings.py.
"""

import asyncio
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)


class Encoder:
    """
    Wraps a SentenceTransformer model.

    Encoding is CPU-bound. To avoid blocking the asyncio event loop,
    batch encoding is offloaded to a thread pool executor via
    asyncio.get_event_loop().run_in_executor().

    Single-item encodes (used in the hot pipeline path) are small enough
    (~8ms) that they run synchronously without noticeable loop blocking.
    The asyncio event loop is typically fine with < 10ms blocking calls
    between awaits.
    """

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded (dim=%d).", settings.EMBEDDING_DIM)

    def encode_sync(self, text: str) -> np.ndarray:
        """
        Encode a single text synchronously. Returns an L2-normalized float32 array
        of shape (EMBEDDING_DIM,).

        Used in the pipeline hot path where a single headline is encoded per event.
        (~8ms on CPU — acceptable blocking time between awaits)
        """
        vec = self._model.encode(
            text,
            normalize_embeddings=True,   # L2-normalize so dot product = cosine similarity
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    async def encode(self, text: str) -> np.ndarray:
        """
        Async wrapper for encode_sync. Offloads to thread pool to avoid blocking.
        Use this when latency tolerance allows (e.g., in the market index builder).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_sync, text)

    async def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts. Returns float32 array of shape (N, EMBEDDING_DIM),
        L2-normalized. Used during FAISS index builds.
        """
        loop = asyncio.get_event_loop()

        def _encode_batch() -> np.ndarray:
            vecs = self._model.encode(
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )
            return vecs.astype(np.float32)

        return await loop.run_in_executor(None, _encode_batch)


# Module-level singleton — import this everywhere to avoid reloading the model.
encoder = Encoder()
