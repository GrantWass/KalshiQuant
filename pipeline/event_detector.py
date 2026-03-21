"""
pipeline/event_detector.py — Two-stage event detection.

Most incoming news is irrelevant to any Kalshi market. This stage filters
the stream to keep only potentially market-moving items, reducing the load
on the expensive FAISS search and probability estimation stages.

Stage 1 — Keyword Scoring (~0.1ms):
  Fast regex/set lookup for market-relevant keywords across categories.
  Items with keyword_score < KEYWORD_SCORE_NLP_THRESHOLD skip Stage 2.
  keyword_score = sum(matched keyword weights) / normalization_factor, clipped to [0..1]

Stage 2 — NLP Scoring (~8ms, only if Stage 1 score passes threshold):
  Encodes the headline using all-MiniLM-L6-v2 and computes cosine similarity
  against a small set of pre-computed "market-moving event" prototype embeddings.
  These prototypes are manually chosen to cover the spectrum of Kalshi market types.

Final score:
  event_score = KEYWORD_WEIGHT * keyword_score + (1 - KEYWORD_WEIGHT) * nlp_score

Items below EVENT_DETECTION_MIN_SCORE are:
  - Written to DB with filtered_out=True and a reason string
  - Dropped from the pipeline (not sent downstream)
  - Still visible on Dashboard Page 1 (red rows)

Items that pass are:
  - Updated in DB with their scores
  - Forwarded to the market matcher
"""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from config.settings import settings
from db.repositories.news import update_news_event_scores
from embeddings.encoder import encoder
from news.base import NewsItem

logger = logging.getLogger(__name__)


# ── Keyword dictionary ─────────────────────────────────────────────────────────
# Each category maps keywords to a weight (1.0 = base, higher = more signal).
# Keywords are matched case-insensitively in the headline + first 500 chars of body.
_KEYWORD_CATEGORIES: dict[str, tuple[float, list[str]]] = {
    "weather_disaster": (1.2, [
        "hurricane", "typhoon", "tornado", "earthquake", "tsunami",
        "flood", "wildfire", "blizzard", "snowstorm", "drought",
        "tropical storm", "landfall", "storm surge", "evacuation order",
        "state of emergency", "extreme wind",
    ]),
    "political": (1.0, [
        "election", "vote", "ballot", "poll", "candidate", "president",
        "senate", "congress", "parliament", "indictment", "impeach",
        "resigns", "appointed", "confirmed", "filibuster", "legislation",
        "passes bill", "veto", "executive order",
    ]),
    "economic": (1.0, [
        "federal reserve", "interest rate", "rate cut", "rate hike",
        "inflation", "cpi", "gdp", "unemployment", "jobs report",
        "recession", "fed funds", "fomc", "jerome powell",
        "treasury", "deficit", "debt ceiling", "tariff", "trade war",
        "earnings", "bankruptcy", "layoffs",
    ]),
    "conflict": (1.1, [
        "war", "military", "troops", "invasion", "missile", "nuclear",
        "sanctions", "ceasefire", "attack", "bombing", "armed conflict",
        "coup", "civil war",
    ]),
    "health": (1.0, [
        "fda approval", "drug approved", "pandemic", "outbreak",
        "vaccine", "clinical trial", "public health emergency",
        "who declares", "disease spread",
    ]),
    "sports": (0.8, [
        "super bowl", "nfl championship", "nba finals", "world series",
        "world cup", "olympic", "championship game",
    ]),
}

# Normalize: max possible keyword score (all categories hit at max weight)
_MAX_KEYWORD_SCORE = sum(w for w, _ in _KEYWORD_CATEGORIES.values())


#TODO: investigate into this further
# ── NLP prototype sentences ────────────────────────────────────────────────────
# These are prototype headlines representing the kinds of events that move
# prediction markets. The NLP score is the max cosine similarity of the
# input headline against all prototypes.
_NLP_PROTOTYPES = [
    "Hurricane makes landfall on the Florida coast",
    "Federal Reserve raises interest rates by 50 basis points",
    "President signs executive order on immigration",
    "Congress passes major legislation on healthcare",
    "Earthquake hits major city causing widespread damage",
    "Military conflict escalates between two countries",
    "Candidate wins election in unexpected result",
    "Supreme Court issues landmark ruling",
    "FDA approves new drug treatment",
    "Economic data shows unexpected inflation spike",
    "Major tornado touches down causing destruction",
    "Country imposes new economic sanctions",
    "Government announces emergency measures",
    "Natural disaster triggers evacuation orders",
    "Election results show shift in polls",
]


@dataclass
class DetectedEvent:
    """A news item that passed event detection, with its scores attached."""
    item: NewsItem
    keyword_score: float
    nlp_score: float
    event_score: float


class EventDetector:
    """
    Two-stage event detector. Initialized with pre-computed NLP prototypes.
    """

    def __init__(self) -> None:
        self._prototype_embeddings: np.ndarray | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Pre-compute prototype embeddings at startup.
        Called once by the orchestrator before the pipeline starts.
        """
        logger.info("Pre-computing %d event detection prototypes...", len(_NLP_PROTOTYPES))
        self._prototype_embeddings = await encoder.encode_batch(_NLP_PROTOTYPES)
        self._initialized = True
        logger.info("Event detector initialized.")

    def _keyword_score(self, text: str) -> float:
        """
        Stage 1: compute a weighted keyword match score.
        text: headline + first 500 chars of body (lowercased)
        Returns a score in [0..1].
        """
        text_lower = text.lower()
        total_weight = 0.0
        for category, (weight, keywords) in _KEYWORD_CATEGORIES.items():
            for kw in keywords:
                if kw in text_lower:
                    total_weight += weight
                    break  # count each category at most once

        return min(total_weight / _MAX_KEYWORD_SCORE, 1.0)

    def _nlp_score(self, headline_embedding: np.ndarray) -> float:
        """
        Stage 2: cosine similarity against prototype embeddings.
        Both the headline and prototypes are L2-normalized, so this is a
        fast dot product.
        Returns the maximum similarity across all prototypes (in [0..1]).
        """
        if self._prototype_embeddings is None:
            return 0.0
        # Shape: (num_prototypes,) — dot products of normalized vectors = cosine similarities
        similarities = self._prototype_embeddings @ headline_embedding
        return float(np.max(similarities))

    async def run(
        self,
        deduped_queue: asyncio.Queue,
        events_queue: asyncio.Queue,
    ) -> None:
        """
        Consume from deduped_queue, score items, produce DetectedEvents to events_queue.
        Items below the score threshold are recorded in DB (filtered_out=True) and dropped.
        """
        if not self._initialized:
            await self.initialize()

        while True:
            item: NewsItem = await deduped_queue.get()
            try:
                await self._process(item, events_queue)
            except Exception as exc:
                logger.error("EventDetector error for %s: %s", item.headline[:60], exc)
            finally:
                deduped_queue.task_done()

    async def _process(self, item: NewsItem, events_queue: asyncio.Queue) -> None:
        """Score a single item and route it appropriately."""
        # Combine headline and body prefix for keyword matching
        text = item.headline
        if item.body:
            text += " " + item.body[:500]

        # Stage 1: keyword scoring (~0.1ms)
        kw_score = self._keyword_score(text)

        # Stage 2: NLP scoring (only if Stage 1 is promising, ~8ms)
        nlp_score = 0.0
        if kw_score >= settings.KEYWORD_SCORE_NLP_THRESHOLD:
            embedding = encoder.encode_sync(item.headline)
            nlp_score = self._nlp_score(embedding)

        # Combined final score
        event_score = (
            settings.KEYWORD_WEIGHT * kw_score
            + (1 - settings.KEYWORD_WEIGHT) * nlp_score
        )

        # Stamp detection timestamp
        if item.trace:
            item.trace.stamp_detected()

        # Update DB with scores
        if item.db_id:
            filtered = event_score < settings.EVENT_DETECTION_MIN_SCORE
            reason = None
            if filtered:
                reason = f"event_score {event_score:.3f} < min {settings.EVENT_DETECTION_MIN_SCORE}"

            await update_news_event_scores(
                event_id=item.db_id,
                event_score=event_score,
                keyword_score=kw_score,
                nlp_score=nlp_score,
                filtered_out=filtered,
                filter_reason=reason,
            )

            if filtered:
                return  # drop from pipeline

        # Item passed — wrap in DetectedEvent and forward
        detected = DetectedEvent(
            item=item,
            keyword_score=kw_score,
            nlp_score=nlp_score,
            event_score=event_score,
        )
        try:
            events_queue.put_nowait(detected)
        except asyncio.QueueFull:
            logger.warning("Events queue full; dropping detected event: %s", item.headline[:60])
