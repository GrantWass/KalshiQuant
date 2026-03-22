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


# ── NLP prototype sentences ────────────────────────────────────────────────────
# These are prototype headlines representing the kinds of events that move
# prediction markets. The NLP score is the max cosine similarity of the
# input headline against all prototypes.
_NLP_PROTOTYPES = [
    # ── Weather / Natural Disasters ──
    "Hurricane makes landfall on the Florida coast causing widespread damage",
    "Category 4 hurricane intensifies and threatens Gulf Coast communities",
    "Tropical storm upgraded to hurricane status in the Atlantic",
    "Powerful earthquake strikes major city leaving thousands homeless",
    "Magnitude 7 earthquake hits Pacific Coast triggering tsunami warning",
    "Tornado outbreak devastates towns across the Midwest",
    "Wildfire spreads rapidly destroying thousands of acres in California",
    "Flash floods force mass evacuations across Southern states",
    "Blizzard shuts down major cities along the Northeast corridor",
    "Record-breaking drought declared in Western states",
    "Governor declares state of emergency after severe storm",
    "Levee breaches flood major metropolitan area",

    # ── Federal Reserve / Monetary Policy ──
    "Federal Reserve raises interest rates by 50 basis points",
    "Fed cuts interest rates to combat slowing economic growth",
    "Federal Reserve holds rates steady at FOMC meeting",
    "Jerome Powell signals rate hike in upcoming Fed meeting",
    "FOMC minutes reveal hawkish stance on inflation",
    "Fed chair hints at end of rate hiking cycle",

    # ── Economic Indicators ──
    "US inflation rate jumps unexpectedly to highest level in decades",
    "CPI report shows inflation cooling more than expected",
    "GDP growth slows sharply raising recession fears",
    "US economy adds fewer jobs than forecast in monthly report",
    "Unemployment rate rises to highest level in years",
    "Jobs report beats expectations with strong payroll gains",
    "Consumer spending falls sharply amid economic uncertainty",
    "US enters technical recession with two consecutive quarters of negative GDP",
    "Debt ceiling crisis threatens US government default",
    "Congress fails to pass budget triggering government shutdown",
    "Treasury yields surge as bond market sells off",

    # ── Politics / Elections ──
    "Presidential candidate wins key primary election in swing state",
    "Poll shows presidential race tightening in battleground states",
    "President signs sweeping executive order on immigration policy",
    "Senate confirms Supreme Court justice in party-line vote",
    "House passes major spending bill with bipartisan support",
    "President faces impeachment proceedings in Congress",
    "Senator announces resignation amid scandal",
    "Governor signs landmark legislation into law",
    "Election officials certify presidential election results",
    "Candidate drops out of presidential race ahead of primary",
    "Special election held to fill vacant Senate seat",
    "Supreme Court agrees to hear major constitutional case",

    # ── Supreme Court / Legal ──
    "Supreme Court issues landmark ruling overturning precedent",
    "Supreme Court rules on abortion rights nationwide",
    "High court strikes down federal law as unconstitutional",
    "Federal judge blocks administration policy pending appeal",
    "Indictment filed against sitting politician on federal charges",
    "Verdict reached in high-profile criminal trial",

    # ── Geopolitics / Conflict ──
    "Russia launches military offensive against Ukraine",
    "Ceasefire agreement reached ending months of fighting",
    "Country announces nuclear weapons test raising global alarm",
    "NATO activates Article 5 after attack on member nation",
    "US military strikes target in Middle East",
    "Sanctions imposed targeting country over nuclear program",
    "Coup attempt overthrows government in sudden military takeover",
    "Peace talks collapse as fighting resumes between warring parties",
    "North Korea fires ballistic missile over Japanese waters",
    "Iran nuclear deal collapses after US withdrawal",

    # ── Health / FDA ──
    "FDA approves breakthrough drug treatment for major disease",
    "WHO declares international public health emergency",
    "Novel virus outbreak spreads across multiple countries",
    "Vaccine shows high efficacy in large clinical trial results",
    "FDA issues warning recalling widely used medication",
    "New pandemic declared as disease spreads globally",

    # ── Crypto / Financial Markets ──
    "Bitcoin price surges past all-time high record",
    "Bitcoin crashes amid regulatory crackdown",
    "SEC approves Bitcoin ETF for US markets",
    "Major cryptocurrency exchange files for bankruptcy",
    "Stock market plunges on recession fears",
    "S&P 500 enters bear market territory with steep decline",

    # ── Energy ──
    "OPEC announces major production cut driving oil prices higher",
    "Oil prices spike after attack on major pipeline infrastructure",
    "US bans Russian oil imports over military aggression",

    # ── Sports ──
    "Team wins Super Bowl championship defeating rival in overtime",
    "NBA Finals winner crowned after seven-game series",
    "World Series champion emerges after dramatic playoff run",
    "US wins gold medal at Olympic Games",
    "World Cup final decided on penalty kicks",
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
