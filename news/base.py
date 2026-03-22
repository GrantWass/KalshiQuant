"""
news/base.py — Core data types for the news ingestion layer.

Two key types:
  1. NewsItem   — a single news article/alert as it enters the pipeline
  2. PipelineTrace — timestamp record that travels with each item through all stages

Every news source produces NewsItem objects. The pipeline attaches a PipelineTrace
at the moment of creation and stamps each stage timestamp as the item progresses.
At the end of the pipeline, the trace is written to the latency_metrics table
and surfaced on Dashboard Page 6.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator
from uuid import UUID, uuid4


@dataclass
class NewsItem:
    """
    A single news article or alert.

    Fields:
        source      — which source produced this item (e.g. "nws", "gdelt", "rss_ap")
        source_id   — source-native unique ID (used for deduplication)
        headline    — the headline or title of the article
        published_at — when the article was published (source's reported time)
        fetched_at  — when we retrieved it (our wall clock)
        body        — article body text (may be None for headline-only sources)
        url         — link to the original article (may be None)
        raw_payload — the original data structure from the source API (for audit)
    """

    source: str
    source_id: str
    headline: str
    published_at: datetime
    fetched_at: datetime

    body: str | None = None
    url: str | None = None
    raw_payload: dict = field(default_factory=dict)

    # Set by the pipeline after DB insertion
    db_id: UUID | None = None

    # Pipeline trace: attached at fetch time, stamped at each stage
    trace: "PipelineTrace | None" = None

    def __post_init__(self) -> None:
        # Auto-attach a trace at creation time
        if self.trace is None:
            self.trace = PipelineTrace(
                source=self.source,
                t_fetched=self.fetched_at,
            )


@dataclass
class PipelineTrace:
    """
    Timestamp record for a single item's journey through the pipeline.

    Each stage stamps the corresponding t_<stage> field when it completes.
    The completed trace is written to the latency_metrics DB table and
    shown on Dashboard Page 6 as a Gantt-style row.

    Stage order:
        t_fetched   → news source created the NewsItem
        t_deduped   → deduplicator confirmed item is new (not a duplicate)
        t_detected  → event detector scored it (may be filtered_out after this)
        t_matched   → market matcher found candidate markets
        t_estimated → probability estimator computed edge
        t_decided   → decision engine made EXECUTE or REJECT decision
        t_executed  → Kalshi API accepted the order (None for rejected items)
    """

    source: str
    t_fetched: datetime

    t_deduped: datetime | None = None
    t_detected: datetime | None = None
    t_matched: datetime | None = None
    t_estimated: datetime | None = None
    t_decided: datetime | None = None
    t_executed: datetime | None = None

    # Set after DB insertion so the latency record can reference it
    news_event_id: UUID | None = None

    def stamp_deduped(self) -> None:
        self.t_deduped = datetime.utcnow()

    def stamp_detected(self) -> None:
        self.t_detected = datetime.utcnow()

    def stamp_matched(self) -> None:
        self.t_matched = datetime.utcnow()

    def stamp_estimated(self) -> None:
        self.t_estimated = datetime.utcnow()

    def stamp_decided(self) -> None:
        self.t_decided = datetime.utcnow()

    def stamp_executed(self) -> None:
        self.t_executed = datetime.utcnow()

    @property
    def total_ms(self) -> float | None:
        """Total elapsed milliseconds from fetch to last stamped stage."""
        t_end = self.t_executed or self.t_decided or self.t_estimated
        if t_end is None:
            return None
        return (t_end - self.t_fetched).total_seconds() * 1000


class NewsSource(ABC):
    """
    Abstract base class for all news sources.

    Each source must implement `fetch()`, which returns an async iterator
    of NewsItem objects. The `run_forever()` default implementation calls
    fetch() on a configurable interval and puts items into an asyncio.Queue.

    To add a new source:
      1. Subclass NewsSource
      2. Implement fetch()
      3. Set self._poll_interval_seconds in __init__
      4. Register in pipeline/orchestrator.py
    """

    def __init__(self, poll_interval_seconds: int) -> None:
        self._poll_interval_seconds = poll_interval_seconds

    @abstractmethod
    async def fetch(self) -> AsyncIterator[NewsItem]:
        """
        Fetch new items from the source.
        Must be an async generator (use `yield`).
        Should not raise — catch internal errors and log them.
        """
        ...

    async def run_forever(self, queue: asyncio.Queue) -> None:
        """
        Poll the source on a fixed interval and enqueue NewsItem objects.
        Runs until the asyncio task is cancelled.

        Queue backpressure: if the queue is full, items are dropped with a warning.
        This prevents memory explosion if the pipeline falls behind.
        """
        while True:
            try:
                async for item in self.fetch():
                    try:
                        queue.put_nowait(item)
                    except asyncio.QueueFull:
                        # Pipeline is backed up; drop this item rather than block
                        pass
            except Exception:
                # Catch all exceptions to keep the source running
                import logging
                import traceback
                logging.getLogger(self.__class__.__name__).error(
                    "Error fetching from %s:\n%s",
                    self.__class__.__name__,
                    traceback.format_exc(),
                )
            await asyncio.sleep(self._poll_interval_seconds)
