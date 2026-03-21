"""
config/settings.py — Central configuration for KalshiQuant.

All parameters are loaded from environment variables (or a .env file).
Every setting is documented with:
  - What it controls
  - Why the default was chosen
  - How to tune it for different risk/performance profiles

Usage:
    from config.settings import settings
    print(settings.KALSHI_BASE_URL)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── Kalshi API ────────────────────────────────────────────────────────────

    KALSHI_BASE_URL: str = "https://api.elections.kalshi.com/trade-api/v2"
    # WebSocket endpoint for real-time market price updates.
    KALSHI_WS_URL: str = "wss://api.elections.kalshi.com/trade-api/v2/ws"

    # Kalshi account credentials (required).
    KALSHI_EMAIL: str = ""
    KALSHI_PASSWORD: str = ""

    # Kalshi auth tokens expire after 30 minutes (1800s).
    # We refresh proactively at 28 min (1700s) so no in-flight trade ever
    # hits an expired token. Do not raise above 1750s.
    KALSHI_TOKEN_TTL_SECONDS: int = 1700

    # Maximum concurrent HTTP requests to Kalshi API.
    # Kalshi rate limits are not publicly documented; 10 is a safe conservative limit.
    KALSHI_MAX_CONCURRENT_REQUESTS: int = 10

    # ── Database ──────────────────────────────────────────────────────────────

    # asyncpg DSN: postgresql://user:password@host:port/dbname
    DATABASE_URL: str = "postgresql://kalshi:kalshi@localhost:5432/kalshiquant"

    # Connection pool size. For a single-server setup, 10 is enough —
    # the pipeline has one writer per stage, the dashboard reads in sync.
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10

    # ── Event Detection ───────────────────────────────────────────────────────

    # Minimum combined score (0..1) for a news item to continue downstream.
    # Items below this are still written to DB with filtered_out=True so they
    # appear on the dashboard. Lower = more items pass (more CPU + latency);
    # higher = more false negatives (missed market-moving events).
    EVENT_DETECTION_MIN_SCORE: float = 0.35

    # Stage 1 (keyword) minimum score to trigger Stage 2 (NLP).
    # Stage 2 costs ~8ms per item. This gate prevents wasting it on obvious noise.
    KEYWORD_SCORE_NLP_THRESHOLD: float = 0.20

    # Weight of keyword score in the final combined detection score.
    # Final = KEYWORD_WEIGHT * keyword_score + (1 - KEYWORD_WEIGHT) * nlp_score
    KEYWORD_WEIGHT: float = 0.40

    # ── Market Matching ───────────────────────────────────────────────────────

    # Number of candidate markets FAISS returns before applying the score cutoff.
    # Increasing this adds a tiny amount of latency but gives the estimator
    # more markets to evaluate.
    SIMILARITY_TOP_K: int = 5

    # Minimum cosine similarity [0..1] for a market to be considered a match.
    # 0.72 empirically separates "topically related" from "coincidentally similar."
    # Raise to 0.80+ for higher precision (fewer but better matches);
    # lower to 0.65 for higher recall (more matches, more noise).
    SIMILARITY_MIN_SCORE: float = 0.72

    # ── Probability Estimation ────────────────────────────────────────────────

    # Maximum probability shift a single news event can contribute.
    # Even with event_score=1.0 and similarity_score=1.0, P_new moves at most
    # MAX_SHIFT from P_market. Prevents runaway estimates from unusual events.
    # Formula: base_shift = event_score × similarity_score × MAX_SHIFT
    MAX_SHIFT: float = 0.15

    # Recency decay half-life in minutes.
    # recency_factor = exp(-age_minutes / RECENCY_HALF_LIFE_MINUTES), clipped to [0.2, 1.0]
    # At age=HALF_LIFE, the item contributes 50% of a fresh article's shift.
    # Lower values penalize old news more aggressively.
    RECENCY_HALF_LIFE_MINUTES: float = 30.0

    # ── Decision Engine ───────────────────────────────────────────────────────

    # Minimum absolute probability edge (P_new - P_market) to pass the first gate.
    # 0.04 = 4 cents. Below this, expected value doesn't reliably cover
    # Kalshi trading fees plus slippage in thin orderbooks.
    # If you're getting too few trades: lower to 0.03. Too many: raise to 0.06.
    PROBABILITY_SHIFT_MIN: float = 0.04

    # Minimum model confidence [0..1] to pass the second gate.
    # Confidence = harmonic_mean(event_score, similarity_score) × recency_factor.
    # Harmonic mean penalizes imbalanced scores (e.g., high event score but
    # weak market match, or vice versa).
    CONFIDENCE_MIN: float = 0.60

    # Maximum contracts held in a single market at any time.
    # Protects against illiquid orderbooks where a large position would move
    # the price against you upon entry or exit.
    MAX_POSITION_PER_MARKET: int = 50

    # Total portfolio exposure cap in USD.
    # Computed as sum(contracts × price_cents / 100) across all open positions.
    # This is your maximum dollar amount at risk at any moment.
    MAX_TOTAL_EXPOSURE_USD: float = 500.0

    # Do not trade in markets closing within this many minutes.
    # Very short time-to-close means the orderbook is often thin and
    # there's insufficient time to realize the edge.
    MIN_MINUTES_TO_CLOSE: int = 30

    # Order sizing uses quarter-Kelly fraction for safety.
    # Full Kelly maximizes long-run growth but has very high variance.
    # Quarter-Kelly (0.25) is a common conservative choice.
    KELLY_FRACTION: float = 0.25

    # ── FAISS Index ───────────────────────────────────────────────────────────

    # Path where the FAISS index binary is persisted (inside Docker volume).
    FAISS_INDEX_PATH: str = "faiss_data/market_index.bin"

    # Path where market metadata (ticker → title, category, close_time) is stored.
    FAISS_METADATA_PATH: str = "faiss_data/market_meta.json"

    # How often (in seconds) to rebuild the FAISS index from Kalshi's open markets.
    # 900s = 15 min keeps the index fresh without hammering the Kalshi API.
    # Reduce to 300s if markets open/close very frequently.
    MARKET_INDEX_REFRESH_INTERVAL: int = 900

    # ── Embedding Model ───────────────────────────────────────────────────────

    # sentence-transformers model to use for all text embeddings.
    # all-MiniLM-L6-v2: 384 dimensions, ~22MB, ~8ms per encode on CPU.
    # Good balance of speed and quality. For higher accuracy at cost of speed,
    # try all-mpnet-base-v2 (768 dim) — requires changing EMBEDDING_DIM.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Must match the output dimension of EMBEDDING_MODEL.
    EMBEDDING_DIM: int = 384

    # Batch size for encoding many texts at once (used during index builds).
    EMBEDDING_BATCH_SIZE: int = 64

    # ── News Source Poll Intervals (seconds) ──────────────────────────────────

    # NWS: 30s gives near-real-time weather alerts with no rate limit risk.
    NWS_POLL_INTERVAL: int = 30

    # GDELT: updates every 15 min; polling faster than 60s wastes requests.
    GDELT_POLL_INTERVAL: int = 60

    # RSS (AP, BBC): articles typically appear 15-30 min after publication.
    RSS_POLL_INTERVAL: int = 90


    # ── Third-Party API Keys ──────────────────────────────────────────────────


    # ── Monitoring ────────────────────────────────────────────────────────────

    # End-to-end latency target in milliseconds (news fetch → order placed).
    # Used by the dashboard to highlight slow pipeline runs in red.
    LATENCY_TARGET_MS: int = 5000

    # Deduplicator in-memory LRU set max size.
    # Each entry is a (source, id) tuple. At ~50 bytes each, 50k = ~2.5MB.
    DEDUP_MAX_MEMORY_ENTRIES: int = 50_000

    # ── Dry Run ───────────────────────────────────────────────────────────────

    # When True, the decision engine logs trades but does NOT call Kalshi API.
    # Use for testing the full pipeline without risking real money.
    DRY_RUN: bool = False


# Module-level singleton — import this everywhere.
settings = Settings()
