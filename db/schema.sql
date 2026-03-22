-- KalshiQuant PostgreSQL Schema
-- All timestamps are UTC (TIMESTAMPTZ).
-- Run via docker-compose (mounted as /docker-entrypoint-initdb.d/schema.sql).

-- ── News Events ──────────────────────────────────────────────────────────────
-- Every item ingested from every source is recorded here, including items
-- that are filtered out before reaching the market matcher. This gives full
-- visibility into what the pipeline is seeing and why items are rejected.

CREATE TABLE IF NOT EXISTS news_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source identity
    source          VARCHAR(32)  NOT NULL,   -- 'gdelt' | 'rss_bbc' | 'rss_npr' | 'nws'
    source_id       VARCHAR(512) NOT NULL,   -- source-native ID for deduplication

    -- Content
    headline        TEXT NOT NULL,
    body            TEXT,
    url             TEXT,

    -- Timestamps
    published_at    TIMESTAMPTZ NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Event detection scores (NULL if item was deduped before scoring)
    event_score     FLOAT,        -- combined final score [0..1]
    keyword_score   FLOAT,        -- Stage 1: keyword-based score
    nlp_score       FLOAT,        -- Stage 2: sentence-transformer similarity score

    -- Filter outcome
    filtered_out    BOOLEAN      NOT NULL DEFAULT FALSE,
    filter_reason   TEXT,         -- e.g. "event_score 0.12 < min 0.35" or "duplicate"

    -- Original payload for auditing
    raw_payload     JSONB        NOT NULL DEFAULT '{}',

    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_news_events_fetched_at     ON news_events (fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_events_filtered_out   ON news_events (filtered_out, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_events_source         ON news_events (source, fetched_at DESC);


-- ── Market Matches ────────────────────────────────────────────────────────────
-- FAISS cosine similarity results for each detected event.
-- One news_event can match multiple markets; each match is a separate row.

CREATE TABLE IF NOT EXISTS market_matches (
    id               UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
    news_event_id    UUID  NOT NULL REFERENCES news_events(id) ON DELETE CASCADE,

    market_ticker    VARCHAR(128) NOT NULL,
    market_title     TEXT         NOT NULL,
    market_category  VARCHAR(64),
    similarity_score FLOAT        NOT NULL,  -- cosine similarity [0..1]
    below_threshold  BOOLEAN      NOT NULL DEFAULT FALSE,  -- TRUE = near-miss, did not pass SIMILARITY_MIN_SCORE

    matched_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_matches_news_event_id ON market_matches (news_event_id);
CREATE INDEX IF NOT EXISTS idx_market_matches_matched_at    ON market_matches (matched_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_matches_ticker        ON market_matches (market_ticker, matched_at DESC);


-- ── Trade Decisions ───────────────────────────────────────────────────────────
-- EVERY decision — both EXECUTE and REJECT — is recorded here.
-- Rejected trades include an array of reasons explaining each failed gate.
-- This table is the backbone of Dashboard Page 4 (decision visibility).

CREATE TABLE IF NOT EXISTS trade_decisions (
    id               UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    news_event_id    UUID    NOT NULL REFERENCES news_events(id) ON DELETE CASCADE,

    market_ticker    VARCHAR(128) NOT NULL,
    market_title     TEXT         NOT NULL,

    -- Decision outcome
    action           VARCHAR(16)  NOT NULL,  -- 'EXECUTE' | 'REJECT'
    side             VARCHAR(4),             -- 'YES' | 'NO' | NULL if rejected before sizing

    -- Order details (NULL if rejected)
    contracts        INTEGER,
    price_cents      INTEGER,                -- limit price in cents (e.g. 43 = $0.43)

    -- Probability analysis
    edge             FLOAT NOT NULL,         -- P_new - P_market (can be negative)
    p_market         FLOAT NOT NULL,         -- market price at decision time
    p_estimated      FLOAT NOT NULL,         -- our estimated probability
    confidence       FLOAT NOT NULL,         -- harmonic_mean(event_score, sim_score) × recency

    -- Rejection detail (NULL if executed; array of reason strings if rejected)
    rejection_reasons  TEXT[],

    -- Kalshi API response (NULL if rejected or dry-run)
    kalshi_order_id  VARCHAR(128),
    kalshi_status    VARCHAR(32),

    decided_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trade_decisions_decided_at ON trade_decisions (decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_action     ON trade_decisions (action, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_ticker     ON trade_decisions (market_ticker, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_decisions_news       ON trade_decisions (news_event_id);


-- ── Positions ─────────────────────────────────────────────────────────────────
-- Current open positions synced from Kalshi API + updated by the order executor.

CREATE TABLE IF NOT EXISTS positions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_ticker       VARCHAR(128) NOT NULL UNIQUE,
    market_title        TEXT         NOT NULL,

    side                VARCHAR(4)   NOT NULL,   -- 'YES' | 'NO'
    contracts           INTEGER      NOT NULL DEFAULT 0,
    avg_price_cents     FLOAT        NOT NULL,   -- average entry price in cents

    -- Updated from WebSocket price feed
    current_price_cents FLOAT,
    unrealized_pnl_cents FLOAT,                  -- (current - avg) × contracts

    -- Accumulated from settled trades
    realized_pnl_cents  FLOAT        NOT NULL DEFAULT 0,

    last_updated        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_updated ON positions (last_updated DESC);


-- ── Latency Metrics ───────────────────────────────────────────────────────────
-- Per-event timestamps at each pipeline stage, enabling stage-by-stage
-- latency analysis on Dashboard Page 6.
-- Stage durations are computed by the dashboard from the timestamp columns.

CREATE TABLE IF NOT EXISTS latency_metrics (
    id               UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    news_event_id    UUID    REFERENCES news_events(id) ON DELETE SET NULL,

    source           VARCHAR(32)  NOT NULL,

    -- Absolute timestamps for each stage (UTC).
    -- NULL if the item did not reach that stage (e.g. filtered_out at detection).
    t_fetched        TIMESTAMPTZ  NOT NULL,   -- NewsItem created by source
    t_deduped        TIMESTAMPTZ,             -- passed deduplication check
    t_detected       TIMESTAMPTZ,             -- EventDetector scored it
    t_matched        TIMESTAMPTZ,             -- MarketMatcher returned results
    t_estimated      TIMESTAMPTZ,             -- ProbabilityEstimator computed edge
    t_decided        TIMESTAMPTZ,             -- DecisionEngine made EXECUTE/REJECT
    t_executed       TIMESTAMPTZ,             -- KalshiClient.place_order() returned (NULL if rejected)

    -- Derived durations in milliseconds (computed + stored for query performance).
    -- Each value is the duration of that stage, not cumulative.
    ms_dedup         FLOAT,   -- t_deduped - t_fetched
    ms_detect        FLOAT,   -- t_detected - t_deduped
    ms_match         FLOAT,   -- t_matched - t_detected
    ms_estimate      FLOAT,   -- t_estimated - t_matched
    ms_decide        FLOAT,   -- t_decided - t_estimated
    ms_execute       FLOAT,   -- t_executed - t_decided (NULL if rejected)
    ms_total         FLOAT    -- (t_executed or t_decided) - t_fetched
);

CREATE INDEX IF NOT EXISTS idx_latency_metrics_fetched_at ON latency_metrics (t_fetched DESC);
CREATE INDEX IF NOT EXISTS idx_latency_metrics_total_ms   ON latency_metrics (ms_total DESC);


-- ── Market Embeddings ─────────────────────────────────────────────────────────
-- Metadata for every market in the current FAISS index.
-- Used to resolve FAISS result indices back to market tickers and titles.

CREATE TABLE IF NOT EXISTS market_embeddings (
    ticker          VARCHAR(128) PRIMARY KEY,
    title           TEXT         NOT NULL,
    subtitle        TEXT,
    category        VARCHAR(64),
    tags            TEXT[],
    close_time      TIMESTAMPTZ,

    faiss_index_id  INTEGER      NOT NULL,   -- position in the current FAISS index
    embedding_text  TEXT         NOT NULL,   -- the concatenated text that was embedded
    last_indexed    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_embeddings_faiss_id ON market_embeddings (faiss_index_id);


-- ── Pipeline Runs ─────────────────────────────────────────────────────────────
-- Heartbeat table updated every 60s by the running orchestrator.
-- Dashboard uses last_heartbeat to detect if the pipeline is alive.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    healthy          BOOLEAN     NOT NULL DEFAULT TRUE,

    -- Cumulative counters since pipeline start
    news_fetched     INTEGER     NOT NULL DEFAULT 0,
    events_detected  INTEGER     NOT NULL DEFAULT 0,
    markets_matched  INTEGER     NOT NULL DEFAULT 0,
    trades_executed  INTEGER     NOT NULL DEFAULT 0,
    trades_rejected  INTEGER     NOT NULL DEFAULT 0,

    last_heartbeat   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
