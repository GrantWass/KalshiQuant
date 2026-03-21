# KalshiQuant

A real-time, event-driven trading system that monitors free news sources, detects market-moving events, and places trades on [Kalshi](https://kalshi.com) prediction markets before the information is fully priced in.

---

## How It Works

```
Real-time news (RSS, GDELT, NWS, GNews, NewsData)
      ↓  [fetch]
Event Detection  (keyword filter + NLP sentence-transformer scoring)
      ↓  [detect]
Market Matching  (FAISS cosine similarity search over all open Kalshi markets)
      ↓  [match]
Probability Update  (Bayesian-style edge estimation vs. current market price)
      ↓  [estimate]
Decision Engine  (5-gate risk checks: edge, confidence, position limits, exposure, time)
      ↓  [decide]
Order Execution  (Kalshi REST API)
      ↓
Monitoring Dashboard  (Streamlit — every event, score, match, and decision visible)
```

Every item that enters the pipeline — including events that are **filtered out or rejected** — is recorded in PostgreSQL and visible in the dashboard. You can always see exactly why a trade was or was not made.

---

## Directory Structure

```
KalshiQuant/
├── README.md
├── requirements.txt             # All Python dependencies
├── .env.example                 # Template for environment variables
├── docker-compose.yml           # postgres + trader + dashboard services
├── Dockerfile                   # Trading pipeline container
├── Dockerfile.dashboard         # Streamlit dashboard container
│
├── config/
│   └── settings.py              # All configuration via Pydantic BaseSettings
│                                #   Every parameter is documented with purpose + tuning guidance
│
├── kalshi/
│   ├── auth.py                  # Token manager — proactive refresh before 30-min expiry
│   ├── client.py                # Async REST client (aiohttp, retry, rate-limiting)
│   ├── websocket.py             # WebSocket feed for real-time market price updates
│   └── models.py                # Pydantic models for all Kalshi API types
│
├── news/
│   ├── base.py                  # NewsItem dataclass + PipelineTrace + abstract NewsSource
│   ├── gdelt.py                 # GDELT global event stream (streams CSV, no full download)
│   ├── rss.py                   # RSS feeds: AP News, BBC (feedparser + aiohttp)
│   ├── gnews.py                 # GNews API — quota-aware (100 req/day free)
│   ├── nws.py                   # National Weather Service alerts (30s poll, no auth)
│   └── newsdata.py              # NewsData.io — quota-aware (200 credits/day free)
│
├── pipeline/
│   ├── orchestrator.py          # asyncio.TaskGroup wiring: news → dedup → detect →
│   │                            #   match → estimate → decide → execute
│   ├── deduplicator.py          # In-memory LRU set + DB check to suppress duplicate items
│   ├── event_detector.py        # Stage 1: keyword scoring; Stage 2: NLP embedding score
│   ├── market_matcher.py        # FAISS cosine similarity search against open markets
│   ├── probability_estimator.py # Computes P_new and edge vs. current market price
│   └── decision_engine.py       # 5-gate risk check; records ALL decisions (executed + rejected)
│
├── embeddings/
│   ├── encoder.py               # sentence-transformers wrapper (all-MiniLM-L6-v2, 384-dim)
│   ├── index.py                 # FAISS IndexFlatIP manager: build, search, atomic refresh
│   └── market_index_builder.py  # Fetches all open Kalshi markets, builds + periodically refreshes
│
├── db/
│   ├── pool.py                  # asyncpg connection pool singleton
│   ├── schema.sql               # Full DDL — run once at startup via docker-compose
│   └── repositories/
│       ├── news.py              # NewsEvent insert/query
│       ├── decisions.py         # TradeDecision insert/query
│       ├── positions.py         # Position tracking + P&L
│       └── metrics.py          # Latency metric insert/query
│
├── dashboard/
│   ├── app.py                   # Streamlit entry point (5s auto-refresh)
│   ├── db_queries.py            # All read-only SQL → pd.DataFrame
│   └── pages/
│       ├── 1_news_feed.py       # Live news stream: green = passed, red = filtered out
│       ├── 2_event_detection.py # Keyword + NLP scores per item
│       ├── 3_market_matches.py  # FAISS similarity results per detected event
│       ├── 4_trade_decisions.py # Executed vs. rejected trades (with rejection reasons)
│       ├── 5_positions.py       # Open positions + unrealized/realized P&L
│       └── 6_latency.py        # Per-stage timestamps, Gantt table, P95 alert
│
└── scripts/
    ├── build_index.py           # One-shot: fetch all open markets + build FAISS index
    └── simulate_event.py        # Inject a synthetic news event to test the full pipeline
```

---

## News Sources (Free Only)

| Source | Poll Interval | Typical Latency | Daily Limit | Auth Required |
|--------|:------------:|:--------------:|:-----------:|:-------------:|
| National Weather Service | 30s | ~100ms | Unlimited | None |
| GDELT | 60s | 1–10s | Unlimited | None |
| RSS (AP News, BBC) | 90s | 15–30 min pub delay | Unlimited | None |
| GNews API | 300s | 100–200ms | 100 req | Free API key |
| NewsData.io | 432s | Real-time | 200 credits | Free API key |

### Expanding to More Sources

The system is designed to be extended. Each source implements the `NewsSource` abstract base class in `news/base.py` — adding a new one is ~50 lines.

| Source | Cost | Benefit | Use Case |
|--------|------|---------|----------|
| **Twitter/X Filtered Stream** | ~$100/mo (Basic tier) | Best latency for breaking news | Political/sports markets |
| **Polygon.io News** | Free tier available | Financial news with metadata | Economic/earnings markets |
| **Alpha Vantage News** | Free (25 req/day) | Earnings + macro news | Economic markets |
| **PredictIt / Metaculus APIs** | Free | Calibration data from other prediction markets | Cross-market probability signals |
| **Bloomberg Terminal API** | Institutional ($) | Highest quality financial news | Macro/commodities |
| **Refinitiv/LSEG Eikon** | Institutional ($) | Best for commodities + FX | Commodities/currency markets |

---

## Configuration Reference

All configuration lives in `config/settings.py` and is loaded from environment variables (`.env` file). Every parameter has a comment explaining **what it does**, **why the default was chosen**, and **how to tune it**.

### Event Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EVENT_DETECTION_MIN_SCORE` | `0.35` | Minimum combined score for an item to pass event detection. Items below this are still recorded in DB (`filtered_out=True`) for dashboard visibility. Lower = more items pass (higher CPU/latency); higher = more false negatives. |

### Market Matching

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMILARITY_TOP_K` | `5` | Number of candidate markets FAISS returns per news event before applying the score cutoff. |
| `SIMILARITY_MIN_SCORE` | `0.72` | Minimum cosine similarity `[0..1]` for a market to count as a valid match. `0.72` empirically separates "topically related" from "coincidentally similar." Raise to `0.80+` for precision; lower for recall. |

### Probability & Edge

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROBABILITY_SHIFT_MIN` | `0.04` | Minimum absolute edge `(P_new - P_market)` to proceed past the decision gate. `0.04` = 4 cents. Below this, expected value doesn't cover Kalshi fees + slippage. |
| `MAX_SHIFT` | `0.15` | Maximum probability shift a single event can contribute. Caps `P_new` update to prevent runaway estimates from unusual high-confidence events. |

### Risk / Position Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_POSITION_PER_MARKET` | `50` | Maximum contracts held in a single market at any time. Protects against illiquid orderbooks moving against large positions. |
| `MAX_TOTAL_EXPOSURE_USD` | `500.0` | Total portfolio exposure cap. Computed as sum of `(contracts × price_cents / 100)` across all open positions. |
| `CONFIDENCE_MIN` | `0.60` | Minimum model confidence `[0..1]` to place an order. Computed as `harmonic_mean(event_score, similarity_score) × recency_factor`. Harmonic mean penalizes imbalanced scores. |

### Infrastructure

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARKET_INDEX_REFRESH_INTERVAL` | `900` | Seconds between FAISS index rebuilds. `900s = 15 min` keeps the index fresh without hammering the Kalshi API. Reduce if many markets open/close rapidly. |
| `KALSHI_TOKEN_TTL_SECONDS` | `1700` | Seconds between token refreshes. Kalshi tokens expire at 30 min (1800s); we refresh at 28 min (1700s) to ensure no trade ever hits an expired token. |

---

## Probability Model

The core question for each `(news event, market)` pair:

> Given this news, how much should the market probability shift?

### Formula

```
P_new = P_market + direction × base_shift

base_shift = event_score × similarity_score × MAX_SHIFT

edge = P_new - P_market

confidence = harmonic_mean(event_score, similarity_score) × recency_factor
```

### Variable Definitions

| Variable | Description |
|----------|-------------|
| `P_market` | Current market price (YES price = implied probability), sourced from Kalshi WebSocket |
| `direction` | `+1` if news increases the probability; `-1` if it decreases it. Determined by headline sentiment polarity. |
| `event_score` | How market-moving is this news? `[0..1]`, from the two-stage detector. |
| `similarity_score` | How relevant is this news to this market? `[0..1]`, cosine similarity from FAISS. |
| `MAX_SHIFT` | Config cap `(0.15)`. Even a perfect signal can only shift our estimate by 15 percentage points. |
| `recency_factor` | `exp(-age_minutes / 30)`, clipped to `[0.2, 1.0]`. 30-min-old news contributes ~50% of a fresh article's shift. |

### Worked Example

```
Event:  NWS alert — "Hurricane Warning issued for Florida Gulf Coast"
Market: "Will a hurricane make landfall in the Gulf Coast in September 2026?"
        Current price = 0.35 (market implies 35% probability)

event_score      = 0.85   (strong keyword hit: "hurricane warning" + high NLP score)
similarity_score = 0.91   (very close semantic match to market title)
direction        = +1     (warning increases landfall probability)
age_minutes      = 1      → recency_factor = exp(-1/30) ≈ 0.97

base_shift = 0.85 × 0.91 × 0.15 = 0.116
P_new      = 0.35 + 0.116 = 0.466
edge       = +0.116   ← above PROBABILITY_SHIFT_MIN (0.04) → passes first gate

confidence = harmonic_mean(0.85, 0.91) × 0.97
           = 0.879 × 0.97 = 0.853   ← above CONFIDENCE_MIN (0.60) → passes second gate

Decision:  BUY YES, quarter-Kelly sizing → ~12 contracts at 0.36
```

---

## Pipeline Architecture

Each news item carries a `PipelineTrace` with timestamps at every stage. These are written to the `latency_metrics` table and shown on Dashboard Page 6.

```
T+0ms    Source fetches news item → t_fetched stamped
             ↓
T+1ms    Deduplicator: (source, id) not seen → t_deduped stamped
             ↓
T+52ms   EventDetector: keyword (0.1ms) + NLP (8ms) → t_detected stamped
             ↓
T+62ms   MarketMatcher: encode (8ms) + FAISS search (2ms) → t_matched stamped
             ↓
T+64ms   ProbabilityEstimator: WebSocket price lookup (0ms) + math → t_estimated stamped
             ↓
T+65ms   DecisionEngine: 5 gates, order sizing → t_decided stamped
             ↓
T+400ms  KalshiClient.place_order() HTTP POST → t_executed stamped

Total processing latency: ~400ms
(Dominant factor is polling interval, not pipeline — NWS items arrive within 30s of real events)
```

---

## Dashboard

Open at **http://localhost:8501** after starting docker-compose. Auto-refreshes every 5 seconds.

| Page | What You See |
|------|-------------|
| **1 — News Feed** | Every item from every source. Green = passed event detection. Red = filtered out. Shows scores. |
| **2 — Event Detection** | Keyword score + NLP score breakdown per item. Score distribution histogram. |
| **3 — Market Matches** | For each detected event: which markets matched and their similarity scores. |
| **4 — Trade Decisions** | Two columns: **Executed** (edge, confidence, contracts) \| **Rejected** (bullet-list of exactly why not). |
| **5 — Positions** | Total exposure, unrealized P&L, realized P&L. Per-market position table. |
| **6 — Latency** | Gantt-style table of stage timestamps per event. Stage breakdown bar chart. P95 alert if > 5s. |

---

## Running Locally

### Prerequisites
- Docker + docker-compose
- Python 3.11+
- A free [GNews API key](https://gnews.io) and [NewsData.io key](https://newsdata.io) (both have free tiers)
- A [Kalshi](https://kalshi.com) account

### Setup

```bash
# 1. Clone and configure
git clone https://github.com/GrantWass/KalshiQuant.git
cd KalshiQuant
cp .env.example .env
# Edit .env with your Kalshi credentials and API keys

# 2. Start PostgreSQL
docker-compose up postgres -d

# 3. Build the FAISS market index (fetches all open Kalshi markets)
pip install -r requirements.txt
python scripts/build_index.py

# 4. Start everything
docker-compose up -d

# 5. Open the dashboard
open http://localhost:8501
```

### E2E Simulation (no real trades placed)

```bash
# Inject a synthetic news event and watch it flow through the pipeline
python scripts/simulate_event.py \
  --headline "Hurricane Category 5 makes landfall in Florida" \
  --source nws \
  --dry-run   # skips actual order placement
```

Watch Dashboard Page 4 for the decision to appear within ~5 seconds.

---

## Testing

```bash
# Unit tests (no external dependencies)
pytest tests/unit/ -v

# Integration tests (require .env with real credentials)
pytest tests/integration/ -v

# Key unit test assertions:
#   EventDetector: "Hurricane Category 5 Florida" → event_score > 0.35
#   ProbabilityEstimator: P_market=0.40, event_score=0.85, sim=0.91 → edge > 0.04
#   DecisionEngine: position at MAX → REJECT with "position limit reached" in reasons
#   DecisionEngine: edge=0.02 → REJECT with "edge below minimum" in reasons
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Kalshi token expiry mid-trade | Proactive refresh at 28 min + `force_refresh()` on any 401 response |
| GDELT CSV memory usage | Stream-parse with `aiohttp` chunked reads — never load full file into memory |
| GNews/NewsData quota exhaustion | Track daily usage in DB; skip source automatically when quota reached |
| FAISS index stale during refresh | `asyncio.Lock` held during 15-min atomic index swap |
| False signals → bad trades | 5-gate decision engine + quarter-Kelly position sizing |
| Dashboard-pipeline coupling | Dashboard reads only from PostgreSQL — never touches pipeline in-memory state |
| Thin Kalshi orderbooks | `MAX_POSITION_PER_MARKET` cap limits price impact; quarter-Kelly keeps orders small |
