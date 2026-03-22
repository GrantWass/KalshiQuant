# KalshiQuant — Signal Expansion TODO

## Reddit (blocked — needs OAuth)

Reddit blocks unauthenticated requests from cloud/container IPs. OAuth app registration requires manual approval from Reddit (self-service was discontinued Nov 2025). Until credentials are obtained, `RedditSource` is disabled in the orchestrator.

To re-enable: apply at https://support.reddithelp.com/hc/en-us/articles/14945211791892, get `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`, add to `.env`, uncomment `RedditSource()` in `pipeline/orchestrator.py`.

---

## Reddit Keyword Search

Poll `/r/all/search.json?q=<term>&sort=new` for Kalshi-relevant keywords (e.g. "federal reserve", "cpi", "hurricane", "election results") across all of Reddit. Catches relevant posts that never surface to the front page of any single subreddit. No auth needed, same public JSON API. Add as a separate `RedditSearchSource` class alongside the existing subreddit poller.

---

## Polymarket Cross-Signal

**Priority: High**

Polymarket is a decentralized prediction market with deep liquidity on political,
economic, and crypto markets. When Polymarket's probability moves on a market that
Kalshi also lists, it's a direct arbitrage signal — Kalshi often hasn't priced in
the same information yet.

### What to build

Create `news/polymarket.py` — a new `NewsSource` that polls the Polymarket API
and emits a `NewsItem` whenever a market's probability shifts significantly.

### Polymarket API

- Base URL: `https://clob.polymarket.com`
- No authentication required for read-only market data
- Key endpoints:
  - `GET /markets` — list all active markets with current prices
  - `GET /markets/{condition_id}` — single market detail
  - `GET /book?token_id={token_id}` — order book (for deeper analysis)

### Implementation sketch

```python
# Poll /markets every 60s
# For each market, track last-seen probability
# Emit a NewsItem when abs(current_prob - last_prob) >= POLYMARKET_SHIFT_THRESHOLD

headline = f"Polymarket: '{market_title}' moved {direction} to {prob:.0%} (was {prev_prob:.0%})"
```

### Signal design decisions to make

1. **Threshold**: What probability shift is worth signaling?
   - Suggested starting point: 3–5% absolute move
   - Smaller = more signals but more noise; larger = fewer but higher confidence

2. **Market mapping**: Polymarket titles ≠ Kalshi titles. Two options:
   - Fuzzy string match on title (fast, brittle)
   - Embed both and use cosine similarity (slower, more robust — already have the infra)
   - Recommended: reuse the FAISS index, search Polymarket title against Kalshi markets

3. **Directionality**: Polymarket moving YES from 40% → 55% means buy YES on Kalshi.
   Encode this in `raw_payload` so the probability estimator can use it directly
   rather than re-deriving from the headline text.

4. **Latency**: Polymarket is on-chain (Polygon). There's a ~1–2 block delay between
   a trade and the API reflecting it. Still faster than news articles.

### Estimated effort

- `news/polymarket.py`: ~150 lines, 2–3 hours
- Market mapping via FAISS: reuse existing embeddings infra, ~1 hour
- Dashboard page update to show Polymarket signals: ~1 hour
