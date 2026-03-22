"""
dashboard/app.py — KalshiQuant monitoring dashboard entry point.
"""

import time
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="KalshiQuant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.db_queries import get_pipeline_health, get_pipeline_funnel, get_detection_detail

# ── Sidebar ─────────────────────────────────────────────────────────────────────
health = get_pipeline_health()

with st.sidebar:
    st.header("Pipeline Status")
    if not health.get("last_heartbeat"):
        st.error("Pipeline not running")
    else:
        hb = health["last_heartbeat"]
        if isinstance(hb, str):
            hb = datetime.fromisoformat(hb)
        age_seconds = (datetime.utcnow() - hb.replace(tzinfo=None)).total_seconds()
        if age_seconds < 120:
            st.success(f"Running — heartbeat {int(age_seconds)}s ago")
        else:
            st.error(f"Stale — heartbeat {int(age_seconds)}s ago")

    st.divider()
    st.metric("News Fetched",     health.get("news_fetched", "—"))
    st.metric("Events Detected",  health.get("events_detected", "—"))
    st.metric("Markets Matched",  health.get("markets_matched", "—"))
    st.metric("Trades Executed",  health.get("trades_executed", "—"))
    st.metric("Trades Rejected",  health.get("trades_rejected", "—"))
    st.divider()
    st.caption("Auto-refreshes every 10 seconds.")

# ── Header ──────────────────────────────────────────────────────────────────────
st.title("KalshiQuant")
st.caption("Real-time news-driven prediction market trading system.")
st.divider()

# ── Today's funnel ──────────────────────────────────────────────────────────────
funnel = get_pipeline_funnel()

st.subheader("Today's Pipeline Funnel")

c1, c2, c3, c4, c5 = st.columns(5)

news_total   = int(funnel.get("news_total", 0))
events_passed = int(funnel.get("events_passed", 0))
matches      = int(funnel.get("matches", 0))
executed     = int(funnel.get("executed", 0))
rejected     = int(funnel.get("rejected", 0))

def _pct(num, denom):
    return f"{num/denom*100:.0f}% pass-through" if denom > 0 else "—"

c1.metric("News Ingested",     news_total)
c2.metric("Events Detected",   events_passed,   delta=_pct(events_passed, news_total),   delta_color="off")
c3.metric("Market Matches",    matches,         delta=_pct(matches, events_passed),       delta_color="off")
c4.metric("Trades Executed",   executed,        delta=_pct(executed, matches),            delta_color="off")
c5.metric("Trades Rejected",   rejected)

st.divider()

# ── Recent events that passed detection ─────────────────────────────────────────
st.subheader("Recent Events")
st.caption("Latest news items that passed event detection and entered the pipeline.")

recent = get_detection_detail(limit=10)

if recent.empty:
    st.info("No events yet — the pipeline may still be starting up or processing.")
else:
    for _, row in recent.iterrows():
        if row.get("filtered_out"):
            continue
        with st.container(border=True):
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                label = f"[{row['headline']}]({row['url']})" if row.get("url") else row["headline"]
                st.markdown(label)
                st.caption(f"{row['source']} · {row['fetched_at']}")
            with col2:
                st.metric("Score", f"{row['event_score']:.2f}" if row.get('event_score') else "—")
            with col3:
                st.metric("KW / NLP", f"{row.get('keyword_score', 0):.2f} / {row.get('nlp_score', 0):.2f}")

# ── Auto-refresh ────────────────────────────────────────────────────────────────
time.sleep(10)
st.rerun()
