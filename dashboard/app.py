"""
dashboard/app.py — KalshiQuant monitoring dashboard entry point.

Launch with:
    streamlit run dashboard/app.py

Or via Docker:
    docker-compose up dashboard

Features:
  - Auto-refreshes every 5 seconds
  - Pipeline health indicator in the sidebar
  - Multi-page navigation (pages/ directory)
  - All data sourced exclusively from PostgreSQL (never touches pipeline memory)
"""

import time
import streamlit as st

st.set_page_config(
    page_title="KalshiQuant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Page header ────────────────────────────────────────────────────────────────
st.title("KalshiQuant — Trading Dashboard")
st.caption("Real-time visibility into news ingestion, event detection, market matching, and trade decisions.")

# ── Pipeline health sidebar ────────────────────────────────────────────────────
from dashboard.db_queries import get_pipeline_health
from datetime import datetime, timezone

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
            st.success(f"Running  (last heartbeat {int(age_seconds)}s ago)")
        else:
            st.error(f"Stale  (last heartbeat {int(age_seconds)}s ago)")

    st.divider()
    st.metric("News Fetched", health.get("news_fetched", "—"))
    st.metric("Events Detected", health.get("events_detected", "—"))
    st.metric("Markets Matched", health.get("markets_matched", "—"))
    st.metric("Trades Executed", health.get("trades_executed", "—"))
    st.metric("Trades Rejected", health.get("trades_rejected", "—"))

    st.divider()
    st.caption("Auto-refreshes every 5 seconds.")
    st.caption("Navigate using the pages in the sidebar.")

# ── Navigation hint ────────────────────────────────────────────────────────────
st.info(
    "Use the **sidebar** to navigate between pages:\n"
    "- **News Feed** — live news stream\n"
    "- **Event Detection** — scoring breakdown\n"
    "- **Market Matches** — FAISS similarity results\n"
    "- **Trade Decisions** — executed and rejected trades\n"
    "- **Positions** — open positions and P&L\n"
    "- **Latency** — per-stage pipeline timing"
)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
time.sleep(5)
st.rerun()
