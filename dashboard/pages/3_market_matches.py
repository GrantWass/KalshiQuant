"""
Page 3 — Market Matches

Shows FAISS similarity search results: for each detected event,
which Kalshi markets were matched and at what similarity score.

Similarity score color coding:
  >= 0.85  → green (strong match)
  0.72-0.85 → yellow (acceptable match)
  < 0.72  → would have been filtered out (not shown, was filtered before DB write)
"""

import time
import streamlit as st
from dashboard.db_queries import get_recent_matches
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Market Matches")
st.caption(
    f"FAISS cosine similarity search results. "
    f"Minimum similarity: **{settings.SIMILARITY_MIN_SCORE}** "
    f"| Top-K: **{settings.SIMILARITY_TOP_K}**"
)

df = get_recent_matches(limit=150)

if df.empty:
    st.info("No market matches yet. Waiting for events to pass detection threshold.")
else:
    # Color similarity scores
    def _similarity_color(val):
        if val is None:
            return ""
        if val >= 0.85:
            return "background-color: #ccffcc"
        if val >= 0.72:
            return "background-color: #ffffcc"
        return "background-color: #ffcccc"

    styled = df.style.applymap(_similarity_color, subset=["similarity_score"])

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "matched_at":       st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "source":           st.column_config.TextColumn("Source", width="small"),
            "headline":         st.column_config.TextColumn("Headline", width="large"),
            "market_ticker":    st.column_config.TextColumn("Ticker", width="medium"),
            "market_title":     st.column_config.TextColumn("Market", width="large"),
            "market_category":  st.column_config.TextColumn("Category", width="small"),
            "similarity_score": st.column_config.NumberColumn("Similarity", format="%.3f"),
        },
        hide_index=True,
    )

    # Summary stats
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches (shown)", len(df))
    col2.metric("Avg Similarity", f"{df['similarity_score'].mean():.3f}" if not df.empty else "—")
    col3.metric("Strong Matches (≥0.85)", int((df["similarity_score"] >= 0.85).sum()))

time.sleep(5)
st.rerun()
