"""
Page 1 — News Feed

Shows every news item ingested from every source, ordered by fetch time.
Color coding:
  Green row  = item passed event detection and continued downstream
  Red row    = item was filtered out (event_score too low or duplicate)

Columns:
  fetched_at    — when the pipeline received it
  source        — which source produced it (nws, gdelt, rss_ap, etc.)
  headline      — the headline text (click URL if available)
  event_score   — combined detection score (None = deduped before scoring)
  filtered_out  — True/False
  filter_reason — why it was filtered (if applicable)
"""

import time
import streamlit as st
from dashboard.db_queries import get_recent_news, get_news_source_counts

st.set_page_config(layout="wide")
st.title("News Feed")
st.caption("All news items ingested from all sources. Green = passed detection. Red = filtered out.")

# ── Source summary ─────────────────────────────────────────────────────────────
counts = get_news_source_counts()
if not counts.empty:
    cols = st.columns(len(counts))
    for i, (_, row) in enumerate(counts.iterrows()):
        cols[i].metric(
            label=row["source"],
            value=f"{int(row['total'])} total",
            delta=f"{int(row['passed'])} passed",
        )
    st.divider()

# ── News table ─────────────────────────────────────────────────────────────────
df = get_recent_news(limit=200)

if df.empty:
    st.info("No news items yet. The pipeline may still be starting up.")
else:
    # Color rows: red for filtered, green for passed
    def _row_style(row):
        color = "#ffcccc" if row["filtered_out"] else "#ccffcc"
        return [f"background-color: {color}"] * len(row)

    styled = df.style.apply(_row_style, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "fetched_at":    st.column_config.DatetimeColumn("Fetched At", format="HH:mm:ss"),
            "source":        st.column_config.TextColumn("Source", width="small"),
            "headline":      st.column_config.TextColumn("Headline", width="large"),
            "event_score":   st.column_config.NumberColumn("Score", format="%.3f"),
            "keyword_score": st.column_config.NumberColumn("KW Score", format="%.3f"),
            "nlp_score":     st.column_config.NumberColumn("NLP Score", format="%.3f"),
            "filtered_out":  st.column_config.CheckboxColumn("Filtered"),
            "filter_reason": st.column_config.TextColumn("Filter Reason"),
            "url":           st.column_config.LinkColumn("URL"),
        },
        hide_index=True,
    )

time.sleep(5)
st.rerun()
