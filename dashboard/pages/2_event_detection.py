"""
Page 2 — Event Detection

Shows the scoring breakdown for each news item:
  - keyword_score (Stage 1: keyword match)
  - nlp_score     (Stage 2: sentence-transformer similarity)
  - event_score   (combined: 0.4 * kw + 0.6 * nlp)

Also shows a score distribution histogram and the detection threshold line.
"""

import time
import streamlit as st
import pandas as pd
from dashboard.db_queries import get_detection_detail, get_score_distribution
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Event Detection")
st.caption(
    f"Items scored by the two-stage event detector. "
    f"Threshold: **{settings.EVENT_DETECTION_MIN_SCORE}** "
    f"(keyword weight: {settings.KEYWORD_WEIGHT:.0%}, NLP weight: {1 - settings.KEYWORD_WEIGHT:.0%})"
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Score Distribution (last 24h)")
    dist_df = get_score_distribution()
    if not dist_df.empty:
        st.bar_chart(
            dist_df.set_index("score_bucket")["count"],
            use_container_width=True,
        )
        # Threshold annotation
        st.caption(f"Threshold at {settings.EVENT_DETECTION_MIN_SCORE} — items to the left are filtered out.")
    else:
        st.info("No data yet.")

with col2:
    st.subheader("Threshold Settings")
    st.metric("Min Score", settings.EVENT_DETECTION_MIN_SCORE)
    st.metric("KW→NLP Threshold", settings.KEYWORD_SCORE_NLP_THRESHOLD)
    st.metric("Keyword Weight", f"{settings.KEYWORD_WEIGHT:.0%}")
    st.metric("NLP Weight", f"{1 - settings.KEYWORD_WEIGHT:.0%}")

st.divider()
st.subheader("Scored Items (most recent 100)")

df = get_detection_detail(limit=100)
if df.empty:
    st.info("No scored items yet.")
else:
    def _row_style(row):
        color = "#ffcccc" if row["filtered_out"] else "#ccffcc"
        return [f"background-color: {color}"] * len(row)

    styled = df.style.apply(_row_style, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "fetched_at":    st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "source":        st.column_config.TextColumn("Source", width="small"),
            "headline":      st.column_config.TextColumn("Headline", width="large"),
            "keyword_score": st.column_config.ProgressColumn("KW Score", min_value=0, max_value=1),
            "nlp_score":     st.column_config.ProgressColumn("NLP Score", min_value=0, max_value=1),
            "event_score":   st.column_config.ProgressColumn("Final Score", min_value=0, max_value=1),
            "filtered_out":  st.column_config.CheckboxColumn("Filtered"),
            "filter_reason": st.column_config.TextColumn("Reason"),
        },
        hide_index=True,
    )

time.sleep(5)
st.rerun()
