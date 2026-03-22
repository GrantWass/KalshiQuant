"""
Page 2 — Event Detection
"""

import time
import streamlit as st
from dashboard.db_queries import get_detection_detail, get_score_distribution
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Event Detection")
st.caption(
    f"Two-stage scorer: keyword match + NLP similarity. "
    f"Threshold: **{settings.EVENT_DETECTION_MIN_SCORE}** "
    f"(keyword weight {settings.KEYWORD_WEIGHT:.0%}, NLP weight {1 - settings.KEYWORD_WEIGHT:.0%})"
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Score Distribution (last 24h)")
    dist_df = get_score_distribution()
    if not dist_df.empty:
        st.bar_chart(dist_df.set_index("score_bucket")["count"], use_container_width=True)
        st.caption(f"Threshold at {settings.EVENT_DETECTION_MIN_SCORE} — items left of this are filtered.")
    else:
        st.info("No data yet.")

with col2:
    st.subheader("Thresholds")
    st.metric("Min Score",        settings.EVENT_DETECTION_MIN_SCORE)
    st.metric("KW→NLP Threshold", settings.KEYWORD_SCORE_NLP_THRESHOLD)
    st.metric("Keyword Weight",   f"{settings.KEYWORD_WEIGHT:.0%}")
    st.metric("NLP Weight",       f"{1 - settings.KEYWORD_WEIGHT:.0%}")

st.divider()
st.subheader("Scored Items — click a row for details")

df = get_detection_detail(limit=100)

if df.empty:
    st.info("No scored items yet.")
    time.sleep(5)
    st.rerun()

display_df = df.drop(columns=["id", "body", "url"], errors="ignore").copy()
display_df.insert(0, "status", display_df["filtered_out"].map({False: "✓", True: "✗"}))

selection = st.dataframe(
    display_df,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="detect_table",
    column_config={
        "status":        st.column_config.TextColumn("", width="small"),
        "fetched_at":    st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
        "source":        st.column_config.TextColumn("Source", width="small"),
        "headline":      st.column_config.TextColumn("Headline", width="large"),
        "keyword_score": st.column_config.ProgressColumn("KW Score",    min_value=0, max_value=1),
        "nlp_score":     st.column_config.ProgressColumn("NLP Score",   min_value=0, max_value=1),
        "event_score":   st.column_config.ProgressColumn("Final Score", min_value=0, max_value=1),
        "filtered_out":  None,
        "filter_reason": st.column_config.TextColumn("Reason"),
    },
    hide_index=True,
)

# ── Drill-down ──────────────────────────────────────────────────────────────────
rows = selection.selection.rows
if rows:
    row = df.iloc[rows[0]]
    st.divider()
    st.subheader("Score Breakdown")

    col1, col2, col3 = st.columns(3)
    kw  = row.get("keyword_score") or 0
    nlp = row.get("nlp_score") or 0
    ev  = row.get("event_score") or 0

    col1.metric("Keyword Score", f"{kw:.3f}")
    col1.progress(float(kw))
    col2.metric("NLP Score", f"{nlp:.3f}")
    col2.progress(float(nlp))
    col3.metric("Final Score", f"{ev:.3f}")
    col3.progress(float(ev))

    threshold = settings.EVENT_DETECTION_MIN_SCORE
    if ev >= threshold:
        st.success(f"Passed (score {ev:.3f} ≥ threshold {threshold})")
    else:
        st.error(f"Filtered (score {ev:.3f} < threshold {threshold}): {row.get('filter_reason', '')}")

    st.markdown(f"**{row['headline']}**")
    st.caption(f"{row['source']} · {row['fetched_at']}")
    if row.get("url"):
        st.markdown(f"[Open article]({row['url']})")
    if row.get("body"):
        with st.expander("Body text"):
            st.write(row["body"])

time.sleep(10 if rows else 5)
st.rerun()
