"""
Page 1 — News Feed
"""

import time
import streamlit as st
from dashboard.db_queries import get_recent_news, get_news_source_counts

st.set_page_config(layout="wide")
st.title("News Feed")
st.caption("All news items ingested from all sources. Click a row to see details.")

# ── Source summary metrics ──────────────────────────────────────────────────────
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

# ── Filters ─────────────────────────────────────────────────────────────────────
fc1, fc2, fc3 = st.columns([2, 2, 1])

with fc1:
    all_sources = sorted(counts["source"].tolist()) if not counts.empty else []
    selected_sources = st.multiselect("Sources", all_sources, default=all_sources, placeholder="All sources")

with fc2:
    status_filter = st.radio("Status", ["All", "Passed", "Filtered"], horizontal=True)

with fc3:
    limit = st.selectbox("Show", [100, 200, 500], index=0)

st.divider()

# ── Load and filter data ────────────────────────────────────────────────────────
df = get_recent_news(limit=limit)

if not df.empty:
    if selected_sources:
        df = df[df["source"].isin(selected_sources)]
    if status_filter == "Passed":
        df = df[~df["filtered_out"]]
    elif status_filter == "Filtered":
        df = df[df["filtered_out"]]

# ── Table ───────────────────────────────────────────────────────────────────────
if df.empty:
    st.info("No news items match your filters.")
else:
    display_df = df.drop(columns=["id", "body"], errors="ignore").copy()
    display_df.insert(0, "status", display_df["filtered_out"].map({False: "✓", True: "✗"}))

    def _color_status(col):
        return col.map({"✓": "color: #2ecc71; font-weight: bold", "✗": "color: #e74c3c; font-weight: bold"})

    styled_df = display_df.style.apply(_color_status, subset=["status"])

    selection = st.dataframe(
        styled_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="news_table",
        column_config={
            "status":        st.column_config.TextColumn("", width="small"),
            "fetched_at":    st.column_config.DatetimeColumn("Fetched At", format="HH:mm:ss"),
            "source":        st.column_config.TextColumn("Source", width="small"),
            "headline":      st.column_config.TextColumn("Headline", width="large"),
            "event_score":   st.column_config.NumberColumn("Score", format="%.3f"),
            "keyword_score": st.column_config.NumberColumn("KW", format="%.3f"),
            "nlp_score":     st.column_config.NumberColumn("NLP", format="%.3f"),
            "filtered_out":  None,
            "filter_reason": st.column_config.TextColumn("Reason"),
            "url":           st.column_config.LinkColumn("URL"),
        },
        hide_index=True,
    )

    # ── Drill-down panel ────────────────────────────────────────────────────────
    rows = selection.selection.rows
    if rows:
        row = df.iloc[rows[0]]
        st.divider()
        st.subheader("Item Detail")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{row['headline']}**")
            if row.get("url"):
                st.markdown(f"[Open article]({row['url']})")
            st.caption(f"Source: {row['source']} · Fetched: {row['fetched_at']}")
            if row.get("body"):
                with st.expander("Full body text"):
                    st.write(row["body"])
            if row.get("filter_reason"):
                st.warning(f"Filtered: {row['filter_reason']}")

        with col2:
            st.metric("Event Score",   f"{row['event_score']:.3f}"   if row.get('event_score')   is not None else "—")
            st.metric("Keyword Score", f"{row['keyword_score']:.3f}" if row.get('keyword_score') is not None else "—")
            st.metric("NLP Score",     f"{row['nlp_score']:.3f}"     if row.get('nlp_score')     is not None else "—")

try:
    refresh = 10 if selection.selection.rows else 5
except Exception:
    refresh = 5
time.sleep(refresh)
st.rerun()
