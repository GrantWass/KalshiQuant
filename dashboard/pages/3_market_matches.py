"""
Page 3 — Market Matches
"""

import time
import streamlit as st
from dashboard.db_queries import get_events_with_matches, get_matches_for_event, get_near_miss_events, get_near_misses_for_event
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Market Matches")
st.caption(
    f"News events that passed detection and reached the FAISS matcher. "
    f"Click a row to see all matched markets ranked by similarity. "
    f"Min similarity: **{settings.SIMILARITY_MIN_SCORE}** | Top-K: **{settings.SIMILARITY_TOP_K}**"
)

events_df = get_events_with_matches(limit=100)

if events_df.empty:
    st.info("No market matches yet. Waiting for events to pass detection threshold.")
    time.sleep(5)
    st.rerun()

# ── Summary stats ───────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Events Matched", len(events_df))
c2.metric("Avg Best Similarity", f"{events_df['best_similarity'].mean():.3f}")
c3.metric("Strong Matches (≥0.85)", int((events_df["best_similarity"] >= 0.85).sum()))

st.divider()

# ── Events table ────────────────────────────────────────────────────────────────
st.subheader("Events That Reached Matching — click a row to see markets")

selection = st.dataframe(
    events_df[["fetched_at", "source", "headline", "event_score", "match_count", "best_similarity", "avg_similarity"]],
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="events_table",
    column_config={
        "fetched_at":      st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
        "source":          st.column_config.TextColumn("Source", width="small"),
        "headline":        st.column_config.TextColumn("Headline", width="large"),
        "event_score":     st.column_config.ProgressColumn("Event Score", min_value=0, max_value=1),
        "match_count":     st.column_config.NumberColumn("Matches", width="small"),
        "best_similarity": st.column_config.ProgressColumn("Best Similarity", min_value=0, max_value=1),
        "avg_similarity":  st.column_config.NumberColumn("Avg Similarity", format="%.3f"),
    },
    hide_index=True,
)

# ── Drill-down: all markets matched for this event ──────────────────────────────
rows = selection.selection.rows
if rows:
    event = events_df.iloc[rows[0]]
    st.divider()

    st.subheader("Matched Markets")
    st.markdown(f"**{event['headline']}**")
    st.caption(f"{event['source']} · {event['fetched_at']}  |  Event score: {event['event_score']:.3f}")
    if event.get("url"):
        st.markdown(f"[Open article]({event['url']})")

    matches_df = get_matches_for_event(event["news_event_id"])

    if matches_df.empty:
        st.info("No matches found.")
    else:
        def _color_similarity(col):
            def _style(v):
                if v >= 0.85:
                    return "color: #2ecc71; font-weight: bold"
                elif v >= 0.70:
                    return "color: #f39c12; font-weight: bold"
                else:
                    return "color: #e74c3c"
            return col.map(_style)

        styled = matches_df[["market_ticker", "market_title", "similarity_score", "decision_action"]].style.apply(
            _color_similarity, subset=["similarity_score"]
        )

        match_sel = st.dataframe(
            styled,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="matches_table",
            column_config={
                "market_ticker":   st.column_config.TextColumn("Ticker", width="small"),
                "market_title":    st.column_config.TextColumn("Market", width="large"),
                "similarity_score":st.column_config.NumberColumn("Similarity", format="%.3f"),
                "decision_action": st.column_config.TextColumn("Decision", width="small"),
            },
            hide_index=True,
        )

        # ── Drill into a specific match ──────────────────────────────────────────
        match_rows = match_sel.selection.rows
        if match_rows:
            match = matches_df.iloc[match_rows[0]]
            st.divider()

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**{match['market_title']}**")
                st.caption(f"Ticker: `{match['market_ticker']}`")
                sim = float(match["similarity_score"])
                st.metric("Cosine Similarity", f"{sim:.3f}")
                st.progress(sim)

            with col_r:
                action = match.get("decision_action")
                if action == "EXECUTE":
                    st.success(
                        f"EXECUTED — {match.get('decision_side', '').upper()} "
                        f"{match.get('decision_contracts', '?')} contracts @ "
                        f"{match.get('decision_price_cents', '?')}¢  |  "
                        f"Edge: {match.get('decision_edge', '?')}  |  "
                        f"Confidence: {match.get('decision_confidence', '?')}"
                    )
                elif action == "REJECT":
                    reasons = match.get("rejection_reasons") or []
                    reason_str = " · ".join(reasons) if isinstance(reasons, list) else str(reasons)
                    st.error(f"REJECTED — {reason_str}")
                else:
                    st.info("No trade decision recorded for this match.")

# ── Near-miss events ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Near-Misses — passed detection but below similarity threshold")
st.caption("These events were detected but all FAISS scores fell below the minimum. Click to see closest markets.")

near_miss_df = get_near_miss_events(limit=100)

if not near_miss_df.empty:
    nm_selection = st.dataframe(
        near_miss_df[["fetched_at", "source", "headline", "event_score", "best_similarity"]],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="near_miss_table",
        column_config={
            "fetched_at":      st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "source":          st.column_config.TextColumn("Source", width="small"),
            "headline":        st.column_config.TextColumn("Headline", width="large"),
            "event_score":     st.column_config.ProgressColumn("Event Score", min_value=0, max_value=1),
            "best_similarity": st.column_config.ProgressColumn("Best Similarity", min_value=0, max_value=1),
        },
        hide_index=True,
    )

    nm_rows = nm_selection.selection.rows
    if nm_rows:
        nm_event = near_miss_df.iloc[nm_rows[0]]
        st.markdown(f"**{nm_event['headline']}**")
        st.caption(f"{nm_event['source']} · {nm_event['fetched_at']}  |  Event score: {nm_event['event_score']:.3f}")
        if nm_event.get("url"):
            st.markdown(f"[Open article]({nm_event['url']})")

        nm_matches = get_near_misses_for_event(nm_event["news_event_id"])
        if not nm_matches.empty:
            st.dataframe(
                nm_matches,
                use_container_width=True,
                column_config={
                    "market_ticker":   st.column_config.TextColumn("Ticker", width="small"),
                    "market_title":    st.column_config.TextColumn("Market", width="large"),
                    "similarity_score": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1),
                },
                hide_index=True,
            )
else:
    st.info("No near-miss events yet.")

time.sleep(10 if rows else 5)
st.rerun()
