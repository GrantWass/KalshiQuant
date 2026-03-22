"""
Page 3 — Market Matches
"""

import time
import streamlit as st
from dashboard.db_queries import get_recent_matches
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Market Matches")
st.caption(
    f"FAISS cosine similarity results. "
    f"Min similarity: **{settings.SIMILARITY_MIN_SCORE}** | Top-K: **{settings.SIMILARITY_TOP_K}**. "
    f"Click a row to see the full news → market → decision chain."
)

df = get_recent_matches(limit=150)

if df.empty:
    st.info("No market matches yet. Waiting for events to pass detection threshold.")
    time.sleep(5)
    st.rerun()

# ── Summary stats ───────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Matches (shown)", len(df))
c2.metric("Avg Similarity", f"{df['similarity_score'].mean():.3f}" if not df.empty else "—")
c3.metric("Strong Matches (≥0.85)", int((df["similarity_score"] >= 0.85).sum()))

st.divider()

# ── Table ───────────────────────────────────────────────────────────────────────
display_cols = ["matched_at", "source", "headline", "market_ticker", "market_title", "similarity_score", "decision_action"]
display_df = df[display_cols].copy()

selection = st.dataframe(
    display_df,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="matches_table",
    column_config={
        "matched_at":      st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
        "source":          st.column_config.TextColumn("Source", width="small"),
        "headline":        st.column_config.TextColumn("Headline", width="large"),
        "market_ticker":   st.column_config.TextColumn("Ticker", width="medium"),
        "market_title":    st.column_config.TextColumn("Market", width="large"),
        "similarity_score":st.column_config.NumberColumn("Similarity", format="%.3f"),
        "decision_action": st.column_config.TextColumn("Decision", width="small"),
    },
    hide_index=True,
)

# ── Drill-down ──────────────────────────────────────────────────────────────────
rows = selection.selection.rows
if rows:
    row = df.iloc[rows[0]]
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("News Item")
        st.markdown(f"**{row['headline']}**")
        st.caption(f"{row['source']} · {row['matched_at']}")
        if row.get("url"):
            st.markdown(f"[Open article]({row['url']})")
        if row.get("body"):
            with st.expander("Body text"):
                st.write(row["body"])

    with col_right:
        st.subheader("Matched Market")
        st.markdown(f"**{row['market_title']}**")
        st.caption(f"Ticker: `{row['market_ticker']}`")
        sim = row.get("similarity_score", 0)
        st.metric("Cosine Similarity", f"{sim:.3f}")
        st.progress(float(sim))

    st.subheader("Trade Decision")
    action = row.get("decision_action")
    if action == "EXECUTE":
        st.success(
            f"EXECUTED — {row.get('decision_side', '').upper()} "
            f"{row.get('decision_contracts', '?')} contracts @ "
            f"{row.get('decision_price_cents', '?')}¢  |  "
            f"Edge: {row.get('decision_edge', '?')}  |  "
            f"Confidence: {row.get('decision_confidence', '?')}"
        )
    elif action == "REJECT":
        reasons = row.get("rejection_reasons") or []
        if isinstance(reasons, list):
            reason_str = " · ".join(reasons)
        else:
            reason_str = str(reasons)
        st.error(f"REJECTED — {reason_str}")
    else:
        st.info("No trade decision recorded for this match.")

time.sleep(10 if rows else 5)
st.rerun()
