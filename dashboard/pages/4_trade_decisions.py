"""
Page 4 — Trade Decisions

THE most important page for understanding system behavior.

Two-column layout:
  LEFT  — Executed trades: ticker, side, contracts, price, edge, confidence
  RIGHT — Rejected trades: ticker + bulleted list of exactly why not traded

The rejection reasons column directly answers the question:
  "The market moved but we didn't trade — why?"

Common rejection reasons:
  - Edge below minimum (0.04) → news wasn't impactful enough
  - Confidence below minimum (0.60) → market match was weak
  - Position limit reached → already holding max contracts
  - Exposure limit → portfolio cap reached
  - Market closing soon → insufficient time to realize edge
"""

import time
import streamlit as st
import pandas as pd
from dashboard.db_queries import get_recent_decisions, get_daily_decision_summary, get_rejection_reason_counts
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Trade Decisions")
st.caption("Every trade decision — executed and rejected — with full reasoning.")

# ── Summary row ────────────────────────────────────────────────────────────────
summary = get_daily_decision_summary()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Executed Today", summary.get("executed", 0))
c2.metric("Rejected Today", summary.get("rejected", 0))
total = summary.get("executed", 0) + summary.get("rejected", 0)
rate = f"{summary.get('executed', 0) / total * 100:.1f}%" if total > 0 else "—"
c3.metric("Execution Rate", rate)
c4.metric("Min Edge Required", settings.PROBABILITY_SHIFT_MIN)
c5.metric("Min Confidence Required", settings.CONFIDENCE_MIN)

st.divider()

# ── Load all decisions ─────────────────────────────────────────────────────────
all_df = get_recent_decisions(limit=300)

executed_df = all_df[all_df["action"] == "EXECUTE"].copy() if not all_df.empty else pd.DataFrame()
rejected_df = all_df[all_df["action"] == "REJECT"].copy() if not all_df.empty else pd.DataFrame()

col_left, col_right = st.columns(2)

# ── LEFT: Executed trades ──────────────────────────────────────────────────────
with col_left:
    st.subheader(f"Executed ({len(executed_df)})")
    if executed_df.empty:
        st.info("No trades executed yet.")
    else:
        display_cols = ["decided_at", "market_ticker", "side", "contracts",
                        "price_cents", "edge", "confidence", "kalshi_order_id"]
        st.dataframe(
            executed_df[display_cols],
            use_container_width=True,
            column_config={
                "decided_at":     st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                "market_ticker":  st.column_config.TextColumn("Ticker"),
                "side":           st.column_config.TextColumn("Side", width="small"),
                "contracts":      st.column_config.NumberColumn("Qty"),
                "price_cents":    st.column_config.NumberColumn("Price (¢)"),
                "edge":           st.column_config.NumberColumn("Edge", format="%.3f"),
                "confidence":     st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "kalshi_order_id": st.column_config.TextColumn("Order ID"),
            },
            hide_index=True,
        )

# ── RIGHT: Rejected trades ─────────────────────────────────────────────────────
with col_right:
    st.subheader(f"Rejected ({len(rejected_df)})")
    if rejected_df.empty:
        st.info("No rejections yet.")
    else:
        # Format rejection_reasons (PostgreSQL array) as a readable string
        def _format_reasons(reasons) -> str:
            if reasons is None:
                return ""
            if isinstance(reasons, list):
                return "\n• ".join([""] + reasons).lstrip()
            return str(reasons)

        rejected_df["reasons"] = rejected_df["rejection_reasons"].apply(_format_reasons)
        display_cols = ["decided_at", "market_ticker", "edge", "confidence", "reasons"]
        st.dataframe(
            rejected_df[display_cols],
            use_container_width=True,
            column_config={
                "decided_at":    st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                "market_ticker": st.column_config.TextColumn("Ticker"),
                "edge":          st.column_config.NumberColumn("Edge", format="%.3f"),
                "confidence":    st.column_config.NumberColumn("Confidence", format="%.3f"),
                "reasons":       st.column_config.TextColumn("Rejection Reasons", width="large"),
            },
            hide_index=True,
        )

# ── Rejection reason breakdown ─────────────────────────────────────────────────
st.divider()
st.subheader("Rejection Reason Breakdown (last 24h)")
reason_df = get_rejection_reason_counts()
if not reason_df.empty:
    st.bar_chart(reason_df.set_index("reason")["count"], use_container_width=True)
else:
    st.caption("No rejection data available.")

time.sleep(5)
st.rerun()
