"""
Page 4 — Trade Decisions
"""

import time
import streamlit as st
import pandas as pd
from dashboard.db_queries import get_recent_decisions, get_daily_decision_summary, get_rejection_reason_counts
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Trade Decisions")
st.caption("Every trade decision — executed and rejected — with full reasoning. Click a row for details.")

# ── Summary ─────────────────────────────────────────────────────────────────────
summary = get_daily_decision_summary()
c1, c2, c3, c4, c5 = st.columns(5)
total = summary.get("executed", 0) + summary.get("rejected", 0)
rate  = f"{summary.get('executed', 0) / total * 100:.1f}%" if total > 0 else "—"
c1.metric("Executed Today",       summary.get("executed", 0))
c2.metric("Rejected Today",       summary.get("rejected", 0))
c3.metric("Execution Rate",       rate)
c4.metric("Min Edge Required",    settings.PROBABILITY_SHIFT_MIN)
c5.metric("Min Confidence",       settings.CONFIDENCE_MIN)

st.divider()

all_df = get_recent_decisions(limit=300)

tab_exec, tab_reject = st.tabs([
    f"Executed ({len(all_df[all_df['action'] == 'EXECUTE']) if not all_df.empty else 0})",
    f"Rejected ({len(all_df[all_df['action'] == 'REJECT']) if not all_df.empty else 0})",
])

# ── Executed tab ────────────────────────────────────────────────────────────────
with tab_exec:
    executed_df = all_df[all_df["action"] == "EXECUTE"].copy() if not all_df.empty else pd.DataFrame()

    if executed_df.empty:
        st.info("No trades executed yet.")
    else:
        disp_cols = ["decided_at", "market_ticker", "market_title", "side",
                     "contracts", "price_cents", "edge", "confidence", "kalshi_order_id"]
        sel_exec = st.dataframe(
            executed_df[disp_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="exec_table",
            column_config={
                "decided_at":      st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                "market_ticker":   st.column_config.TextColumn("Ticker"),
                "market_title":    st.column_config.TextColumn("Market", width="large"),
                "side":            st.column_config.TextColumn("Side", width="small"),
                "contracts":       st.column_config.NumberColumn("Qty"),
                "price_cents":     st.column_config.NumberColumn("Price (¢)"),
                "edge":            st.column_config.NumberColumn("Edge", format="%.3f"),
                "confidence":      st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "kalshi_order_id": st.column_config.TextColumn("Order ID"),
            },
            hide_index=True,
        )

        rows = sel_exec.selection.rows
        if rows:
            row = executed_df.iloc[rows[0]]
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Side",        row.get("side", "—").upper())
            c2.metric("Contracts",   row.get("contracts", "—"))
            c3.metric("Price",       f"{row.get('price_cents', '?')}¢")
            c4.metric("Order ID",    row.get("kalshi_order_id", "—") or "dry-run")
            c1.metric("Edge",        f"{row.get('edge', 0):.3f}")
            c2.metric("Confidence",  f"{row.get('confidence', 0):.3f}")
            c3.metric("P Market",    f"{row.get('p_market', 0):.3f}")
            c4.metric("P Estimated", f"{row.get('p_estimated', 0):.3f}")
            st.markdown(f"**Market:** {row.get('market_title', '')}  `{row.get('market_ticker', '')}`")

# ── Rejected tab ────────────────────────────────────────────────────────────────
with tab_reject:
    rejected_df = all_df[all_df["action"] == "REJECT"].copy() if not all_df.empty else pd.DataFrame()

    if rejected_df.empty:
        st.info("No rejections yet.")
    else:
        def _fmt_reasons(reasons) -> str:
            if reasons is None:
                return ""
            if isinstance(reasons, list):
                return " · ".join(reasons)
            return str(reasons)

        rejected_df["reasons"] = rejected_df["rejection_reasons"].apply(_fmt_reasons)
        disp_cols = ["decided_at", "market_ticker", "market_title", "edge", "confidence", "reasons"]

        sel_rej = st.dataframe(
            rejected_df[disp_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="rej_table",
            column_config={
                "decided_at":    st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                "market_ticker": st.column_config.TextColumn("Ticker"),
                "market_title":  st.column_config.TextColumn("Market", width="large"),
                "edge":          st.column_config.NumberColumn("Edge", format="%.3f"),
                "confidence":    st.column_config.NumberColumn("Confidence", format="%.3f"),
                "reasons":       st.column_config.TextColumn("Rejection Reasons", width="large"),
            },
            hide_index=True,
        )

        rows = sel_rej.selection.rows
        if rows:
            row = rejected_df.iloc[rows[0]]
            st.divider()
            st.subheader("Why was this rejected?")
            reasons = row.get("rejection_reasons") or []
            if isinstance(reasons, list):
                for r in reasons:
                    st.error(f"• {r}")
            else:
                st.error(str(reasons))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Edge",        f"{row.get('edge', 0):.3f}",       delta=f"min {settings.PROBABILITY_SHIFT_MIN}", delta_color="off")
            c2.metric("Confidence",  f"{row.get('confidence', 0):.3f}", delta=f"min {settings.CONFIDENCE_MIN}",        delta_color="off")
            c3.metric("P Market",    f"{row.get('p_market', 0):.3f}")
            c4.metric("P Estimated", f"{row.get('p_estimated', 0):.3f}")
            st.markdown(f"**Market:** {row.get('market_title', '')}  `{row.get('market_ticker', '')}`")

# ── Rejection reason breakdown ──────────────────────────────────────────────────
st.divider()
st.subheader("Rejection Reason Breakdown (last 24h)")
reason_df = get_rejection_reason_counts()
if not reason_df.empty:
    st.bar_chart(reason_df.set_index("reason")["count"], use_container_width=True)
else:
    st.caption("No rejection data yet.")

time.sleep(5)
st.rerun()
