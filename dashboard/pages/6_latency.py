"""
Page 6 — Latency

Shows per-stage pipeline timing for each news item that passed deduplication.

Visualizations:
  1. Metric row: P50 / P95 / P99 total latency (last hour)
  2. Gantt-style table: each item as a row, each stage as a column (ms elapsed)
  3. Stage breakdown bar chart: average time per stage
  4. Rolling line chart: total_ms over time
  5. P95 alert: red banner if P95 > LATENCY_TARGET_MS (5000ms)

Stage columns:
  ms_dedup    — time from fetch to dedup check passing
  ms_detect   — time in event detector (keyword + NLP)
  ms_match    — time in FAISS search + DB writes
  ms_estimate — time in probability estimator + price lookup
  ms_decide   — time in decision engine (all 5 gates)
  ms_execute  — time in Kalshi API order placement (NULL if rejected)
  ms_total    — total from fetch to final stage
"""

import time
import streamlit as st
import pandas as pd
from dashboard.db_queries import (
    get_recent_latency,
    get_latency_percentiles,
    get_stage_averages,
)
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Pipeline Latency")
st.caption(
    f"Per-stage timing for items passing deduplication. "
    f"Target: **{settings.LATENCY_TARGET_MS}ms** end-to-end."
)

# ── Percentiles ────────────────────────────────────────────────────────────────
pcts = get_latency_percentiles()
p50 = pcts.get("p50")
p95 = pcts.get("p95")
p99 = pcts.get("p99")

c1, c2, c3, c4 = st.columns(4)
c1.metric("P50 Latency", f"{p50:.0f}ms" if p50 else "—")
c2.metric("P95 Latency", f"{p95:.0f}ms" if p95 else "—")
c3.metric("P99 Latency", f"{p99:.0f}ms" if p99 else "—")
c4.metric("Target", f"{settings.LATENCY_TARGET_MS}ms")

# Alert if P95 exceeds target
if p95 is not None and p95 > settings.LATENCY_TARGET_MS:
    st.error(
        f"P95 latency ({p95:.0f}ms) exceeds target ({settings.LATENCY_TARGET_MS}ms). "
        f"Check which stage is the bottleneck in the chart below."
    )

st.divider()

df = get_recent_latency(limit=200)

if df.empty:
    st.info("No latency data yet. Items appear here after passing deduplication.")
else:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Average Time per Stage (last hour)")
        stage_df = get_stage_averages()
        if not stage_df.empty and stage_df["avg_ms"].notna().any():
            st.bar_chart(
                stage_df.dropna(subset=["avg_ms"]).set_index("stage")["avg_ms"],
                use_container_width=True,
            )
        else:
            st.caption("Not enough data yet.")

    with col_right:
        st.subheader("Total Latency Over Time")
        ts_df = df[["t_fetched", "ms_total"]].dropna()
        if not ts_df.empty:
            ts_df = ts_df.sort_values("t_fetched")
            st.line_chart(ts_df.set_index("t_fetched")["ms_total"], use_container_width=True)

    st.divider()

    # ── Gantt-style table ──────────────────────────────────────────────────────
    st.subheader("Per-Item Stage Breakdown (most recent 100)")

    gantt_cols = ["t_fetched", "source", "ms_dedup", "ms_detect", "ms_match",
                  "ms_estimate", "ms_decide", "ms_execute", "ms_total"]
    gantt_df = df[gantt_cols].head(100)

    # Highlight slow rows (total > target)
    def _row_style(row):
        if row.get("ms_total") is not None and row["ms_total"] > settings.LATENCY_TARGET_MS:
            return [f"background-color: #ffcccc"] * len(row)
        return [""] * len(row)

    styled = gantt_df.style.apply(_row_style, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "t_fetched":   st.column_config.DatetimeColumn("Fetched", format="HH:mm:ss"),
            "source":      st.column_config.TextColumn("Source", width="small"),
            "ms_dedup":    st.column_config.NumberColumn("Dedup (ms)", format="%.1f"),
            "ms_detect":   st.column_config.NumberColumn("Detect (ms)", format="%.1f"),
            "ms_match":    st.column_config.NumberColumn("Match (ms)", format="%.1f"),
            "ms_estimate": st.column_config.NumberColumn("Estimate (ms)", format="%.1f"),
            "ms_decide":   st.column_config.NumberColumn("Decide (ms)", format="%.1f"),
            "ms_execute":  st.column_config.NumberColumn("Execute (ms)", format="%.1f"),
            "ms_total":    st.column_config.NumberColumn("TOTAL (ms)", format="%.0f"),
        },
        hide_index=True,
    )

time.sleep(5)
st.rerun()
