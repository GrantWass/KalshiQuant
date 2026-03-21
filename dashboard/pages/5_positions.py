"""
Page 5 — Positions

Shows current open positions with P&L tracking:
  - Total portfolio exposure (sum of entry costs)
  - Unrealized P&L (based on current WebSocket prices)
  - Realized P&L (from settled positions)
  - Per-position table with avg entry price vs current price
"""

import time
import streamlit as st
from dashboard.db_queries import get_positions, get_portfolio_summary
from config.settings import settings

st.set_page_config(layout="wide")
st.title("Positions")
st.caption("Open positions and profit/loss. Prices updated from Kalshi WebSocket feed.")

# ── Portfolio summary ──────────────────────────────────────────────────────────
summary = get_portfolio_summary()

c1, c2, c3, c4 = st.columns(4)
exposure = summary.get("total_exposure_usd", 0)
unrealized = summary.get("total_unrealized_pnl", 0)
realized = summary.get("total_realized_pnl", 0)
open_pos = summary.get("open_positions", 0)

c1.metric(
    "Total Exposure",
    f"${exposure:.2f}",
    delta=f"Max: ${settings.MAX_TOTAL_EXPOSURE_USD:.0f}",
    delta_color="off",
)
c2.metric(
    "Unrealized P&L",
    f"${unrealized:.2f}",
    delta_color="normal",
)
c3.metric(
    "Realized P&L",
    f"${realized:.2f}",
    delta_color="normal",
)
c4.metric("Open Positions", open_pos)

# Exposure bar
if settings.MAX_TOTAL_EXPOSURE_USD > 0:
    pct = min(exposure / settings.MAX_TOTAL_EXPOSURE_USD, 1.0)
    st.progress(pct, text=f"Exposure: {pct:.0%} of ${settings.MAX_TOTAL_EXPOSURE_USD:.0f} limit")

st.divider()

# ── Positions table ────────────────────────────────────────────────────────────
df = get_positions()
if df.empty:
    st.info("No open positions. Trades will appear here after the first execution.")
else:
    def _pnl_color(val):
        if val is None or val == "":
            return ""
        try:
            v = float(val)
            if v > 0:
                return "color: green"
            if v < 0:
                return "color: red"
        except (TypeError, ValueError):
            pass
        return ""

    styled = df.style.applymap(_pnl_color, subset=["unrealized_pnl", "realized_pnl"])

    st.dataframe(
        styled,
        use_container_width=True,
        column_config={
            "market_ticker":  st.column_config.TextColumn("Ticker"),
            "market_title":   st.column_config.TextColumn("Market", width="large"),
            "side":           st.column_config.TextColumn("Side", width="small"),
            "contracts":      st.column_config.NumberColumn("Qty"),
            "avg_price":      st.column_config.NumberColumn("Avg Entry", format="$%.2f"),
            "current_price":  st.column_config.NumberColumn("Current", format="$%.2f"),
            "unrealized_pnl": st.column_config.NumberColumn("Unrealized P&L", format="$%.2f"),
            "realized_pnl":   st.column_config.NumberColumn("Realized P&L", format="$%.2f"),
            "last_updated":   st.column_config.DatetimeColumn("Updated", format="HH:mm:ss"),
        },
        hide_index=True,
    )

time.sleep(5)
st.rerun()
