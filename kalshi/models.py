"""
kalshi/models.py — Pydantic models for Kalshi API types.

These mirror the shapes returned by the Kalshi REST API v2.
Only the fields used by KalshiQuant are included.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class Market(BaseModel):
    """A single Kalshi prediction market."""

    ticker: str
    event_ticker: str | None = None
    title: str = ""
    subtitle: str | None = None
    status: str = ""

    # Prices returned as dollar strings e.g. "0.5600" = 56 cents
    yes_bid_dollars: str | None = None
    yes_ask_dollars: str | None = None
    last_price_dollars: str | None = None

    close_time: datetime | None = None
    expiration_time: datetime | None = None
    latest_expiration_time: datetime | None = None

    # Non-empty for multi-leg parlay markets — used to filter them out
    mve_collection_ticker: str | None = None
    mve_selected_legs: list[dict] = Field(default_factory=list)

    @property
    def is_parlay(self) -> bool:
        return bool(self.mve_collection_ticker or self.mve_selected_legs)

    def _dollars_to_cents(self, value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return round(float(value) * 100)
        except (ValueError, TypeError):
            return None

    @property
    def yes_bid(self) -> int | None:
        return self._dollars_to_cents(self.yes_bid_dollars)

    @property
    def yes_ask(self) -> int | None:
        return self._dollars_to_cents(self.yes_ask_dollars)

    @property
    def last_price(self) -> int | None:
        return self._dollars_to_cents(self.last_price_dollars)

    @property
    def mid_price_cents(self) -> int | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) // 2
        return self.last_price

    @property
    def mid_price(self) -> float | None:
        mid = self.mid_price_cents
        return mid / 100.0 if mid is not None else None


class OrderRequest(BaseModel):
    """Parameters for placing a new limit order."""

    ticker: str
    side: str                # "yes" | "no"
    action: str              # "buy" | "sell"
    count: int               # number of contracts
    type: str = "limit"      # "limit" | "market"
    yes_price: int           # limit price for YES in cents (1-99)


class OrderResponse(BaseModel):
    """Response from a successful order placement."""

    order_id: str
    ticker: str
    status: str              # "resting" | "filled" | "canceled"
    side: str
    action: str
    count: int
    yes_price: int
    created_time: datetime | None = None


class Orderbook(BaseModel):
    """Current order book depth for a market."""

    ticker: str
    yes_bids: list[list[int]] = Field(default_factory=list)  # [[price_cents, size], ...]
    yes_asks: list[list[int]] = Field(default_factory=list)

    @property
    def best_yes_bid(self) -> int | None:
        return max((b[0] for b in self.yes_bids), default=None)

    @property
    def best_yes_ask(self) -> int | None:
        return min((a[0] for a in self.yes_asks), default=None)


class Portfolio(BaseModel):
    """User portfolio summary."""

    balance: int             # available balance in cents
    total_value: int         # total portfolio value in cents (balance + positions)


class Position(BaseModel):
    """A single open position in the user's portfolio."""

    ticker: str
    title: str | None = None
    market_exposure: int     # total value of open contracts in cents
    resting_orders_count: int = 0
    position: int            # net contract count (positive = YES, negative = NO)
    realized_pnl: int = 0    # realized P&L in cents
    unrealized_pnl: int = 0  # unrealized P&L in cents
