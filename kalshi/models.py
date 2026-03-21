"""
kalshi/models.py — Pydantic models for Kalshi API types.

These mirror the shapes returned by the Kalshi REST API v2.
Only the fields used by KalshiQuant are included.
"""

from datetime import datetime
from pydantic import BaseModel, Field


class Market(BaseModel):
    """A single Kalshi prediction market."""

    ticker: str                          # unique market identifier, e.g. "KXBTC-25JAN-T50001"
    title: str                           # human-readable title
    subtitle: str | None = None          # additional context
    category: str | None = None          # broad category (e.g. "Economics", "Weather")
    tags: list[str] = Field(default_factory=list)

    status: str                          # "open" | "closed" | "settled"
    yes_bid: int | None = None           # best YES bid in cents
    yes_ask: int | None = None           # best YES ask in cents
    last_price: int | None = None        # last traded YES price in cents
    close_time: datetime | None = None   # when the market closes to new orders
    expiration_time: datetime | None = None

    @property
    def mid_price_cents(self) -> int | None:
        """Midpoint of best YES bid/ask in cents."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) // 2
        return self.last_price

    @property
    def mid_price(self) -> float | None:
        """Midpoint as a probability [0..1]."""
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
