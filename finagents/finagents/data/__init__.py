"""Online data access: SEC filings and XBRL facts, news, and prices.

There is no offline corpus. `build_provider()` returns the single online
provider, which assembles each ticker's documents from live primary sources.
"""

from __future__ import annotations

from .news import NewsClient
from .providers import (
    Bar,
    DEFAULT_WATCHLIST,
    MarketDataProvider,
    OnlineProvider,
    build_provider,
    build_snapshot_document,
    compute_indicators,
    max_drawdown,
    realised_volatility,
    rsi,
    sma,
    support_resistance,
    total_return,
)
from .sec import SECClient, html_to_text, split_into_sections

__all__ = [
    "Bar",
    "DEFAULT_WATCHLIST",
    "MarketDataProvider",
    "NewsClient",
    "OnlineProvider",
    "SECClient",
    "build_provider",
    "build_snapshot_document",
    "compute_indicators",
    "html_to_text",
    "max_drawdown",
    "realised_volatility",
    "rsi",
    "sma",
    "split_into_sections",
    "support_resistance",
    "total_return",
]
