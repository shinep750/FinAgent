"""Online data provider.

`MarketDataProvider` is the seam between the agents and the outside world, and
this implementation is the only one: documents come from SEC EDGAR and Google
News, prices come from Yahoo Finance. There is no bundled corpus and no offline
path, so a research question is always answered from live primary sources.

What a run assembles per ticker:

* **XBRL financial facts** (SEC) - authoritative tagged figures, the highest
  reliability document in the corpus and the one the verifier checks numbers
  against most easily.
* **Filing sections** (SEC) - 10-K/10-Q/8-K split at their own headings, so a
  risk question retrieves "Item 1A. Risk Factors" rather than a 300k-character
  filing.
* **News** (Google News RSS) - headlines and lead paragraphs, attributed to the
  original outlet.
* **A market snapshot** derived from the real bar series, so the prose about
  price and volatility is computed from the same bars the indicators use and
  the two cannot disagree.

Every fetch degrades independently: a news outage must not cost the run its
filings, and a price outage must not cost it the filings either.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..schemas import Document, EvidenceType, Settings
from .news import NewsClient
from .sec import SECClient

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """One session of OHLCV."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(ABC):
    """Supplies documents and OHLCV series to the retrieval + agent layers."""

    name: str = "base"
    is_live: bool = True

    @abstractmethod
    def load_documents(self, tickers: Optional[Sequence[str]] = None) -> List[Document]:
        """Return the retrievable corpus for `tickers`."""

    @abstractmethod
    def price_series(self, ticker: str) -> List[Bar]:
        """Return chronological OHLCV bars for `ticker`."""

    def supported_tickers(self) -> List[str]:
        """A small default watchlist, used only to hint the planner.

        Online the universe is every US filer, so this is not a whitelist: see
        `resolve_ticker`, which is what actually validates a symbol.
        """
        return list(DEFAULT_WATCHLIST)

    def resolve_ticker(self, symbol: str) -> Optional[str]:
        """Return the canonical ticker if it names a real filer, else None."""
        return None

    def canonical_tickers(self, tickers: Optional[Sequence[str]]) -> List[str]:
        """Resolve caller-supplied symbols, preserving order and dropping repeats.

        Every public entry point goes through this. A caller who writes `BRK.B`
        because that is how the stock appears in a browser must get Berkshire's
        filings, not an empty result: `SECClient.cik("BRK.B")` is None (EDGAR
        spells it `BRK-B`) and Yahoo does not know the dotted form either. So the
        symbol would silently degrade to *news only* - the worst failure mode,
        because the query looks like it worked.

        A symbol with no canonical match is kept as given rather than dropped: a
        security that only Yahoo knows should still get priced, and the SEC
        lookups will come back empty instead of raising.
        """
        if tickers is None:
            return self.supported_tickers()
        resolved: List[str] = []
        for symbol in tickers:
            canonical = self.resolve_ticker(symbol) or symbol.upper().strip()
            if canonical and canonical not in resolved:
                resolved.append(canonical)
        return resolved

    def company_name(self, ticker: str) -> Optional[str]:
        return None

    def provenance(self) -> Dict[str, str]:
        return {
            "provider": self.name,
            "filings": "SEC EDGAR",
            "financial_facts": "SEC EDGAR XBRL",
            "news": "Google News RSS",
            "prices": "Yahoo Finance",
        }


# A liquid large-cap watchlist. Deliberately small: it only seeds the planner's
# prompt, and every symbol is still resolved against EDGAR before use.
DEFAULT_WATCHLIST: Sequence[str] = (
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "AVGO",
    "NFLX",
)


# Separators people use for a share class, in the order they are tried. EDGAR
# writes `BRK-B`; a human writes `BRK.B`, `BRK/B`, or "BRK B".
_CLASS_SEPARATORS = (".", "/", "_", " ", "-")


def _ticker_spellings(raw: str) -> List[str]:
    """Every plausible EDGAR spelling of `raw`, most literal first."""
    spellings = [raw]
    canonical = raw
    for separator in _CLASS_SEPARATORS:
        if separator in canonical:
            canonical = canonical.replace(separator, "-")
            spellings.append(canonical)
    if "-" in canonical:
        # `BRK B` -> `BRK-B`; also handles a class letter given as `BRKA`.
        spellings.append(canonical.replace("-", ""))
    seen: List[str] = []
    for spelling in spellings:
        if spelling and spelling not in seen:
            seen.append(spelling)
    return seen


class OnlineProvider(MarketDataProvider):
    """Fetches live documents and prices, and caches both for the process."""

    name = "online"

    def __init__(
        self,
        sec: Optional[SECClient] = None,
        news: Optional[NewsClient] = None,
        filings_per_ticker: int = 3,
        max_chunks_per_filing: int = 20,
        max_chunks_per_section: int = 3,
        news_per_ticker: int = 10,
    ) -> None:
        self.sec = sec or SECClient()
        self.news = news or NewsClient()
        self.filings_per_ticker = filings_per_ticker
        self.max_chunks_per_filing = max_chunks_per_filing
        self.max_chunks_per_section = max_chunks_per_section
        self.news_per_ticker = news_per_ticker
        self._documents: Dict[str, List[Document]] = {}
        self._bars: Dict[str, List[Bar]] = {}

    def close(self) -> None:
        self.sec.close()
        self.news.close()

    # -- ticker resolution ------------------------------------------------

    def resolve_ticker(self, symbol: str) -> Optional[str]:
        """Return the canonical ticker if it names a real filer, else None.

        Deliberately not a `[A-Z]{1,5}` regex test. EDGAR's map holds 10,431
        symbols and 543 of them are not plain letters: `BRK-A` and `BRK-B` class
        shares, `BAC-PB` preferreds, `OAK-PA` units. A regex gate silently drops
        every one of them, and Berkshire class B is not an exotic listing - it is
        the most widely held individual stock in the country.

        Separators are normalised instead, because the SEC writes `BRK-B` while
        people type `BRK.B`, `BRK/B` and `BRK B`. The map decides what is real;
        this method only decides what a user probably meant.
        """
        raw = symbol.upper().strip().lstrip("$")
        if not raw:
            return None
        for candidate in _ticker_spellings(raw):
            if self.sec.is_known_ticker(candidate):
                return candidate
        return None

    def company_name(self, ticker: str) -> Optional[str]:
        return self.sec.company_name(ticker)

    def supported_tickers(self) -> List[str]:
        """Watchlist plus anything already fetched, for the planner's prompt."""
        return sorted(set(DEFAULT_WATCHLIST) | set(self._documents))

    # -- documents --------------------------------------------------------

    def load_documents(self, tickers: Optional[Sequence[str]] = None) -> List[Document]:
        wanted = self.canonical_tickers(tickers)
        for ticker in wanted:
            if ticker not in self._documents:
                self._documents[ticker] = self._fetch_documents(ticker)
        docs: List[Document] = []
        for ticker in wanted:
            docs.extend(self._documents.get(ticker, []))
        if not docs:
            logger.warning("no documents retrieved for %s", ", ".join(wanted))
        return docs

    def _fetch_documents(self, ticker: str) -> List[Document]:
        """Assemble one ticker's corpus, tolerating each source failing."""
        docs: List[Document] = []
        for label, fetch in (
            ("financial facts", self._fetch_facts),
            ("filings", self._fetch_filings),
            ("news", self._fetch_news),
            ("market snapshot", self._fetch_snapshot),
        ):
            try:
                docs.extend(fetch(ticker))
            except (
                Exception
            ) as exc:  # noqa: BLE001 - one dead source must not sink the ticker
                logger.warning(
                    "%s: %s fetch failed for %s: %s", self.name, label, ticker, exc
                )
        logger.info("%s: assembled %d document(s)", ticker, len(docs))
        return docs

    def _fetch_facts(self, ticker: str) -> List[Document]:
        document = self.sec.financials_document(ticker)
        return [document] if document else []

    def _fetch_filings(self, ticker: str) -> List[Document]:
        return self.sec.filing_documents(
            ticker,
            limit=self.filings_per_ticker,
            max_chunks_per_filing=self.max_chunks_per_filing,
            max_chunks_per_section=self.max_chunks_per_section,
        )

    def _fetch_news(self, ticker: str) -> List[Document]:
        return self.news.fetch(ticker, limit=self.news_per_ticker)

    def _fetch_snapshot(self, ticker: str) -> List[Document]:
        document = build_snapshot_document(ticker, self)
        return [document] if document else []

    # -- prices -----------------------------------------------------------

    def price_series(self, ticker: str) -> List[Bar]:
        symbol = self.resolve_ticker(ticker) or ticker.upper().strip()
        if symbol not in self._bars:
            self._bars[symbol] = self._fetch_bars(symbol)
        return self._bars[symbol]

    def _fetch_bars(self, symbol: str) -> List[Bar]:
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RuntimeError(
                "yfinance is required for online price data: pip install -e ."
            ) from exc
        try:
            frame = yf.Ticker(symbol).history(period="1y", auto_adjust=False)
        except Exception as exc:  # noqa: BLE001 - network/parser issues vary
            logger.warning("price fetch failed for %s: %s", symbol, exc)
            return []
        if frame is None or frame.empty:
            logger.warning("no price rows returned for %s", symbol)
            return []
        bars: List[Bar] = []
        for idx, row in frame.iterrows():
            try:
                bars.append(
                    Bar(
                        date=idx.date().isoformat(),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return bars


def build_provider(settings: Optional[Settings] = None) -> MarketDataProvider:
    """The online provider. There is no other mode.

    `Settings` is the single source of truth for how the network is used. Without
    it the provider would silently fall back to library defaults, and the one
    setting the SEC actually cares about - the declared contact in the User-Agent
    - would be the one nobody sees.
    """
    s = settings or Settings.from_env()
    return OnlineProvider(
        sec=SECClient(
            user_agent=s.sec_user_agent,
            cache_dir=Path(s.cache_dir) / "sec",
            timeout_s=s.http_timeout_s,
            max_filings=s.filings_per_ticker,
        ),
        news=NewsClient(
            timeout_s=s.http_timeout_s,
            limit=s.news_per_ticker,
            min_body_chars=s.news_min_body_chars,
        ),
        filings_per_ticker=s.filings_per_ticker,
        max_chunks_per_filing=s.max_chunks_per_filing,
        max_chunks_per_section=s.max_chunks_per_section,
        news_per_ticker=s.news_per_ticker,
    )


# --------------------------------------------------------------------------
# Indicator computation (shared by technical + risk agents)
# --------------------------------------------------------------------------


def sma(values: Sequence[float], window: int) -> Optional[float]:
    if len(values) < window or window <= 0:
        return None
    return float(np.mean(values[-window:]))


def rsi(closes: Sequence[float], window: int = 14) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    deltas = np.diff(np.asarray(closes[-window - 1 :], dtype=float))
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = (gains / window) / (losses / window)
    return float(100.0 - (100.0 / (1.0 + rs)))


def realised_volatility(closes: Sequence[float], window: int = 60) -> Optional[float]:
    """Annualised standard deviation of *percentage* returns.

    Percentage returns, not price differences: the difference series of a $600
    stock is mechanically more volatile than that of a $30 stock, which would
    make cross-company volatility comparisons meaningless.
    """
    if len(closes) < window + 1:
        return None
    series = np.asarray(closes[-(window + 1) :], dtype=float)
    returns = np.diff(series) / series[:-1]
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return None
    return float(returns.std(ddof=1) * np.sqrt(252.0))


def max_drawdown(closes: Sequence[float]) -> Optional[float]:
    if len(closes) < 2:
        return None
    series = np.asarray(closes, dtype=float)
    peak = np.maximum.accumulate(series)
    drawdown = (peak - series) / peak
    return float(drawdown.max())


def total_return(closes: Sequence[float]) -> Optional[float]:
    if len(closes) < 2 or not closes[0]:
        return None
    return float((closes[-1] - closes[0]) / closes[0])


def support_resistance(
    closes: Sequence[float], lookback: int = 60
) -> Dict[str, Optional[float]]:
    if len(closes) < lookback:
        return {"support": None, "resistance": None}
    window = closes[-lookback:]
    return {"support": float(min(window)), "resistance": float(max(window))}


def compute_indicators(ticker: str, provider: MarketDataProvider) -> Dict[str, float]:
    """Full indicator set for a ticker, from real bars."""
    bars = provider.price_series(ticker)
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    out: Dict[str, float] = {}
    if not bars:
        return out
    out["last_price"] = float(closes[-1])
    for window in (20, 50, 100, 200):
        value = sma(closes, window)
        if value is not None:
            out[f"sma_{window}"] = round(value, 4)
    value = rsi(closes)
    if value is not None:
        out["rsi_14"] = round(value, 4)
    value = realised_volatility(closes)
    if value is not None:
        out["volatility_60d_annualised"] = round(value, 4)
    value = max_drawdown(closes)
    if value is not None:
        out["max_drawdown"] = round(value, 4)
    value = total_return(closes)
    if value is not None:
        out["total_return_1y"] = round(value, 4)
    sr = support_resistance(closes)
    if sr["support"] is not None:
        out["support_60d"] = round(sr["support"], 4)
    if sr["resistance"] is not None:
        out["resistance_60d"] = round(sr["resistance"], 4)
    if len(volumes) >= 50 and sum(volumes[-200:]) > 0:
        out["volume_50_vs_200"] = round(
            float(np.mean(volumes[-50:]) / (np.mean(volumes[-200:]) or 1.0)), 4
        )
    out["sessions"] = float(len(bars))
    return out


def _trend_phrase(price: float, sma50: Optional[float], sma200: Optional[float]) -> str:
    if sma50 is None or sma200 is None:
        return "insufficient history to characterise the trend"
    if price > sma50 > sma200:
        return (
            "the last price is above both the 50-day and the 200-day moving average and the "
            "50-day average is above the 200-day average, a configuration consistent with an "
            "established uptrend"
        )
    if price > sma200 and price < sma50 and sma50 > sma200:
        return (
            "the last price is below the 50-day moving average but above the 200-day average, "
            "and the 50-day average remains above the 200-day average, a configuration "
            "consistent with a longer-term uptrend experiencing a near-term pullback"
        )
    if price < sma50 < sma200:
        return (
            "the last price is below both moving averages and the 50-day average is below the "
            "200-day average, a configuration consistent with a downtrend"
        )
    if price < sma200 and sma50 < sma200:
        return (
            "the last price is below the 200-day moving average, a weak configuration"
        )
    return (
        "the last price is above the 50-day average while the 50-day average remains below the "
        "200-day average, a transitional configuration"
    )


def _rsi_phrase(rsi_value: Optional[float]) -> str:
    if rsi_value is None:
        return "The 14-day RSI is unavailable."
    if rsi_value >= 70:
        band = "overbought"
    elif rsi_value <= 30:
        band = "oversold"
    elif rsi_value >= 55:
        band = "firm but not overbought"
    elif rsi_value <= 45:
        band = "soft but not oversold"
    else:
        band = "neutral"
    return f"The 14-day RSI is {rsi_value:.1f}, which is {band}."


def build_snapshot_document(
    ticker: str, provider: MarketDataProvider
) -> Optional[Document]:
    """Render a market-data summary from the ticker's real bar series.

    Returns None rather than a partial document when no bars are available: a
    snapshot with a missing price is worse than no snapshot, because the
    technical agent would report on a series that does not exist.
    """
    ticker = ticker.upper()
    ind = compute_indicators(ticker, provider)
    bars = provider.price_series(ticker)
    if not bars or "last_price" not in ind:
        logger.warning("no bars for %s; skipping market snapshot", ticker)
        return None
    company = provider.company_name(ticker) or ticker

    price = ind["last_price"]
    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")
    vol = ind.get("volatility_60d_annualised", 0.0)
    drawdown = ind.get("max_drawdown", 0.0)
    ratio = ind.get("volume_50_vs_200", 1.0)
    total_ret = ind.get("total_return_1y", 0.0)
    high_52w = max(b.high for b in bars)
    low_52w = min(b.low for b in bars)

    if ratio >= 1.15:
        vol_word = "confirming volume expansion on the move"
    elif ratio <= 0.9:
        vol_word = "indicating no distribution"
    else:
        vol_word = "broadly in line with the longer-term average"

    text = (
        f"{company} ({ticker}) last traded at ${price:,.2f} on {bars[-1].date}. The 50-day moving "
        f"average is ${sma50:,.2f} and the 200-day moving average is ${sma200:,.2f}: "
        f"{_trend_phrase(price, sma50, sma200)}. {_rsi_phrase(ind.get('rsi_14'))} "
        f"Annualised volatility over the trailing 60 sessions is {vol * 100:.1f}%. The maximum "
        f"drawdown from the 52-week high is {drawdown * 100:.1f}%. Average daily volume over the "
        f"last 50 sessions is {ratio:.2f} times the 200-session average, {vol_word}. Total return "
        f"over the trailing year is {total_ret * 100:.1f}%. The 52-week range is "
        f"${low_52w:,.2f} to ${high_52w:,.2f} across {int(ind.get('sessions', 0))} sessions."
    )

    return Document(
        doc_id=f"market-{ticker.lower()}-snapshot",
        title=f"{company} ({ticker}) price and volume statistics",
        text=text,
        doc_type=EvidenceType.MARKET_DATA,
        source="Yahoo Finance (via yfinance)",
        published_at=datetime.fromisoformat(bars[-1].date).replace(tzinfo=timezone.utc),
        reliability=0.85,
        metadata={
            "tickers": [ticker],
            "perspectives": ["technical", "risk"],
            "source_kind": "market_data",
            "derived": "market_snapshot",
            "collection": f"market-{ticker.lower()}",
            "as_of": bars[-1].date,
            "indicators": ind,
        },
    )


__all__ = [
    "Bar",
    "MarketDataProvider",
    "OnlineProvider",
    "build_provider",
    "build_snapshot_document",
    "compute_indicators",
    "DEFAULT_WATCHLIST",
    "max_drawdown",
    "realised_volatility",
    "rsi",
    "sma",
    "support_resistance",
    "total_return",
]
