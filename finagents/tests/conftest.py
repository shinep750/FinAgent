"""Shared fixtures.

Every test runs with a deterministic fake LLM and a small local document
fixture: no network, no Ollama daemon, no API keys. That constraint is
deliberate and independent of the product's online-only design - a test suite
for a financial system that depends on a live model daemon stops being runnable
the moment the daemon is down, and one that hits SEC in a test run is a test
run that breaks on a rate limit.

The fixture documents imitate the *shape* of live SEC/news/market data (doc_ids,
reliability, collections, doc types) so the retrieval and citation logic is
exercised against realistic input without depending on any of it.
"""

from __future__ import annotations

import pytest

from finagents.data.providers import Bar, MarketDataProvider
from finagents.schemas import (
    Document,
    EvidenceType,
    Settings,
)
from finagents.workflow import FinancialResearchWorkflow

from .fake_llm import FakeLLMClient

TICKER = "NVDA"
OTHER = "AMD"


# --------------------------------------------------------------------------
# Fixture documents
# --------------------------------------------------------------------------


def _filing(doc_id: str, title: str, text: str, reliability: float = 0.78) -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        text=text,
        doc_type=EvidenceType.REGULATORY,
        source="SEC EDGAR",
        reliability=reliability,
        metadata={
            "tickers": [TICKER],
            "collection": "sec-nvda-10q",
            "source_kind": "regulatory_filing",
            "perspectives": ["risk", "fundamental"],
            "form": "10-Q",
        },
    )


@pytest.fixture(scope="session")
def documents() -> list:
    return [
        Document(
            doc_id="xbrl-nvda-facts",
            title="NVIDIA CORP (NVDA) XBRL financial facts",
            text=(
                "Fiscal year 2026 for NVIDIA CORP: revenue of $215.9 billion, gross margin of "
                "71.1%, operating margin of 60.4%, net margin of 55.6%, diluted earnings per "
                "share of $4.90, and total assets of $206.8 billion."
            ),
            doc_type=EvidenceType.FILING,
            source="SEC EDGAR XBRL",
            reliability=0.95,
            metadata={
                "tickers": [TICKER],
                "collection": "xbrl-nvda",
                "source_kind": "xbrl_facts",
                "perspectives": ["fundamental"],
            },
        ),
        _filing(
            "sec-nvda-10q-risks",
            "NVDA 10-Q - Item 1A. Risk Factors",
            "Item 1A. Risk Factors. Demand for accelerated computing products is concentrated in a "
            "small number of hyperscale customers, so a reduction in capital expenditure by any "
            "one of them could reduce revenue materially. Export controls restrict sales of "
            "advanced products to certain regions and may be tightened further. The company "
            "competes with custom silicon designs that could displace merchant GPUs. Supply is "
            "dependent on third-party foundries and advanced packaging capacity.",
        ),
        _filing(
            "sec-nvda-10q-mda",
            "NVDA 10-Q - Item 2. Management's Discussion",
            "Item 2. Management's Discussion and Analysis of Financial Condition. Revenue "
            "increased year over year on higher volumes of data center accelerators. Gross "
            "margin declined sequentially as the mix shifted toward large configurations with "
            "more networking content. We expect continued volatility in the pace of orders.",
        ),
        Document(
            doc_id="news-nvda-reuters-1",
            title="Nvidia sets biggest-ever buyback plan as AI chip competition weighs on growth",
            text=(
                "Nvidia said it would authorise $150 billion of share repurchases, the largest "
                "such plan for the company, while analysts questioned the durability of demand "
                "as custom silicon competition increased. The company said AI infrastructure "
                "spending would continue to grow. Shares rose in after-hours trading."
            ),
            doc_type=EvidenceType.NEWS,
            source="Reuters (via Google News)",
            reliability=0.80,
            metadata={
                "tickers": [TICKER],
                "collection": "news-nvda",
                "source_kind": "news",
                "outlet": "Reuters",
                "perspectives": ["news", "risk"],
            },
        ),
        Document(
            doc_id="market-nvda-snapshot",
            title="NVIDIA CORP (NVDA) price and volume statistics",
            text=(
                "NVIDIA CORP (NVDA) last traded at $181.20. The 50-day moving average is "
                "$169.40 and the 200-day moving average is $151.05: the last price is above "
                "both. The 14-day RSI is 58.2, which is firm but not overbought. Annualised "
                "volatility over the trailing 60 sessions is 41.3%."
            ),
            doc_type=EvidenceType.MARKET_DATA,
            source="Yahoo Finance (via yfinance)",
            reliability=0.85,
            metadata={
                "tickers": [TICKER],
                "collection": "market-nvda",
                "source_kind": "market_data",
                "perspectives": ["technical", "risk"],
                "indicators": {"sma_50": 169.40, "sma_200": 151.05, "rsi_14": 58.2},
            },
        ),
        Document(
            doc_id="news-amd-reuters-1",
            title="AMD expands server share as hyperscalers diversify away from single suppliers",
            text=(
                "Advanced Micro Devices reported server processor revenue growth and said "
                "several hyperscale customers had committed to large deployments, reflecting a "
                "wider push to diversify accelerator supply."
            ),
            doc_type=EvidenceType.NEWS,
            source="Reuters (via Google News)",
            reliability=0.80,
            metadata={
                "tickers": [OTHER],
                "collection": "news-amd",
                "source_kind": "news",
                "outlet": "Reuters",
                "perspectives": ["news"],
            },
        ),
    ]


# --------------------------------------------------------------------------
# Offline-safe provider double
# --------------------------------------------------------------------------


class FixtureProvider(MarketDataProvider):
    """Serves the fixture documents and a deterministic price series.

    Implements the same interface as `OnlineProvider`, so the workflow cannot tell
    the difference - which is exactly the property that makes the test worth
    having.
    """

    name = "fixture"
    is_live = False

    def __init__(self, documents: list, known=("NVDA", "AMD", "AAPL", "BRK-B")) -> None:
        self._documents = list(documents)
        self._known = {t.upper() for t in known}
        self.loaded_for: list = []

    def load_documents(self, tickers=None) -> list:
        wanted = [t.upper() for t in (tickers or [])]
        self.loaded_for.append(tuple(wanted))
        if not wanted:
            return list(self._documents)
        selected = [
            d
            for d in self._documents
            if set(d.metadata.get("tickers", [])) & set(wanted)
        ]
        return selected or list(self._documents)

    def price_series(self, ticker: str) -> list:
        symbol = ticker.upper()
        bars = []
        price = 100.0
        for day in range(260):
            # Deterministic, monotonic-ish walk: no RNG, so indicators are stable.
            price *= 1.0 + (((day * 7) % 5) - 2) * 0.0015
            bars.append(
                Bar(
                    date=f"2025-{1 + day // 28:02d}-{(day % 28) + 1:02d}",
                    open=price * 0.995,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    volume=1_000_000 + (day % 11) * 25_000,
                )
            )
        return bars

    def resolve_ticker(self, symbol: str) -> str:
        """Same share-class normalisation as the real provider.

        If the double resolved `BRK.B` and the product did not, the tests would
        pass against an interface the product does not actually implement.
        """
        from finagents.data.providers import _ticker_spellings

        raw = (symbol or "").upper().strip().lstrip("$")
        for candidate in _ticker_spellings(raw):
            if candidate in self._known:
                return candidate
        return None

    def company_name(self, ticker: str) -> str:
        return {"NVDA": "NVIDIA CORP", "AMD": "ADVANCED MICRO DEVICES"}.get(
            ticker.upper()
        )


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(parallel_agents=False)


@pytest.fixture(scope="session")
def llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture(scope="session")
def provider(documents) -> FixtureProvider:
    return FixtureProvider(documents)


@pytest.fixture(scope="session")
def workflow(settings, llm, provider) -> FinancialResearchWorkflow:
    return FinancialResearchWorkflow(settings=settings, llm=llm, provider=provider)


@pytest.fixture(scope="session")
def nvda_state(workflow: FinancialResearchWorkflow):
    return workflow.run(
        "What are the main risks facing NVDA and is the valuation justified?"
    )
