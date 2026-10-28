"""Technical analysis agent.

Unlike the other agents, this one mostly *computes*. Moving averages, RSI,
realised volatility, drawdown and volume ratios are calculated from the actual
price series and handed to the model as grounding, so the numbers it reports
are not model-invented. The model's job is interpretation, not arithmetic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from ..data.providers import MarketDataProvider, compute_indicators
from ..schemas import Perspective, ScoredDocument, Settings, TaskSpec
from .base import FinancialAgent

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent(FinancialAgent):
    """Price trend, momentum and volume structure."""

    name = "technical_analysis"
    perspective = Perspective.TECHNICAL

    def __init__(
        self,
        llm,
        settings: Optional[Settings] = None,
        provider: Optional[MarketDataProvider] = None,
    ) -> None:
        super().__init__(llm, settings)
        self.provider = provider

    def grounding(
        self, task: TaskSpec, evidence: Sequence[ScoredDocument]
    ) -> Dict[str, float]:
        tickers = self._tickers(task, evidence)
        if not tickers or self.provider is None:
            return {}
        metrics: Dict[str, float] = {}
        for ticker in tickers[:2]:
            slug = ticker.lower()
            try:
                indicators = compute_indicators(ticker, self.provider)
            except (KeyError, ValueError) as exc:
                logger.warning("no indicator data for %s: %s", ticker, exc)
                continue
            # One naming convention for the whole block: ticker first. A metric
            # named `momentum_state_nvda` alongside `nvda_sma_50` reads as two
            # unrelated schemas, and a multi-ticker run makes the inconsistency
            # obvious the moment the states sort apart from the numbers.
            metrics[f"{slug}_trend_state"] = _trend_state(indicators)
            metrics[f"{slug}_momentum_state"] = _momentum_state(indicators)
            for key, value in indicators.items():
                metrics[f"{slug}_{key}"] = value
            for signal, value in _derived_signals(indicators).items():
                metrics[f"{slug}_{signal}"] = value
        return metrics

    def instructions(self, task: TaskSpec) -> str:
        base = super().instructions(task)
        return (
            base
            + "\nFocus on: trend direction and structure, moving-average configuration, "
            "momentum (RSI), volume confirmation, support and resistance, and volatility. "
            "The COMPUTED METRICS line already contains the arithmetic -- use those exact "
            "values and do not recompute or round them differently. Distinguish a "
            "long-term trend from a near-term pullback."
        )

    def _tickers(self, task: TaskSpec, evidence: Sequence[ScoredDocument]) -> List[str]:
        if task.metadata_filters.tickers:
            return list(task.metadata_filters.tickers)
        found: List[str] = []
        for scored in evidence:
            for ticker in scored.document.metadata.get("tickers", []):
                if ticker not in found:
                    found.append(ticker)
        return found


def _trend_state(ind: Dict[str, float]) -> float:
    """Encode trend structure as a small ordinal score.

    2 = price above both averages, 1 = above the long average only,
    0 = below the long average, -1 = below both.
    """
    price = ind.get("last_price")
    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")
    if price is None or sma50 is None or sma200 is None:
        return 0.0
    if price > sma50 > sma200:
        return 2.0
    if price > sma200 and price <= sma50:
        return 1.0
    if price <= sma200 and sma50 > sma200:
        return 0.0
    return -1.0


def _momentum_state(ind: Dict[str, float]) -> float:
    rsi_value = ind.get("rsi_14")
    if rsi_value is None:
        return 0.0
    if rsi_value >= 70:
        return -1.0
    if rsi_value <= 30:
        return 1.0
    return 0.0


def _derived_signals(ind: Dict[str, float]) -> Dict[str, float]:
    """Signals that are only meaningful as derived quantities."""
    out: Dict[str, float] = {}
    price, sma50, sma200 = ind.get("last_price"), ind.get("sma_50"), ind.get("sma_200")
    if price and sma50:
        out["pct_above_sma_50"] = round((price / sma50 - 1.0) * 100.0, 3)
    if sma50 and sma200:
        out["sma_50_vs_sma_200_pct"] = round((sma50 / sma200 - 1.0) * 100.0, 3)
    if price and sma200:
        out["pct_above_sma_200"] = round((price / sma200 - 1.0) * 100.0, 3)
    support, resistance = ind.get("support_60d"), ind.get("resistance_60d")
    if price and support and resistance and resistance > support:
        # Position of price within the 60-day range, 0 = at support.
        out["range_position"] = round((price - support) / (resistance - support), 3)
    return out


__all__ = ["TechnicalAnalysisAgent"]
