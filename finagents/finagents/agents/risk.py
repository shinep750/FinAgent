"""Risk analysis agent: downside channels, fragility and volatility.

Risk framing is intentionally inverted relative to the other agents: a
constructive finding here is a *bearish* one. The base coercion logic maps a
positive risk score to a bearish stance so the stance fields stay comparable
across the four agents, and conflict detection can therefore treat "risk agent
says bearish" as genuinely disagreeing with "fundamental agent says bullish".
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from ..data.providers import MarketDataProvider, compute_indicators
from ..schemas import Perspective, ScoredDocument, Settings, TaskSpec
from .base import FinancialAgent

# Each risk channel carries a weight so the aggregate score reflects how much a
# given channel historically matters, not just whether it was mentioned.
_RISK_CHANNELS: List[tuple[str, str, float]] = [
    (
        "customer_concentration",
        r"(?:four|top \d+|largest|largest \w+) (?:direct )?customers|concentration of (?:revenue|sales)",
        1.0,
    ),
    (
        "regulatory",
        r"export control|regulat\w+|licen[cs]e|antitrust|lawsuit|investigation|tariff|restrictions?",
        0.95,
    ),
    (
        "supply_chain",
        r"third[- ]party|foundr\w+|packaging capacity|single (?:or limited )?source|allocation of .* capacity|supply chain",
        0.85,
    ),
    (
        "cyclicality",
        r"capital expenditure|cyclical|demand could decline|adversely affect|economic downturn|recession",
        0.7,
    ),
    ("valuation", r"valuation|price target|de-?rate|multiple|expectations", 0.6),
    ("competition", r"competit\w+|second[- ]source|market share|disintermediat", 0.65),
    (
        "margin_compression",
        r"gross margin will be lower|margin compression|lower[- ]margin|headwind",
        0.75,
    ),
    (
        "guidance_uncertainty",
        r"cannot predict|uncertain|range of outcomes|we have modelled",
        0.6,
    ),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class RiskAnalysisAgent(FinancialAgent):
    """Downside scenarios, concentration, exposure and volatility."""

    name = "risk_analysis"
    perspective = Perspective.RISK

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
        text = " ".join(s.document.text for s in evidence)
        lowered = text.lower()
        metrics: Dict[str, float] = {}

        weighted = 0.0
        weight_sum = 0.0
        for name, pattern, weight in _RISK_CHANNELS:
            hits = len(re.findall(pattern, lowered))
            if hits:
                metrics[f"channel_{name}"] = min(1.0, hits / 3.0)
                weighted += weight * min(1.0, hits / 3.0)
                weight_sum += weight
        metrics["risk_channel_intensity"] = (
            round(weighted / weight_sum, 3) if weight_sum else 0.0
        )
        metrics["distinct_channels"] = float(
            sum(1 for n, _, _ in _RISK_CHANNELS if f"channel_{n}" in metrics)
        )

        quantified = _quantify(text)
        metrics.update(quantified)

        for ticker in self._tickers(task)[:2]:
            if self.provider is None:
                continue
            try:
                indicators = compute_indicators(ticker, self.provider)
            except (KeyError, ValueError):
                continue
            vol = indicators.get("volatility_60d_annualised")
            drawdown = indicators.get("max_drawdown")
            if vol is not None:
                metrics[f"{ticker.lower()}_volatility"] = vol * 100.0
                metrics["volatility_band"] = _vol_band(vol)
            if drawdown is not None:
                metrics[f"{ticker.lower()}_max_drawdown"] = drawdown
                metrics["drawdown_band"] = _drawdown_band(drawdown)
            total_return = indicators.get("total_return_1y")
            if total_return is not None:
                metrics[f"{ticker.lower()}_total_return_1y"] = total_return
            ratio = indicators.get("volume_50_vs_200")
            if ratio is not None:
                metrics[f"{ticker.lower()}_volume_ratio"] = ratio
        return metrics

    def instructions(self, task: TaskSpec) -> str:
        base = super().instructions(task)
        return (
            base
            + "\nFocus on: named downside channels, concentration, regulatory and litigation "
            "exposure, supply dependency, valuation fragility and volatility. For each risk, "
            "say what would have to happen for it to materialise and what the evidence says "
            "about its current likelihood. Quantify where the evidence quantifies. "
            "A constructive risk finding means elevated risk."
        )

    def _tickers(self, task: TaskSpec) -> List[str]:
        return list(task.metadata_filters.tickers)


def _quantify(text: str) -> Dict[str, float]:
    """Extract the numbers that make a risk concrete rather than rhetorical."""
    out: Dict[str, float] = {}
    concentration = re.search(
        r"(\d{1,2})\s*%\s*of\s*(?:Data Center\s*)?revenue", text, re.IGNORECASE
    )
    if concentration:
        out["concentration_pct"] = float(concentration.group(1))
    charge = re.search(r"\$(\d{1,3}(?:\.\d+)?)\s*billion charge", text, re.IGNORECASE)
    if charge:
        out["charge_billion"] = float(charge.group(1))
    affected = re.search(
        r"(?:could affect|affect) (?:a )?(low-|mid-|high-)?\w*single[- ]digit percentage",
        text,
        re.IGNORECASE,
    )
    if affected:
        out["revenue_at_risk_band"] = {
            "low-": 0.05,
            "mid-": 0.07,
            "high-": 0.09,
        }.get((affected.group(1) or "").lower(), 0.07)
    return out


def _vol_band(vol: float) -> float:
    if vol >= 0.6:
        return 3.0
    if vol >= 0.4:
        return 2.0
    if vol >= 0.25:
        return 1.0
    return 0.0


def _drawdown_band(drawdown: float) -> float:
    if drawdown >= 0.5:
        return 3.0
    if drawdown >= 0.3:
        return 2.0
    if drawdown >= 0.15:
        return 1.0
    return 0.0


__all__ = ["RiskAnalysisAgent"]
