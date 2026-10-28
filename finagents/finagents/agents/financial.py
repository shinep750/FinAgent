"""Financial analysis agent (fundamental lens)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from ..data.providers import MarketDataProvider
from ..schemas import Perspective, ScoredDocument, Settings, TaskSpec
from .base import FinancialAgent

_PERCENT = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s*(?:%|percent)")
_MONEY = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|trillion)?", re.I
)
_MULTIPLIER = {"billion": 1e9, "million": 1e6, "trillion": 1e12, None: 1.0}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Most specific first: "gross margin" must win over a bare "margin", and
# "revenue growth" over a bare "revenue".
_LABELS: List[tuple[str, str]] = [
    ("free cash flow margin", "fcf_margin"),
    ("gross margin", "gross_margin"),
    ("operating margin", "operating_margin"),
    ("net margin", "net_margin"),
    ("data center margin", "data_center_margin"),
    ("margin", "margin"),
    ("revenue growth", "revenue_growth"),
    ("revenue increase", "revenue_growth"),
    ("increase in revenue", "revenue_growth"),
    ("grew", "revenue_growth"),
    ("revenue", "revenue"),
    ("net sales", "net_sales"),
    ("earnings per share", "eps"),
    ("eps", "eps"),
    ("free cash flow", "free_cash_flow"),
    ("dividend yield", "dividend_yield"),
    ("debt to equity", "debt_to_equity"),
]


class FinancialAnalysisAgent(FinancialAgent):
    """Fundamental analysis: earnings quality, margins, cash flow, valuation.

    Grounding is extracted from the *retrieved text* rather than the model, so
    the numbers in this agent's output are traceable to a document the
    verification layer can re-check.
    """

    name = "financial_analysis"
    perspective = Perspective.FUNDAMENTAL

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
        metrics: Dict[str, float] = {}
        for scored in evidence:
            for key, value in _extract_financials(scored.document.text).items():
                metrics.setdefault(key, value)
        return metrics

    def instructions(self, task: TaskSpec) -> str:
        base = super().instructions(task)
        return (
            base
            + "\nFocus on: revenue growth, gross and operating margin, earnings quality, "
            "free cash flow conversion, customer or geographic concentration, and guidance. "
            "Prefer quantitative claims and cite the document that states each figure. "
            "If guidance is given, note whether it is above or below consensus."
        )


def _extract_financials(text: str) -> Dict[str, float]:
    """Pull labelled percentages and currency figures out of disclosure prose.

    Matching is scoped per sentence. An earlier whole-text version let a label
    match bleed across a sentence boundary and read "revenue" as taking the
    *next* sentence's gross-margin figure -- producing `revenue: 75.0` in a
    report about a company whose revenue was $130.5bn.
    """
    metrics: Dict[str, float] = {}
    sentences = _SENTENCE.split(re.sub(r"\s+", " ", text))

    for sentence in sentences:
        lowered = sentence.lower()
        for label, key in _LABELS:
            if key in metrics:
                continue
            position = lowered.find(label)
            if position == -1:
                continue
            tail = lowered[position + len(label) :]
            match = _PERCENT.search(tail)
            if match:
                value = float(match.group(1))
                before = tail[: match.start()]
                after = tail[match.end() : match.end() + 40]
                if "increase" in before or "growth" in before:
                    # "revenue ... an increase of 114%" is growth, not the level.
                    metrics["revenue_growth"] = value
                elif (
                    re.search(r"of (consolidated |total |net )?(revenue|sales)", after)
                    or "of consolidated" in after
                ):
                    # "Data Center revenue ... 88% of consolidated revenue" is a
                    # mix percentage, and reporting it as `revenue` would be a
                    # number with the wrong unit entirely.
                    metrics[f"{key}_mix_pct"] = value
                else:
                    metrics[key] = value
                break
            money = _MONEY.search(tail)
            if money and key in ("revenue", "net_sales", "free_cash_flow"):
                unit = money.group(2)
                metrics[f"{key}_amount"] = float(
                    money.group(1).replace(",", "")
                ) * _MULTIPLIER.get(unit.lower() if unit else None, 1.0)
                break

    if not any(k.endswith("margin") for k in metrics):
        percent = _PERCENT.search(text)
        if percent:
            metrics["reported_percentage"] = float(percent.group(1))

    for match in _MONEY.finditer(text):
        raw, unit = match.group(1), match.group(2)
        try:
            value = float(raw.replace(",", "")) * _MULTIPLIER.get(
                unit.lower() if unit else None, 1.0
            )
        except ValueError:
            continue
        prefix = text[max(0, match.start() - 60) : match.start()].lower()
        if "revenue" in prefix or "net sales" in prefix:
            metrics.setdefault("revenue_amount", value)
        elif "charge" in prefix:
            metrics.setdefault("charge_amount", value)
        elif "repurchas" in prefix:
            metrics.setdefault("buyback_amount", value)
        elif "dividend" in prefix:
            metrics.setdefault("dividend_amount", value)
        metrics.setdefault("monetary_amount", value)
    return metrics


__all__ = ["FinancialAnalysisAgent"]
