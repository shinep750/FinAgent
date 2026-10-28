"""News analysis agent: material events, catalysts and sentiment."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ..schemas import Perspective, ScoredDocument, TaskSpec
from .base import FinancialAgent

# Event taxonomy. Each pattern maps to a materiality weight: a regulatory action
# or guidance change matters more to a thesis than a routine product headline.
_EVENT_PATTERNS: List[tuple[str, str, float]] = [
    (
        "regulatory_action",
        r"\b(regulat\w+|licen[cs]e|export control|restriction|ban|probe|investigation|lawsuit|antitrust)\b",
        0.9,
    ),
    (
        "guidance_change",
        r"\b(guidance|forecast|outlook|raised (?:its )?|lowered (?:its )?|guided)\b",
        0.8,
    ),
    (
        "earnings_result",
        r"\b(beat|miss|consensus|earnings per share|eps|quarterly results)\b",
        0.75,
    ),
    (
        "capital_allocation",
        r"\b(repurchas\w+|buyback|dividend|offering|issuance)\b",
        0.6,
    ),
    (
        "supply_chain",
        r"\b(supply|packaging|foundry|capacity|inventory|shortage|allocation|ramp)\b",
        0.65,
    ),
    (
        "competition",
        r"\b(competit\w+|market share|second source|disintermediat\w+|rival)\b",
        0.6,
    ),
    ("product", r"\b(launch|unveil|production|platform|next[- ]generation)\b", 0.45),
]

_SENTIMENT = {
    "positive": (
        "beat",
        "raised",
        "strong",
        "record",
        "surge",
        "approval",
        "robust",
        "accelerating",
        "exceeded",
        "improving",
    ),
    "negative": (
        "miss",
        "cut",
        "weak",
        "decline",
        "lawsuit",
        "investigation",
        "shortfall",
        "delayed",
        "halted",
        "writedown",
        "impairment",
    ),
}


class NewsAnalysisAgent(FinancialAgent):
    """Turns a stream of headlines into a material, directional read."""

    name = "news_analysis"
    perspective = Perspective.NEWS

    def grounding(
        self, task: TaskSpec, evidence: Sequence[ScoredDocument]
    ) -> Dict[str, float]:
        if not evidence:
            return {}
        text = " ".join(s.document.text for s in evidence)
        lowered = text.lower()

        scores: Dict[str, float] = {}
        for name, pattern, _weight in _EVENT_PATTERNS:
            hits = len(re.findall(pattern, lowered))
            if hits:
                # Saturating count: the tenth mention of a theme is not ten times
                # as informative as the third.
                scores[f"event_count_{name}"] = round(min(1.0, hits / 4.0), 3)

        pos = sum(lowered.count(w) for w in _SENTIMENT["positive"])
        neg = sum(lowered.count(w) for w in _SENTIMENT["negative"])
        total = pos + neg
        scores["news_sentiment"] = round((pos - neg) / total, 3) if total else 0.0
        scores["document_count"] = float(len(evidence))
        scores["max_reliability"] = round(
            max(s.document.reliability for s in evidence), 3
        )
        return scores

    def instructions(self, task: TaskSpec) -> str:
        base = super().instructions(task)
        return (
            base
            + "\nFocus on: what is materially new, which events are catalysts versus noise, "
            "the direction of news flow, and how management or third parties framed it. "
            "Weigh regulatory actions and guidance changes above routine product news. "
            "State explicitly when the news flow is mixed."
        )


__all__ = ["NewsAnalysisAgent"]
