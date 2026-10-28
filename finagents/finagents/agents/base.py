"""Agent base class and registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Type

from ..llm.base import LLMClient
from ..prompts import ANALYST_SYSTEM, format_evidence, role_header
from ..schemas import (
    AgentFinding,
    Citation,
    Claim,
    ClaimType,
    Perspective,
    ScoredDocument,
    Settings,
    Stance,
    TaskSpec,
)
from ..llm.base import build_prompt

logger = logging.getLogger(__name__)

_AGENT_NAMES = {
    Perspective.FUNDAMENTAL: "financial_analysis",
    Perspective.NEWS: "news_analysis",
    Perspective.TECHNICAL: "technical_analysis",
    Perspective.RISK: "risk_analysis",
}


class FinancialAgent(ABC):
    """Base class for the specialised analysis agents.

    Subclasses declare a perspective and provide perspective-specific grounding
    (computed metrics, domain guidance). The shared `run` method handles prompt
    assembly, the LLM call, and coercion of the response into a validated
    `AgentFinding` -- with citation repair, because a model that cites a
    document id which was not in its evidence has produced an unsupportable
    claim, and silently keeping it would defeat the verification layer.
    """

    name: str = "agent"
    perspective: Perspective = Perspective.FUNDAMENTAL

    def __init__(self, llm: LLMClient, settings: Optional[Settings] = None) -> None:
        self.llm = llm
        self.settings = settings or Settings()

    @abstractmethod
    def grounding(
        self, task: TaskSpec, evidence: Sequence[ScoredDocument]
    ) -> Dict[str, float]:
        """Perspective-specific computed context, e.g. real indicator values."""

    def instructions(self, task: TaskSpec) -> str:
        subject = ", ".join(task.metadata_filters.tickers) or task.description[:120]
        return (
            f"{role_header('analyst')}\n"
            f"TASK: {task.task_id}\n"
            f"PERSPECTIVE: {self.perspective.value}\n"
            f"SUBJECT: {subject}\n"
            f"TASK DESCRIPTION: {task.description}\n"
            "Analyse ONLY the evidence provided. Cite the exact `doc_id` of every "
            "document you rely on. If the evidence does not support a conclusion, say so."
        )

    def output_schema(self) -> str:
        return (
            '{"summary": "string", "stance": "bullish|bearish|neutral|mixed", '
            '"confidence": 0.0, "claims": [{"text": "string", "claim_type": '
            '"quantitative|qualitative|causal|comparative|predictive", "stance": '
            '"bullish|bearish|neutral", "confidence": 0.0, "citations": [{"doc_id": '
            '"string", "title": "string", "source": "string", "quote": "string"}], '
            '"metrics": {"name": 0.0}, "caveats": ["string"]}], '
            '"key_metrics": {"name": 0.0}, "risks": ["string"], '
            '"caveats": ["string"], "reasoning_steps": ["string"]}'
        )

    def run(self, task: TaskSpec, evidence: Sequence[ScoredDocument]) -> AgentFinding:
        metrics = self.grounding(task, evidence)
        context = self._build_context(evidence, metrics)
        prompt = build_prompt(
            system=ANALYST_SYSTEM[self.perspective],
            instruction=self.instructions(task),
            context=context,
            output_schema=self.output_schema(),
        )
        payload = self.llm.generate_json(
            prompt,
            ANALYST_SYSTEM[self.perspective],
            temperature=self.settings.llm_temperature,
        )
        return self.coerce(task, payload, evidence, metrics)

    def _build_context(
        self, evidence: Sequence[ScoredDocument], metrics: Dict[str, float]
    ) -> str:
        parts = [format_evidence(evidence)]
        if metrics:
            rendered = ", ".join(f"{k}={v:g}" for k, v in sorted(metrics.items()))
            parts.append(
                f"COMPUTED METRICS (calculated from the market data, not from the text):\n{rendered}"
            )
        if evidence and evidence[0].kg_context:
            graph = "\n".join(evidence[0].kg_context)
            parts.append(f"KNOWLEDGE GRAPH CONTEXT (structured relations):\n{graph}")
        return "\n\n".join(parts)

    def coerce(
        self,
        task: TaskSpec,
        payload: object,
        evidence: Sequence[ScoredDocument],
        metrics: Dict[str, float],
    ) -> AgentFinding:
        """Turn a raw model payload into a validated, citation-repaired finding."""
        data = payload if isinstance(payload, dict) else {}
        by_id = {d.document.doc_id: d.document for d in evidence}
        bad_citations: list[str] = []
        uncited: list[str] = []

        claims: List[Claim] = []
        for raw in data.get("claims", []) or []:
            if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
                continue
            citations: List[Citation] = []
            for raw_cite in raw.get("citations", []) or []:
                if not isinstance(raw_cite, dict):
                    continue
                doc_id = str(raw_cite.get("doc_id", "")).strip()
                doc = by_id.get(doc_id)
                if doc is None:
                    # Hallucinated or out-of-evidence citation.
                    bad_citations.append(doc_id or "<blank>")
                    continue
                citations.append(
                    Citation(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        source=doc.source,
                        quote=str(raw_cite.get("quote", ""))[:500],
                        url=doc.url,
                    )
                )
            if not citations:
                uncited.append(str(raw.get("text"))[:60])
            claims.append(
                Claim(
                    claim_id="",
                    text=str(raw["text"])[:600],
                    claim_type=_as_claim_type(raw.get("claim_type")),
                    stance=_as_stance(raw.get("stance"), Perspective.FUNDAMENTAL),
                    confidence=_as_float(raw.get("confidence"), 0.5),
                    citations=citations,
                    metrics=_as_float_dict(raw.get("metrics")),
                    caveats=[str(c)[:200] for c in (raw.get("caveats") or [])][:5],
                )
            )

        caveats = [str(c)[:200] for c in (data.get("caveats") or [])][:6]
        if bad_citations:
            caveats.append(
                f"{len(bad_citations)} citation(s) named documents absent from the "
                f"retrieved evidence and were discarded: {', '.join(bad_citations[:5])}"
            )
        if uncited:
            caveats.append(
                f"{len(uncited)} claim(s) were returned without a citable reference: "
                f"{', '.join(uncited[:5])}"
            )
        if not claims:
            caveats.append("The agent returned no citable claims.")

        key_metrics = _as_float_dict(data.get("key_metrics"))
        key_metrics.update({k: v for k, v in metrics.items() if k not in key_metrics})

        return AgentFinding(
            task_id=task.task_id,
            agent_name=self.name,
            perspective=self.perspective,
            summary=str(data.get("summary", "")).strip()[:1200]
            or "No summary produced.",
            stance=_as_stance(data.get("stance"), self.perspective),
            confidence=_as_float(data.get("confidence"), 0.4),
            claims=claims,
            evidence=list(evidence),
            key_metrics=key_metrics,
            risks=[str(r)[:240] for r in (data.get("risks") or [])][:6],
            caveats=caveats,
            reasoning_steps=[str(s)[:240] for s in (data.get("reasoning_steps") or [])][
                :8
            ],
        )


def _as_stance(value: object, perspective: Perspective) -> Stance:
    try:
        return Stance(str(value).lower())
    except (ValueError, AttributeError):
        return Stance.NEUTRAL


def _as_claim_type(value: object) -> ClaimType:
    try:
        return ClaimType(str(value).lower())
    except (ValueError, AttributeError):
        return ClaimType.QUALITATIVE


def _as_float(value: object, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_float_dict(value: object) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


class AgentRegistry:
    """Name -> agent class, with perspective routing."""

    def __init__(
        self, classes: Optional[Dict[str, Type[FinancialAgent]]] = None
    ) -> None:
        self.classes: Dict[str, Type[FinancialAgent]] = classes or {}

    def register(self, name: str, cls: Type[FinancialAgent]) -> None:
        self.classes[name] = cls

    def get(self, name: str, llm: LLMClient, settings: Settings) -> FinancialAgent:
        if name not in self.classes:
            raise KeyError(
                f"unknown agent {name!r}; registered: {sorted(self.classes)}"
            )
        return self.classes[name](llm, settings)

    def for_perspective(self, perspective: Perspective) -> str:
        return _AGENT_NAMES[perspective]

    def names(self) -> List[str]:
        return sorted(self.classes)


__all__ = ["AgentRegistry", "FinancialAgent"]
