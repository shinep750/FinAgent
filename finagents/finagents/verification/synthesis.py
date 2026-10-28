"""Multi-agent synthesis.

Synthesis is gated on verification: a claim that failed verification cannot
reach the thesis, the sections, or the key-metrics table. It is listed in the
report's excluded set instead, with its reason, because silently dropping
claims would hide the fact that an agent asserted something unsupported.

The overall stance is decided by weighted evidence, not by counting agents, and
an unresolved directional conflict forces a `mixed` thesis rather than letting
a majority vote manufacture certainty.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from ..llm.base import LLMClient, build_prompt
from ..prompts import SYSTEM_SYNTHESIZER, format_findings_context, role_header
from ..schemas import (
    AgentError,
    AgentFinding,
    Citation,
    Claim,
    ClaimVerification,
    Conflict,
    ConflictType,
    ConfidenceReport,
    FinancialResearchReport,
    ScoredDocument,
    Settings,
    Stance,
    SynthesizedSection,
    UserQuery,
    Verdict,
)

logger = logging.getLogger(__name__)

_USABLE_VERDICTS = (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


class Synthesizer:
    def __init__(self, llm: LLMClient, settings: Optional[Settings] = None) -> None:
        self.llm = llm
        self.settings = settings or Settings()

    def synthesise(
        self,
        query: UserQuery,
        plan,
        findings: Sequence[AgentFinding],
        verifications: Sequence[ClaimVerification],
        conflicts: Sequence[Conflict],
        confidence: ConfidenceReport,
        errors: Sequence[AgentError] = (),
        execution_trace: Sequence[dict] = (),
        data_sources: Optional[Dict[str, str]] = None,
    ) -> FinancialResearchReport:
        usable, excluded = self._partition_claims(findings, verifications)
        evidence = self._rank_evidence(findings)
        stance, stance_note = self._overall_stance(findings, conflicts, confidence)
        sections = self._sections(findings, usable, query, stance_note)
        thesis = self._thesis(query, findings, usable, stance, confidence, conflicts)

        return FinancialResearchReport(
            report_id=_id("rep", query.raw_text, str(len(findings))),
            query=query,
            thesis=thesis,
            overall_stance=stance,
            confidence=confidence,
            sections=sections,
            verified_claims=list(verifications),
            conflicts=list(conflicts),
            findings=list(findings),
            evidence=evidence,
            key_metrics=self._key_metrics(findings, usable),
            risks=self._risks(findings, confidence),
            limitations=self._limitations(
                plan, findings, verifications, excluded, errors, confidence
            ),
            data_sources=dict(data_sources or {}),
            execution_trace=list(execution_trace),
        )

    # -- claim gating -----------------------------------------------------

    def _partition_claims(
        self,
        findings: Sequence[AgentFinding],
        verifications: Sequence[ClaimVerification],
    ) -> Tuple[Dict[str, List[Claim]], Dict[str, ClaimVerification]]:
        by_id = {v.claim_id: v for v in verifications}
        usable: Dict[str, List[Claim]] = {}
        excluded: Dict[str, ClaimVerification] = {}
        for finding in findings:
            kept: List[Claim] = []
            for claim in finding.claims:
                verdict = by_id.get(claim.claim_id)
                if verdict is None or verdict.verdict not in _USABLE_VERDICTS:
                    if verdict is not None:
                        excluded[claim.claim_id] = verdict
                    continue
                kept.append(claim)
            if kept:
                usable[finding.task_id] = kept
        return usable, excluded

    # -- stance -----------------------------------------------------------

    def _overall_stance(
        self,
        findings: Sequence[AgentFinding],
        conflicts: Sequence[Conflict],
        confidence: ConfidenceReport,
    ) -> Tuple[Stance, str]:
        directional = [
            f for f in findings if f.stance in (Stance.BULLISH, Stance.BEARISH)
        ]
        unresolved = [
            c
            for c in conflicts
            if c.conflict_type == ConflictType.DIRECTION and not c.resolved
        ]
        if not directional:
            return Stance.NEUTRAL, "no agent reached a directional conclusion"

        bull = sum(f.confidence for f in directional if f.stance == Stance.BULLISH)
        bear = sum(f.confidence for f in directional if f.stance == Stance.BEARISH)
        if unresolved:
            # A live disagreement is a mixed picture. Do not let a weighted vote
            # resolve it into a false consensus.
            return (
                Stance.MIXED,
                f"{len(unresolved)} unresolved directional conflict(s) between agents",
            )
        if bull == bear:
            return Stance.MIXED, "agents split evenly on conviction"
        if confidence.overall < 0.3:
            return (
                Stance.MIXED,
                f"a lean exists but confidence is only {confidence.band()}",
            )
        if bull > bear:
            return (
                Stance.BULLISH,
                f"bullish weight {bull:.2f} exceeds bearish {bear:.2f}",
            )
        return Stance.BEARISH, f"bearish weight {bear:.2f} exceeds bullish {bull:.2f}"

    # -- sections ---------------------------------------------------------

    def _sections(
        self,
        findings: Sequence[AgentFinding],
        usable: Dict[str, List[Claim]],
        query: UserQuery,
        stance_note: str,
    ) -> List[SynthesizedSection]:
        drafted = self._draft_sections(findings, usable, query, stance_note)
        by_task = {f.task_id: f for f in findings}
        sections: List[SynthesizedSection] = []
        for task_id, draft in drafted.items():
            finding = by_task.get(task_id)
            if finding is None:
                continue
            claims = usable.get(task_id, [])
            citations: List[Citation] = []
            for claim in claims:
                citations.extend(claim.citations)
            body = str(draft.get("body") or "").strip() or finding.summary
            sections.append(
                SynthesizedSection(
                    heading=str(
                        draft.get("heading")
                        or f"{finding.perspective.value.title()} perspective"
                    ),
                    perspective=finding.perspective,
                    body=body[:2500],
                    stance=_as_stance(draft.get("stance"), finding.stance),
                    confidence=round(
                        min(
                            1.0, _as_float(draft.get("confidence"), finding.confidence)
                        ),
                        4,
                    ),
                    claim_ids=[c.claim_id for c in claims],
                    citations=_dedupe_citations(citations),
                )
            )
        if not sections:
            for finding in findings:
                sections.append(
                    SynthesizedSection(
                        heading=f"{finding.perspective.value.title()} perspective",
                        perspective=finding.perspective,
                        body=finding.summary,
                        stance=finding.stance,
                        confidence=finding.confidence,
                        claim_ids=[c.claim_id for c in usable.get(finding.task_id, [])],
                        citations=_dedupe_citations(
                            [
                                c
                                for claim in usable.get(finding.task_id, [])
                                for c in claim.citations
                            ]
                        ),
                    )
                )
        return sections

    def _draft_sections(
        self,
        findings: Sequence[AgentFinding],
        usable: Dict[str, List[Claim]],
        query: UserQuery,
        stance_note: str,
    ) -> Dict[str, Dict[str, object]]:
        lines: List[str] = []
        for finding in findings:
            claims = usable.get(finding.task_id, [])
            if not claims:
                continue
            rendered = "\n".join(
                f"  - [{c.claim_id}] {c.text} (stance={c.stance.value}, "
                f"cites={','.join(c.doc_id for c in c.citations) or 'none'})"
                for c in claims
            )
            lines.append(
                f"FINDING {finding.task_id} | agent={finding.agent_name} | "
                f"perspective={finding.perspective.value} | stance={finding.stance.value} | "
                f"confidence={finding.confidence:.2f}\n"
                f"summary: {finding.summary}\n"
                f"claims:\n{rendered}"
            )
        if not lines:
            return {}
        prompt = build_prompt(
            system=SYSTEM_SYNTHESIZER,
            instruction=(
                f"{role_header('synthesizer')}\n"
                f"USER QUESTION: {query.raw_text}\n"
                f"NOTE: {stance_note}\n"
                "Write one section per agent finding. Each section must use ONLY the "
                "verified claims listed for that finding, must keep the agent's stance, and "
                "must not introduce any fact or number that is not in those claims."
            ),
            context=format_findings_context(findings),
            output_schema=(
                '{"sections": [{"task_id": "string", "heading": "string", '
                '"body": "string", "stance": "bullish|bearish|neutral|mixed", '
                '"confidence": 0.0}]}'
            ),
        )
        try:
            payload = self.llm.generate_json(
                prompt, SYSTEM_SYNTHESIZER, temperature=self.settings.llm_temperature
            )
        except Exception as exc:  # noqa: BLE001 - fall back to raw agent summaries
            logger.warning(
                "section drafting unavailable (%s); using agent summaries", exc
            )
            return {}
        drafted: Dict[str, Dict[str, object]] = {}
        for entry in (payload or {}).get("sections", []) or []:
            if isinstance(entry, dict) and entry.get("task_id"):
                drafted[str(entry["task_id"])] = entry
        return drafted

    # -- thesis -----------------------------------------------------------

    def _thesis(
        self,
        query: UserQuery,
        findings: Sequence[AgentFinding],
        usable: Dict[str, List[Claim]],
        stance: Stance,
        confidence: ConfidenceReport,
        conflicts: Sequence[Conflict],
    ) -> str:
        drafted = ""
        try:
            prompt = build_prompt(
                system=SYSTEM_SYNTHESIZER,
                instruction=(
                    f"{role_header('synthesizer')}\n"
                    f"USER QUESTION: {query.raw_text}\n"
                    f"RESOLVED STANCE: {stance.value}\n"
                    f"CONFIDENCE: {confidence.overall:.2f} ({confidence.band()})\n"
                    "Write a two-to-four sentence thesis answering the user's question. "
                    "State the direction, the single strongest piece of evidence for it, the "
                    "main thing that argues against it, and the key uncertainty. "
                    "Do not introduce facts or numbers that are not in the findings."
                ),
                context=format_findings_context(findings),
                output_schema='{"thesis": "string"}',
            )
            payload = self.llm.generate_json(
                prompt, SYSTEM_SYNTHESIZER, temperature=self.settings.llm_temperature
            )
            drafted = str((payload or {}).get("thesis", "")).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "thesis drafting unavailable (%s); composing deterministically", exc
            )

        if drafted:
            return drafted[:1500]

        # Deterministic composition, so a report is always producible.
        by_perspective = {f.perspective.value: f for f in findings}
        parts = [
            f"Across {len(findings)} specialised agents the evidence points "
            f"{stance.value} (confidence {confidence.overall:.2f}, {confidence.band()})."
        ]
        for perspective, finding in by_perspective.items():
            parts.append(
                f"{perspective.title()}: {finding.stance.value} at {finding.confidence:.2f} "
                f"-- {finding.summary}"
            )
        if conflicts:
            unresolved = [c for c in conflicts if not c.resolved]
            if unresolved:
                parts.append(
                    f"{len(unresolved)} disagreement(s) between agents remain unresolved: "
                    + "; ".join(c.description for c in unresolved[:2])
                )
        return " ".join(parts)[:2000]

    # -- aggregates -------------------------------------------------------

    def _rank_evidence(self, findings: Sequence[AgentFinding]) -> List[ScoredDocument]:
        """The evidence table: highest retrieval score first, one row per document."""
        best: Dict[str, ScoredDocument] = {}
        for finding in findings:
            for scored in finding.evidence:
                existing = best.get(scored.document.doc_id)
                if existing is None or scored.score > existing.score:
                    best[scored.document.doc_id] = scored
        return sorted(best.values(), key=lambda s: s.score, reverse=True)[:20]

    def _key_metrics(
        self, findings: Sequence[AgentFinding], usable: Dict[str, List[Claim]]
    ) -> Dict[str, float]:
        """Metrics from verified claims plus agent groundings, deduped by name.

        Agent-computed values win over model-extracted ones only on a tie, so a
        figure the model read from a filing is not silently overwritten -- but a
        value that was actually computed from the price series is preferred.
        """
        out: Dict[str, float] = {}
        for finding in findings:
            claims = usable.get(finding.task_id, [])
            claim_metrics: Dict[str, float] = {}
            for claim in claims:
                claim_metrics.update(claim.metrics)
            for metric, value in claim_metrics.items():
                if _is_reportable(metric):
                    out.setdefault(metric, float(value))
            for metric, value in finding.key_metrics.items():
                if metric not in out and _is_reportable(metric):
                    out[metric] = float(value)
        return {
            k: round(v, 6)
            for k, v in sorted(out.items())
            if isinstance(v, (int, float))
        }

    def _risks(
        self, findings: Sequence[AgentFinding], confidence: ConfidenceReport
    ) -> List[str]:
        risks: List[str] = []
        for finding in findings:
            for risk in finding.risks:
                text = risk.strip()
                if text and text not in risks:
                    risks.append(text[:400])
        return risks[:12]

    def _limitations(
        self,
        plan,
        findings: Sequence[AgentFinding],
        verifications: Sequence[ClaimVerification],
        excluded: Dict[str, ClaimVerification],
        errors: Sequence[AgentError],
        confidence: ConfidenceReport,
    ) -> List[str]:
        limits: List[str] = []
        if plan is not None and getattr(plan, "skipped_perspectives", None):
            limits.append(
                "Perspectives not analysed for this question: "
                + ", ".join(p.value for p in plan.skipped_perspectives)
                + "."
            )
        if excluded:
            verdicts: Dict[str, int] = {}
            for verdict in excluded.values():
                verdicts[verdict.verdict.value] = (
                    verdicts.get(verdict.verdict.value, 0) + 1
                )
            detail = ", ".join(
                f"{count} {name}" for name, count in sorted(verdicts.items())
            )
            limits.append(
                f"{len(excluded)} agent claim(s) were excluded from the conclusion because "
                f"verification did not support them ({detail})."
            )
        for finding in findings:
            for caveat in finding.caveats:
                text = caveat.strip()
                if text and text not in limits:
                    limits.append(f"{finding.agent_name}: {text}"[:400])
        for error in errors:
            limits.append(f"Agent {error.agent_name} failed: {error.error}"[:400])
        if confidence.penalties:
            limits.extend(confidence.penalties[:6])
        return limits[:20]


def _dedupe_citations(citations: Sequence[Citation]) -> List[Citation]:
    seen: Dict[str, Citation] = {}
    for citation in citations:
        seen.setdefault(citation.doc_id, citation)
    return list(seen.values())


# Pipeline bookkeeping, not financials. These are useful for debugging a run but
# would read as gibberish in a research report, where a reader reasonably
# assumes every number is a fact about the company.
_INTERNAL_METRIC_PREFIXES = ("channel_", "event_count_")
_INTERNAL_METRIC_NAMES = {
    "document_count",
    "max_reliability",
    "distinct_channels",
    "risk_channel_intensity",
    "max_drawdown_band",
}
_INTERNAL_METRIC_SUFFIXES = ("_band", "_count", "_ids", "_doc_ids")


def _is_reportable(metric: str) -> bool:
    name = str(metric).strip().lower()
    if not name:
        return False
    if name in _INTERNAL_METRIC_NAMES:
        return False
    if name.startswith(_INTERNAL_METRIC_PREFIXES):
        return False
    if name.endswith(_INTERNAL_METRIC_SUFFIXES):
        return False
    return True


def _as_stance(value: object, default: Stance) -> Stance:
    try:
        return Stance(str(value).lower())
    except (ValueError, AttributeError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["Synthesizer"]
