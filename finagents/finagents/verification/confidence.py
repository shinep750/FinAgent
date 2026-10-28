"""Confidence assessment.

A single number would be dishonest, so confidence is decomposed into four
independent components and then penalised for specific, named defects. The
decomposition is the point: a report that says "confidence 0.42" is useless,
whereas "agreement is high but coverage is partial and one claim's figures are
not in the source" tells a reader exactly what to distrust.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

from ..schemas import (
    AgentError,
    AgentFinding,
    ClaimVerification,
    Conflict,
    ConfidenceReport,
    Severity,
    Stance,
    Verdict,
)

logger = logging.getLogger(__name__)

# Component weights. Evidence quality and agreement dominate because a
# well-supported multi-lens agreement is what makes a conclusion trustworthy;
# coverage matters because a missing lens is invisible in the other components.
_WEIGHTS = {
    "evidence_quality": 0.32,
    "agreement": 0.28,
    "coverage": 0.20,
    "corroboration": 0.20,
}

_SEVERITY_PENALTY = {
    Severity.LOW: 0.03,
    Severity.MEDIUM: 0.07,
    Severity.HIGH: 0.14,
    Severity.CRITICAL: 0.24,
}


class ConfidenceAssessor:
    def assess(
        self,
        findings: Sequence[AgentFinding],
        verifications: Sequence[ClaimVerification],
        conflicts: Sequence[Conflict],
        expected_tasks: int = 0,
        errors: Sequence[AgentError] = (),
    ) -> ConfidenceReport:
        evidence_quality = self._evidence_quality(findings, verifications)
        agreement = self._agreement(findings)
        coverage = self._coverage(findings, expected_tasks)
        corroboration = self._corroboration(verifications)

        components = {
            "evidence_quality": evidence_quality,
            "agreement": agreement,
            "coverage": coverage,
            "corroboration": corroboration,
        }
        base = sum(components[name] * weight for name, weight in _WEIGHTS.items())

        penalties: List[str] = []
        unsupported = [v for v in verifications if v.verdict == Verdict.UNSUPPORTED]
        contradicted = [v for v in verifications if v.verdict == Verdict.CONTRADICTED]
        uncited = [v for v in verifications if not v.supporting_citations]

        if unsupported:
            penalties.append(
                f"{len(unsupported)} claim(s) had no supporting evidence and were excluded"
            )
        if contradicted:
            penalties.append(
                f"{len(contradicted)} claim(s) were contradicted by their own citation"
            )
        if uncited:
            penalties.append(
                f"{len(uncited)} verified claim(s) rest on a single weak source"
            )

        for conflict in conflicts:
            if conflict.resolved:
                continue
            penalty = _SEVERITY_PENALTY[conflict.severity]
            if penalty:
                penalties.append(
                    f"unresolved {conflict.conflict_type.value} conflict "
                    f"({conflict.severity.value}): -{penalty:.2f}"
                )
                base -= penalty

        for error in errors:
            penalties.append(f"agent {error.agent_name} failed: {error.error[:120]}")
            base -= 0.1

        stale = [
            v for v in verifications if v.source_quality and v.source_quality < 0.65
        ]
        if stale:
            penalties.append(
                f"{len(stale)} claim(s) rest on low-reliability sources (below 0.65)"
            )
            base -= 0.04

        overall = max(0.0, min(1.0, base))
        return ConfidenceReport(
            overall=round(overall, 4),
            evidence_quality=round(evidence_quality, 4),
            agreement=round(agreement, 4),
            coverage=round(coverage, 4),
            corroboration=round(corroboration, 4),
            penalties=penalties,
            components={k: round(v, 4) for k, v in components.items()},
        )

    # -- components -------------------------------------------------------

    def _evidence_quality(
        self,
        findings: Sequence[AgentFinding],
        verifications: Sequence[ClaimVerification],
    ) -> float:
        """How good is the underlying material: source reliability and retrieval rank."""
        qualities = [v.source_quality for v in verifications if v.source_quality]
        retrieval = [d.score for f in findings for d in f.evidence]
        if not qualities and not retrieval:
            return 0.2
        source = sum(qualities) / len(qualities) if qualities else 0.3
        ranked = sum(retrieval) / len(retrieval) if retrieval else 0.3
        return max(0.0, min(1.0, 0.6 * source + 0.4 * ranked))

    def _agreement(self, findings: Sequence[AgentFinding]) -> str:
        """Do the lenses converge? Weighted by each agent's own confidence.

        Neutral lenses are *not* dropped: a lens that abstains while two others
        agree is partial disagreement, and excluding it would let a report claim
        unanimity on the strength of the subset that bothered to answer.
        """
        directional = [
            f for f in findings if f.stance in (Stance.BULLISH, Stance.BEARISH)
        ]
        if not directional:
            return 0.0
        if len(directional) < 2:
            # A single directional lens cannot corroborate itself.
            return 0.35
        weights = [max(0.05, f.confidence) for f in findings]
        signed = [
            w
            * (
                1.0
                if f.stance == Stance.BULLISH
                else -1.0 if f.stance == Stance.BEARISH else 0.0
            )
            for f, w in zip(findings, weights)
        ]
        total = sum(weights)
        if total <= 0:
            return 0.5
        # Perfect agreement -> every lens signed the same way -> |mean| == 1.
        return max(0.0, min(1.0, abs(sum(signed)) / total))

    def _coverage(self, findings: Sequence[AgentFinding], expected_tasks: int) -> float:
        """Did every planned lens actually deliver a usable conclusion?"""
        if expected_tasks <= 0:
            return 1.0 if findings else 0.0
        with_claims = sum(1 for f in findings if f.claims)
        with_summary = sum(1 for f in findings if f.summary and f.summary.strip())
        claim_ratio = with_claims / expected_tasks
        summary_ratio = with_summary / expected_tasks
        return max(0.0, min(1.0, 0.6 * claim_ratio + 0.4 * summary_ratio))

    def _corroboration(self, verifications: Sequence[ClaimVerification]) -> float:
        """How much of the analysis rests on more than a single source."""
        if not verifications:
            return 0.0
        multi = sum(1 for v in verifications if len(v.supporting_citations) > 1)
        supported = sum(
            1
            for v in verifications
            if v.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)
        )
        multi_ratio = multi / len(verifications)
        support_ratio = supported / len(verifications)
        return max(0.0, min(1.0, 0.5 * multi_ratio + 0.5 * support_ratio))


__all__ = ["ConfidenceAssessor"]
