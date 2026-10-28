"""Conflict detection across agents.

Independent agents disagreeing is the expected case, not an error: a
fundamentals agent calling a stock bullish while a risk agent calls the same
downside exposure elevated is a real, reportable tension. The job here is to
locate disagreements precisely, type them, judge how much they matter, and where
possible resolve them by preferring the better-evidenced side.

The important discipline is *not* to average stances into mush. An unresolved
directional conflict must survive into the report as a conflict.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from ..schemas import (
    AgentFinding,
    Claim,
    ClaimVerification,
    Conflict,
    ConflictType,
    Document,
    Severity,
    Stance,
)

logger = logging.getLogger(__name__)

_OPPOSITE = {
    (Stance.BULLISH, Stance.BEARISH),
    (Stance.BEARISH, Stance.BULLISH),
}


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _direction(stance: Stance) -> int:
    if stance == Stance.BULLISH:
        return 1
    if stance == Stance.BEARISH:
        return -1
    return 0


class ConflictDetector:
    def __init__(self, numeric_tolerance: float = 0.02) -> None:
        # Relative tolerance for "two agents report the same figure differently".
        self.numeric_tolerance = numeric_tolerance

    def detect(
        self,
        findings: Sequence[AgentFinding],
        verifications: Optional[Sequence[ClaimVerification]] = None,
        documents: Optional[Dict[str, Document]] = None,
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        conflicts.extend(self._directional_conflicts(findings))
        conflicts.extend(self._numeric_conflicts(findings))
        conflicts.extend(self._temporal_conflicts(findings, documents or {}))
        return conflicts

    # -- directional ------------------------------------------------------

    def _directional_conflicts(
        self, findings: Sequence[AgentFinding]
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []
        for left, right in itertools.combinations(findings, 2):
            pair = (left.stance, right.stance)
            if (pair[0], pair[1]) not in _OPPOSITE:
                continue
            conviction = min(left.confidence, right.confidence)
            gap = abs(_direction(left.stance) - _direction(right.stance))
            severity = self._severity(gap, conviction, left, right)
            names = [left.agent_name, right.agent_name]
            description = (
                f"{left.agent_name} ({left.perspective.value}) is {left.stance.value} at "
                f"confidence {left.confidence:.2f} while {right.agent_name} "
                f"({right.perspective.value}) is {right.stance.value} at "
                f"{right.confidence:.2f}."
            )
            conflicts.append(
                Conflict(
                    conflict_id=_id("cf", "dir", *sorted(names), left.stance.value),
                    conflict_type=ConflictType.DIRECTION,
                    description=description,
                    claim_ids=sorted(
                        [c.claim_id for c in left.claims]
                        + [c.claim_id for c in right.claims]
                    )[:12],
                    agent_names=names,
                    severity=severity,
                    resolution=(
                        "Left unresolved: the perspectives genuinely disagree, which is itself "
                        "a finding. The synthesis reports both sides rather than averaging them."
                    ),
                    resolved=False,
                )
            )
        return conflicts

    def _severity(
        self, gap: int, conviction: float, left: AgentFinding, right: AgentFinding
    ) -> Severity:
        if gap >= 2 and conviction >= 0.6:
            return Severity.CRITICAL
        if gap >= 2:
            return Severity.HIGH
        if conviction >= 0.6:
            return Severity.MEDIUM
        # A low-conviction disagreement is weak evidence of a real conflict.
        return Severity.LOW

    # -- numeric ----------------------------------------------------------

    def _numeric_conflicts(self, findings: Sequence[AgentFinding]) -> List[Conflict]:
        """Find the same named metric reported with materially different values.

        Deduplication matters: many agents quote the same 10-K, so identical
        figures must not be reported as agreeing, and a genuine mismatch must be
        reported once, not once per agent pair.
        """
        by_metric: Dict[str, List[Tuple[str, str, float]]] = {}
        for finding in findings:
            for claim in finding.claims:
                for metric, value in claim.metrics.items():
                    if not isinstance(value, (int, float)):
                        continue
                    by_metric.setdefault(metric, []).append(
                        (finding.agent_name, claim.claim_id, float(value))
                    )
            for metric, value in finding.key_metrics.items():
                if metric.startswith(
                    ("channel_", "event_count_", "document_count", "sessions")
                ):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                by_metric.setdefault(metric, []).append(
                    (finding.agent_name, finding.task_id, float(value))
                )

        conflicts: List[Conflict] = []
        for metric, entries in by_metric.items():
            distinct_agents = {agent for agent, _, _ in entries}
            if len(distinct_agents) < 2:
                continue
            values = [v for _, _, v in entries]
            low, high = min(values), max(values)
            if low == 0 and high == 0:
                continue
            spread = abs(high - low) / max(abs(low), abs(high), 1e-9)
            if spread <= self.numeric_tolerance:
                continue
            involved = sorted({agent for agent, _, _ in entries})
            conflicts.append(
                Conflict(
                    conflict_id=_id("cf", "num", metric, *involved),
                    conflict_type=ConflictType.FACTUAL,
                    description=(
                        f"Agents disagree on '{metric}': {', '.join(f'{a}={v:g}' for a, _, v in entries)} "
                        f"({spread * 100:.1f}% spread)."
                    ),
                    claim_ids=sorted({cid for _, cid, _ in entries}),
                    agent_names=involved,
                    severity=Severity.HIGH if spread > 0.1 else Severity.MEDIUM,
                    resolution=(
                        "Unresolved: the figures are not reconcilable from the retrieved evidence "
                        "alone. The higher-reliability sources should be preferred."
                    ),
                    resolved=False,
                )
            )
        return conflicts

    # -- temporal ---------------------------------------------------------

    def _temporal_conflicts(
        self, findings: Sequence[AgentFinding], documents: Dict[str, Document]
    ) -> List[Conflict]:
        """Flag near-identical claims drawn from sources far apart in time.

        Two agents can both be right about different periods. When a bullish and
        a bearish claim rest on sources 18 months apart, that is a scope problem
        worth naming, not a contradiction.
        """
        dates: List[Tuple[AgentFinding, Claim, str]] = []
        for finding in findings:
            for claim in finding.claims:
                for cite in claim.citations:
                    doc = documents.get(cite.doc_id)
                    if doc is not None and doc.published_at is not None:
                        dates.append(
                            (finding, claim, doc.published_at.date().isoformat())
                        )
        if len(dates) < 2:
            return []

        dates.sort(key=lambda item: item[2])
        oldest_finding, oldest_claim, oldest_date = dates[0]
        newest_finding, newest_claim, newest_date = dates[-1]
        if oldest_finding.agent_name == newest_finding.agent_name:
            return []

        gap_days = _days_between(oldest_date, newest_date)
        if gap_days < 365:
            return []
        if (oldest_finding.stance, newest_finding.stance) not in _OPPOSITE:
            return []
        return [
            Conflict(
                conflict_id=_id(
                    "cf", "time", oldest_finding.agent_name, newest_finding.agent_name
                ),
                conflict_type=ConflictType.TEMPORAL,
                description=(
                    f"{newest_finding.agent_name} ({newest_finding.stance.value}, "
                    f"{newest_date}) and {oldest_finding.agent_name} ({oldest_finding.stance.value}, "
                    f"{oldest_date}) reach opposite conclusions {gap_days} days apart. This is more "
                    "likely a period difference than a factual contradiction."
                ),
                claim_ids=[oldest_claim.claim_id, newest_claim.claim_id],
                agent_names=[oldest_finding.agent_name, newest_finding.agent_name],
                severity=Severity.MEDIUM,
                resolution=(
                    "Scoped: conclusions are period-specific. The synthesis presents the more "
                    "recent evidence as current and the older as historical context."
                ),
                resolved=True,
            )
        ]


def _days_between(earlier: str, later: str) -> int:
    from datetime import date

    try:
        return (date.fromisoformat(later) - date.fromisoformat(earlier)).days
    except ValueError:
        return 0


__all__ = ["ConflictDetector"]
