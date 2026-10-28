"""The verification layer is the trust boundary.

These tests are adversarial on purpose: claims are deliberately mis-cited,
contradicted and duplicated to confirm the pipeline refuses to launder them into
the report. A financial system that only tests the happy path here is a system
that will confidently publish a wrong number.
"""

from __future__ import annotations

import pytest

from finagents.schemas import (
    AgentFinding,
    Citation,
    Claim,
    ConflictType,
    ConfidenceReport,
    Document,
    EvidenceType,
    Perspective,
    ScoredDocument,
    Stance,
    Verdict,
)
from finagents.verification import (
    ClaimVerifier,
    ConfidenceAssessor,
    ConflictDetector,
    Synthesizer,
)

FILING_TEXT = (
    "For fiscal year 2025 total revenue reached $130.5 billion, an increase of 114 "
    "percent from the prior year. Gross margin was 75.0% for the fiscal year and "
    "73.5% in the fourth quarter. Two customers accounted for 46% of total revenue. "
    "We recorded a $4.5 billion charge related to inventory and purchase commitments."
)


FILING = Document(
    doc_id="test-10k",
    title="Test FY2025 10-K",
    text=FILING_TEXT,
    doc_type=EvidenceType.FILING,
    source="SEC EDGAR",
    reliability=0.95,
    # Real provenance, not a synthetic flag: the product has no synthetic
    # corpus, and a fixture that models one teaches the wrong invariant.
    metadata={"tickers": ["NVDA"], "form": "10-K"},
)


@pytest.fixture()
def filing() -> Document:
    return FILING


def _finding(claim_text: str, cite_ids, **kw) -> AgentFinding:
    """Build a finding shaped like the ones the agents really produce.

    Evidence matters: the verifier resolves citations against
    `finding.evidence`, so a finding with citations but no evidence is a shape
    the pipeline never emits, and tests written against it prove nothing.
    """
    doc_id = cite_ids[0] if cite_ids else "test-10k"
    evidence = kw.pop("evidence", None)
    if evidence is None:
        evidence = [ScoredDocument(document=FILING, score=0.9)]
    claim = Claim(
        text=claim_text,
        stance=kw.pop("stance", Stance.BULLISH),
        confidence=kw.pop("confidence", 0.8),
        citations=[
            Citation(doc_id=doc_id, title="Test FY2025 10-K", source="SEC EDGAR")
            for doc_id in cite_ids
        ],
    )
    return AgentFinding(
        task_id=kw.pop("task_id", "t1"),
        agent_name=kw.pop("agent_name", "financial_analysis"),
        perspective=kw.pop("perspective", Perspective.FUNDAMENTAL),
        summary=kw.pop("summary", "summary"),
        stance=kw.pop("finding_stance", Stance.BULLISH),
        confidence=kw.pop("finding_confidence", 0.8),
        claims=[claim],
        evidence=evidence,
        **kw,
    )


@pytest.fixture()
def verifier(llm, settings) -> ClaimVerifier:
    return ClaimVerifier(llm, settings)


class TestDeterministicChecks:
    def test_numbers_present_in_source(self, verifier, filing):
        claim = _finding(
            "Total revenue reached $130.5 billion, an increase of 114 percent.",
            ["test-10k"],
        ).claims[0]
        ok, note = verifier.numeric_check(claim, {"test-10k": filing})
        assert ok, note

    def test_number_absent_from_source_is_caught(self, verifier, filing):
        """The failure mode that matters: a confident figure not in the filing."""
        claim = _finding("Total revenue reached $999.9 billion.", ["test-10k"]).claims[
            0
        ]
        ok, note = verifier.numeric_check(claim, {"test-10k": filing})
        assert not ok
        assert "999.9" in note

    def test_lexical_support_high_for_verbatim(self, verifier, filing):
        claim = _finding(
            "Gross margin was 75.0% for the fiscal year.", ["test-10k"]
        ).claims[0]
        assert verifier.lexical_support(claim, {"test-10k": filing}) > 0.3

    def test_uncited_claim_has_no_support(self, verifier, filing):
        claim = _finding("Gross margin was 75.0%.", []).claims[0]
        assert verifier.lexical_support(claim, {"test-10k": filing}) == 0.0

    def test_citation_integrity_flags_missing_document(self, verifier, filing):
        claim = _finding("Anything at all here.", ["does-not-exist"]).claims[0]
        ok, note = verifier.citation_integrity(claim, {"test-10k": filing})
        assert not ok
        assert "does-not-exist" in note


class TestVerificationOutcomes:
    def test_verbatim_claim_is_supported(self, verifier, filing):
        result = verifier.verify(
            [
                _finding(
                    "Total revenue reached $130.5 billion, an increase of 114 percent.",
                    ["test-10k"],
                )
            ]
        )[0]
        assert result.verdict == Verdict.SUPPORTED
        assert result.support_score > 0.5

    def test_fabricated_number_is_not_supported(self, verifier, filing):
        result = verifier.verify(
            [_finding("Total revenue reached $999.9 billion this year.", ["test-10k"])]
        )[0]
        assert result.verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)
        assert result.support_score <= 0.25

    def test_uncited_claim_is_rejected(self, verifier, filing):
        result = verifier.verify([_finding("Margin will double next year.", [])])[0]
        assert result.verdict == Verdict.UNSUPPORTED
        assert "no citation" in result.rationale.lower()

    def test_unknown_document_is_rejected(self, verifier, filing):
        result = verifier.verify(
            [_finding("A claim citing a document we never fetched.", ["ghost"])]
        )[0]
        assert result.verdict == Verdict.UNSUPPORTED

    def test_no_claims_returns_nothing(self, verifier):
        assert (
            verifier.verify(
                [
                    _finding("x", ["test-10k"]).__class__(
                        task_id="t",
                        agent_name="a",
                        perspective=Perspective.NEWS,
                        summary="s",
                        stance=Stance.NEUTRAL,
                        confidence=0.5,
                    )
                ]
            )
            == []
        )

    def test_duplicate_claim_ids_verified_once(self, verifier, filing):
        """Two agents reaching one conclusion must not double-count as two."""
        a = _finding("Gross margin was 75.0% for the fiscal year.", ["test-10k"])
        b = _finding(
            "Gross margin was 75.0% for the fiscal year.",
            ["test-10k"],
            task_id="t2",
            agent_name="risk_analysis",
            perspective=Perspective.RISK,
        )
        results = verifier.verify([a, b])
        assert len(results) == 1

    def test_every_verification_cites_a_document(self, verifier, filing):
        for v in verifier.verify(
            [_finding("Gross margin was 75.0% for the fiscal year.", ["test-10k"])]
        ):
            assert v.arithmetic_check


class TestConflictDetection:
    @pytest.fixture()
    def detector(self) -> ConflictDetector:
        return ConflictDetector()

    def test_bull_vs_bear_is_a_direction_conflict(self, detector):
        bull = _finding("Revenue grew 114 percent.", ["test-10k"])
        bear = _finding(
            "Customer concentration is a material risk to results.",
            ["test-10k"],
            task_id="t2",
            agent_name="risk_analysis",
            perspective=Perspective.RISK,
            stance=Stance.BEARISH,
            finding_stance=Stance.BEARISH,
        )
        conflicts = detector.detect([bull, bear], [], {})
        assert any(c.conflict_type == ConflictType.DIRECTION for c in conflicts)

    def test_agreeing_agents_produce_no_direction_conflict(self, detector):
        a = _finding("Revenue grew 114 percent.", ["test-10k"])
        b = _finding(
            "Revenue grew 114 percent.",
            ["test-10k"],
            task_id="t2",
            agent_name="news_analysis",
            perspective=Perspective.NEWS,
        )
        assert detector.detect([a, b], [], []) == []

    def test_conflicts_are_resolved_or_flagged(self, detector):
        bull = _finding("Revenue grew 114 percent.", ["test-10k"])
        bear = _finding(
            "Customer concentration is a material risk.",
            ["test-10k"],
            task_id="t2",
            agent_name="risk_analysis",
            perspective=Perspective.RISK,
            stance=Stance.BEARISH,
            finding_stance=Stance.BEARISH,
        )
        for conflict in detector.detect([bull, bear], [], {}):
            assert (
                conflict.resolution
            ), "a detected conflict must state how it was handled"


class TestConfidence:
    @pytest.fixture()
    def assessor(self) -> ConfidenceAssessor:
        return ConfidenceAssessor()

    def test_no_evidence_scores_low(self, assessor):
        report = assessor.assess([], [], [], expected_tasks=1)
        assert report.overall < 0.3

    def test_unanimous_strong_evidence_scores_high(self, assessor):
        findings = [
            _finding("Revenue grew 114 percent.", ["test-10k"], task_id="a"),
            _finding(
                "Gross margin was 75.0%.",
                ["test-10k"],
                task_id="b",
                agent_name="news_analysis",
                perspective=Perspective.NEWS,
            ),
        ]
        verifier = ClaimVerifier.__new__(ClaimVerifier)
        from finagents.schemas import ClaimVerification

        verifications = [
            ClaimVerification(
                claim_id=c.claim_id,
                verdict=Verdict.SUPPORTED,
                support_score=0.9,
                supporting_citations=c.citations,
                source_quality=0.95,
            )
            for f in findings
            for c in f.claims
        ]
        report = assessor.assess(findings, verifications, [], expected_tasks=2)
        assert report.overall > 0.5
        assert report.evidence_quality > 0.7

    def test_neutral_lens_lowers_agreement(self, assessor):
        """A lens that abstains is not evidence of unanimity."""
        base = [
            _finding("Revenue grew.", ["test-10k"], task_id="a"),
            _finding(
                "Margin held.",
                ["test-10k"],
                task_id="b",
                agent_name="news_analysis",
                perspective=Perspective.NEWS,
            ),
        ]
        with_neutral = base + [
            _finding(
                "Risks are balanced.",
                ["test-10k"],
                task_id="c",
                agent_name="risk_analysis",
                perspective=Perspective.RISK,
                finding_stance=Stance.NEUTRAL,
            )
        ]
        assert (
            assessor.assess(with_neutral, [], []).agreement
            < assessor.assess(base, [], []).agreement
        )

    def test_unsupported_claims_are_named_as_penalties(self, assessor):
        from finagents.schemas import ClaimVerification

        bad = ClaimVerification(
            claim_id="c1",
            verdict=Verdict.UNSUPPORTED,
            support_score=0.1,
            rationale="no evidence",
            supporting_citations=[],
        )
        report = assessor.assess(
            [_finding("x", ["test-10k"])], [bad], [], expected_tasks=1
        )
        assert any("no supporting evidence" in p for p in report.penalties)

    def test_agent_failure_penalises_confidence(self, assessor):
        from finagents.schemas import AgentError

        finding = _finding("Revenue grew.", ["test-10k"])
        error = AgentError(task_id="t2", agent_name="technical_analysis", error="boom")
        with_error = assessor.assess(
            [finding], [], [], expected_tasks=2, errors=[error]
        )
        assert (
            with_error.overall
            < assessor.assess([finding], [], [], expected_tasks=1).overall
        )
        assert any("technical_analysis" in p for p in with_error.penalties)

    def test_score_is_bounded(self, assessor):
        report = assessor.assess([], [], [], expected_tasks=0)
        assert 0.0 <= report.overall <= 1.0


class TestSynthesis:
    """Synthesis must never crash, and never manufacture a consensus."""

    def _synthesizer(self, llm, settings):
        return Synthesizer(llm, settings)

    @staticmethod
    def _conflict(conflict_type=ConflictType.DIRECTION, resolved=False):
        from finagents.schemas import Conflict

        return Conflict(
            conflict_id="cf-1",
            conflict_type=conflict_type,
            description="agents disagree on direction",
            resolved=resolved,
        )

    def test_an_unresolved_directional_conflict_yields_mixed(self, llm, settings):
        """Regression: this path raised `NameError: ConflictType` in production.

        Every conflict enters the list comprehension, so the moment any conflict
        exists the synthesiser crashed before producing a report - and conflicts
        are exactly what a well-functioning verification layer finds.
        """
        synth = self._synthesizer(llm, settings)
        finding = _finding("Revenue grew.", ["test-10k"])
        stance, note = synth._overall_stance(
            [finding], [self._conflict()], ConfidenceReport(overall=0.6)
        )
        assert stance is Stance.MIXED
        assert "unresolved" in note

    def test_a_resolved_conflict_does_not_block_a_stance(self, llm, settings):
        synth = self._synthesizer(llm, settings)
        finding = _finding("Revenue grew.", ["test-10k"])
        stance, _ = synth._overall_stance(
            [finding], [self._conflict(resolved=True)], ConfidenceReport(overall=0.6)
        )
        assert stance is Stance.BULLISH

    def test_non_directional_conflicts_are_ignored(self, llm, settings):
        synth = self._synthesizer(llm, settings)
        finding = _finding("Revenue grew.", ["test-10k"])
        stance, _ = synth._overall_stance(
            [finding],
            [self._conflict(conflict_type=ConflictType.TEMPORAL)],
            ConfidenceReport(overall=0.6),
        )
        assert stance is Stance.BULLISH

    def test_no_directional_finding_is_neutral(self, llm, settings):
        synth = self._synthesizer(llm, settings)
        finding = _finding(
            "Nothing to report.", ["test-10k"], finding_stance=Stance.NEUTRAL
        )
        stance, note = synth._overall_stance(
            [finding], [self._conflict()], ConfidenceReport(overall=0.5)
        )
        assert stance is Stance.NEUTRAL
        assert "no agent" in note
