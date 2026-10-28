"""Claim verification.

Each claim is checked twice, and the two checks are deliberately independent:

* **Deterministic checks** -- do the cited documents exist, does the claim's
  vocabulary actually appear in them, and do the numbers in the claim match the
  numbers in the cited text. These are reproducible and testable, and they catch
  the failure mode that matters most in a financial report: a confident,
  plausible number that is not in the source.
* **Model adjudication** -- does the cited text *entail* the claim. Lexical
  overlap cannot tell "revenue rose 114%" from "revenue rose 5%"; only reading
  comprehension can.

The final verdict is the model's, but it is bounded by the deterministic
evidence. A model that calls a claim supported when its numbers do not appear
in any cited document does not get to overrule arithmetic.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..llm.base import LLMClient, build_prompt
from ..prompts import SYSTEM_VERIFIER, role_header
from ..retrieval.channels import tokenize
from ..schemas import (
    AgentFinding,
    Citation,
    Claim,
    ClaimVerification,
    Document,
    Settings,
    Verdict,
)

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?%?")

# Support thresholds, tuned so a claim quoting its source verbatim is solidly
# supported while a claim merely on the same topic is not.
_SUPPORTED_THRESHOLD = 0.34
_PARTIAL_THRESHOLD = 0.16


class ClaimVerifier:
    def __init__(self, llm: LLMClient, settings: Optional[Settings] = None) -> None:
        self.llm = llm
        self.settings = settings or Settings()

    # -- deterministic checks --------------------------------------------

    def lexical_support(self, claim: Claim, documents: Dict[str, Document]) -> float:
        """Fraction of the claim's content words present in the cited text."""
        claim_tokens = set(tokenize(claim.text))
        if not claim_tokens:
            return 0.0
        if not claim.citations:
            return 0.0
        blob = " ".join(
            documents[cite.doc_id].text
            for cite in claim.citations
            if cite.doc_id in documents
        )
        if not blob:
            return 0.0
        blob_tokens = set(tokenize(blob))
        return len(claim_tokens & blob_tokens) / len(claim_tokens)

    def numeric_check(
        self, claim: Claim, documents: Dict[str, Document]
    ) -> Tuple[bool, str]:
        """Verify every number in the claim appears in a cited document.

        A claim asserting a figure its own citation does not contain is the
        single most damaging error a financial report can make, so it is checked
        explicitly rather than left to fuzzy matching. Small integers (1-4) are
        exempt because they appear as list markers, quarters and section numbers.
        """
        claim_numbers = {
            n.replace(",", "")
            for n in _NUMBER.findall(claim.text)
            if len(n.replace(",", "")) > 1
        }
        if not claim_numbers:
            return True, "no numeric assertions in claim"
        cited = " ".join(
            documents[cite.doc_id].text
            for cite in claim.citations
            if cite.doc_id in documents
        )
        cited_numbers = {n.replace(",", "") for n in _NUMBER.findall(cited)}
        missing = sorted(claim_numbers - cited_numbers)
        if not missing:
            return (
                True,
                f"all {len(claim_numbers)} numeric value(s) appear in the cited text",
            )
        return (
            False,
            f"numeric value(s) {', '.join(missing[:4])} do not appear in any cited document",
        )

    def citation_integrity(
        self, claim: Claim, documents: Dict[str, Document]
    ) -> Tuple[bool, str]:
        if not claim.citations:
            return False, "claim carries no citation"
        missing = [c.doc_id for c in claim.citations if c.doc_id not in documents]
        if missing:
            return (
                False,
                f"cites document(s) absent from the evidence set: {', '.join(missing[:3])}",
            )
        return (
            True,
            f"all {len(claim.citations)} citation(s) resolve to retrieved evidence",
        )

    # -- verification -----------------------------------------------------

    def verify(self, findings: Sequence[AgentFinding]) -> List[ClaimVerification]:
        documents = {
            d.document.doc_id: d.document for f in findings for d in f.evidence
        }
        # Claim IDs are content hashes, so two agents that independently reach the
        # same conclusion emit the same ID. That is a useful signal (it means the
        # statement was not agent-specific), but it must be verified once or
        # corroboration and confidence are double-counted.
        claims: List[Tuple[Claim, AgentFinding]] = []
        seen: Dict[str, str] = {}
        duplicates = 0
        for finding in findings:
            for claim in finding.claims:
                first = seen.get(claim.claim_id)
                if first is None:
                    seen[claim.claim_id] = finding.agent_name
                    claims.append((claim, finding))
                elif first != finding.agent_name:
                    duplicates += 1
                    logger.info(
                        "claim %s reported by both %s and %s; verifying once",
                        claim.claim_id,
                        first,
                        finding.agent_name,
                    )
                else:
                    duplicates += 1
        if duplicates:
            logger.info("%d duplicate claim(s) collapsed", duplicates)
        if not claims:
            return []

        deterministic = {
            claim.claim_id: self._deterministic(claim, documents) for claim, _ in claims
        }
        adjudicated = self._adjudicate(claims, documents)

        results: List[ClaimVerification] = []
        for claim, _finding in claims:
            checks = deterministic[claim.claim_id]
            model_result = adjudicated.get(claim.claim_id, {})
            results.append(self._combine(claim, checks, model_result, documents))
        return results

    def _deterministic(
        self, claim: Claim, documents: Dict[str, Document]
    ) -> Dict[str, object]:
        integrity_ok, integrity_note = self.citation_integrity(claim, documents)
        numbers_ok, numbers_note = self.numeric_check(claim, documents)
        overlap = self.lexical_support(claim, documents)
        quality = sum(
            documents[c.doc_id].reliability
            for c in claim.citations
            if c.doc_id in documents
        ) / max(1, len([c for c in claim.citations if c.doc_id in documents]))
        return {
            "integrity_ok": integrity_ok,
            "integrity_note": integrity_note,
            "numbers_ok": numbers_ok,
            "numbers_note": numbers_note,
            "overlap": overlap,
            "source_quality": quality,
        }

    def _adjudicate(
        self,
        claims: Sequence[Tuple[Claim, AgentFinding]],
        documents: Dict[str, Document],
    ) -> Dict[str, Dict[str, object]]:
        """Ask the model to judge entailment. Failure here is non-fatal."""
        lines: List[str] = []
        for claim, _finding in claims:
            cites = ",".join(c.doc_id for c in claim.citations) or "none"
            lines.append(
                f"CLAIM {claim.claim_id} | stance={claim.stance.value} | conf={claim.confidence:.2f} | cites={cites}\n"
                f"{claim.text}"
            )
        context_parts = ["\n\n".join(lines), "EVIDENCE"]
        for claim, _ in claims:
            for cite in claim.citations:
                doc = documents.get(cite.doc_id)
                if doc is None:
                    continue
                context_parts.append(
                    f"DOC {doc.doc_id} | {doc.title} | source={doc.source} | "
                    f"date={doc.published_at.date().isoformat() if doc.published_at else 'undated'} | "
                    f"type={doc.doc_type.value} | reliability={doc.reliability:.2f} | score=0.000\n"
                    f"{doc.text[:1600]}"
                )
        prompt = build_prompt(
            system=SYSTEM_VERIFIER,
            instruction=(
                f"{role_header('verifier')}\n"
                "For each CLAIM, decide whether the cited evidence actually supports it. "
                "Use verdict 'supported' only if a cited document states the claim. Use "
                "'partially_supported' if it supports the direction but not the specifics. "
                "Use 'contradicted' if a cited document states something incompatible. "
                "Use 'unsupported' if no cited document addresses it. Return one entry per claim_id."
            ),
            context="\n\n".join(context_parts),
            output_schema=(
                '{"verifications": [{"claim_id": "string", "verdict": '
                '"supported|partially_supported|unsupported|contradicted|unverifiable", '
                '"support_score": 0.0, "rationale": "string", '
                '"supporting_citations": [{"doc_id": "string", "title": "string", '
                '"source": "string", "quote": "string"}], '
                '"contradicting_citations": [{"doc_id": "string", "title": "string", '
                '"source": "string", "quote": "string"}]}]}'
            ),
        )
        try:
            payload = self.llm.generate_json(
                prompt, SYSTEM_VERIFIER, temperature=self.settings.llm_temperature
            )
        except Exception as exc:  # noqa: BLE001 - verification must not abort the run
            logger.warning(
                "model adjudication unavailable (%s); relying on deterministic checks",
                exc,
            )
            return {}
        out: Dict[str, Dict[str, object]] = {}
        for entry in (payload or {}).get("verifications", []) or []:
            if not isinstance(entry, dict) or not entry.get("claim_id"):
                continue
            try:
                verdict = Verdict(str(entry.get("verdict", "unverifiable")).lower())
            except ValueError:
                verdict = Verdict.UNVERIFIABLE
            out[str(entry["claim_id"])] = {
                "verdict": verdict,
                "support_score": _as_float(entry.get("support_score"), 0.0),
                "rationale": str(entry.get("rationale", ""))[:600],
                "supporting": _as_citations(
                    entry.get("supporting_citations"), documents
                ),
                "contradicting": _as_citations(
                    entry.get("contradicting_citations"), documents
                ),
            }
        return out

    def _combine(
        self,
        claim: Claim,
        checks: Dict[str, object],
        model: Dict[str, object],
        documents: Dict[str, Document],
    ) -> ClaimVerification:
        overlap = float(checks["overlap"])  # type: ignore[arg-type]
        integrity_ok = bool(checks["integrity_ok"])
        numbers_ok = bool(checks["numbers_ok"])
        quality = float(checks["source_quality"])  # type: ignore[arg-type]

        model_verdict = model.get("verdict")
        if model_verdict is None:
            verdict = (
                _verdict_from_overlap(overlap) if integrity_ok else Verdict.UNSUPPORTED
            )
            rationale = (
                f"Deterministic check: {int(overlap * 100)}% of the claim's content words "
                f"appear in the cited text; {checks['numbers_note']}."
            )
        else:
            verdict = model_verdict
            rationale = str(model.get("rationale") or "")
            rationale += f" Lexical overlap with cited text: {int(overlap * 100)}%."
            rationale += f" {checks['numbers_note']}."

        # Arithmetic and citation integrity bound the model. A figure that is
        # absent from the source cannot be "supported", whatever the model says.
        if not numbers_ok and verdict in (
            Verdict.SUPPORTED,
            Verdict.PARTIALLY_SUPPORTED,
        ):
            verdict = (
                Verdict.CONTRADICTED
                if overlap >= _SUPPORTED_THRESHOLD
                else Verdict.UNSUPPORTED
            )
            rationale += " Downgraded: the claim's figures are not present in the cited evidence."
        if not integrity_ok:
            verdict = Verdict.UNSUPPORTED
            rationale += f" {checks['integrity_note']}."
        elif overlap < _PARTIAL_THRESHOLD and verdict == Verdict.SUPPORTED:
            verdict = Verdict.PARTIALLY_SUPPORTED
            rationale += (
                " Downgraded: the claim shares little vocabulary with the cited text."
            )

        support = _as_float(model.get("support_score"), 0.0)
        blended = (
            0.5 * support
            + 0.3 * min(1.0, overlap / _SUPPORTED_THRESHOLD)
            + 0.2 * quality
        )
        if verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED):
            blended = min(blended, 0.25)
        elif verdict == Verdict.PARTIALLY_SUPPORTED:
            blended = min(blended, 0.65)
        elif verdict == Verdict.SUPPORTED:
            blended = max(blended, 0.6)

        return ClaimVerification(
            claim_id=claim.claim_id,
            verdict=verdict,
            support_score=round(min(1.0, blended), 4),
            rationale=rationale.strip()[:1200],
            supporting_citations=list(model.get("supporting") or claim.citations[:1]),
            contradicting_citations=list(model.get("contradicting") or []),
            arithmetic_check=str(checks["numbers_note"]),
            source_quality=round(quality, 4),
        )


def _verdict_from_overlap(overlap: float) -> Verdict:
    if overlap >= _SUPPORTED_THRESHOLD:
        return Verdict.SUPPORTED
    if overlap >= _PARTIAL_THRESHOLD:
        return Verdict.PARTIALLY_SUPPORTED
    return Verdict.UNSUPPORTED


def _as_citations(value: object, documents: Dict[str, Document]) -> List[Citation]:
    if not isinstance(value, list):
        return []
    out: List[Citation] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id", ""))
        doc = documents.get(doc_id)
        if doc is None:
            continue
        out.append(
            Citation(
                doc_id=doc.doc_id,
                title=doc.title,
                source=doc.source,
                quote=str(item.get("quote", ""))[:400],
                url=doc.url,
            )
        )
    return out


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["ClaimVerifier"]
