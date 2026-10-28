"""Prompt construction.

All agents render evidence through :func:`format_evidence`, so the context a
real model sees and the context the test double parses are byte-identical in
structure. That keeps the tested path an honest exercise of the pipeline rather
than a separate mock world.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .schemas import Perspective, ScoredDocument

ROLE_MARKER = "ROLE:"

SYSTEM_PLANNER = (
    "You are the Planner of an explainable multi-agent financial research system. "
    "You analyse user intent, decompose the question into tasks, decide which "
    "financial perspectives matter, and assign each task to a specialised agent. "
    "You never invent facts and you always respond with valid JSON."
)

SYSTEM_ANALYST = (
    "You are a specialised financial analyst. You reason ONLY from the evidence "
    "provided. Every claim you make must cite the id of a document in that "
    "evidence. If the evidence does not support a claim, you must not make it. "
    "You always respond with valid JSON."
)

SYSTEM_VERIFIER = (
    "You are an adversarial verifier. Your job is to check whether each claim is "
    "actually supported by the cited evidence, to flag contradictions between "
    "analysts, and to keep the analysis honest. You always respond with valid JSON."
)

SYSTEM_SYNTHESIZER = (
    "You synthesise verified multi-agent financial analysis into a single coherent, "
    "explainable research report. You may only use claims that passed verification, "
    "you must surface disagreements rather than paper over them, and you always "
    "respond with valid JSON."
)

ANALYST_SYSTEM = {
    Perspective.FUNDAMENTAL: (
        SYSTEM_ANALYST
        + " Your lens is FUNDAMENTAL analysis: revenue, margins, earnings quality, "
        "cash flow, balance sheet strength, valuation, and guidance."
    ),
    Perspective.NEWS: (
        SYSTEM_ANALYST
        + " Your lens is NEWS analysis: material events, catalysts, sentiment, "
        "management commentary, and what is new relative to prior expectations."
    ),
    Perspective.TECHNICAL: (
        SYSTEM_ANALYST
        + " Your lens is TECHNICAL analysis: price trend, momentum, moving averages, "
        "volume behaviour, volatility, and support/resistance structure."
    ),
    Perspective.RISK: (
        SYSTEM_ANALYST
        + " Your lens is RISK analysis: downside scenarios, concentration, regulatory "
        "and litigation exposure, valuation fragility, and volatility."
    ),
}


def role_header(role: str) -> str:
    """First line of every prompt; a client can dispatch on this."""
    return f"{ROLE_MARKER} {role}"


def format_evidence(documents: Sequence[ScoredDocument], max_chars: int = 1400) -> str:
    """Render scored documents as a stable, parseable evidence block."""
    if not documents:
        return "(no evidence retrieved)"
    blocks: List[str] = []
    for scored in documents:
        doc = scored.document
        date_str = (
            doc.published_at.date().isoformat() if doc.published_at else "undated"
        )
        text = doc.text.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(
            f"DOC {doc.doc_id} | {doc.title} | source={doc.source} | date={date_str} | "
            f"type={doc.doc_type.value} | reliability={doc.reliability:.2f} | score={scored.score:.3f}\n"
            f"{text}"
        )
    return "\n\n".join(blocks)


def format_findings_context(findings: Iterable, max_chars: int = 1200) -> str:
    """Render agent findings for the synthesizer."""
    blocks: List[str] = []
    for finding in findings:
        claims = "\n".join(
            f"  - [{c.claim_id}] {c.text} (stance={c.stance.value}, conf={c.confidence:.2f}, "
            f"cites={','.join(c.doc_id for c in c.citations) or 'none'})"
            for c in finding.claims
        )
        body = finding.summary.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(
            f"FINDING {finding.task_id} | agent={finding.agent_name} | "
            f"perspective={finding.perspective.value} | stance={finding.stance.value} | "
            f"confidence={finding.confidence:.2f}\n"
            f"summary: {body}\n"
            f"claims:\n{claims or '  (none)'}"
        )
    return "\n\n".join(blocks)


def truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


__all__ = [
    "ANALYST_SYSTEM",
    "SYSTEM_ANALYST",
    "SYSTEM_PLANNER",
    "SYSTEM_SYNTHESIZER",
    "SYSTEM_VERIFIER",
    "format_evidence",
    "format_findings_context",
    "role_header",
    "truncate",
]
