"""Report rendering.

The renderer exists to make the audit trail legible: every claim in the report
can be traced to a document, every document to a retrieval score and a reason it
was selected, and every confidence number to its components. A financial
recommendation whose provenance cannot be inspected is not explainable, however
well it is written.
"""

from __future__ import annotations

import json
from typing import List

from ..schemas import (
    FinancialResearchReport,
    Verdict,
)

_RULE = "=" * 78
_THIN = "-" * 78


def render(report: FinancialResearchReport, include_evidence: bool = True) -> str:
    lines: List[str] = [_RULE, "MULTI-AGENT FINANCIAL RESEARCH REPORT", _RULE, ""]
    lines.append(f"Question : {report.query.raw_text}")
    if report.query.tickers:
        lines.append(f"Subject  : {', '.join(report.query.tickers)}")
    lines.append(f"Report   : {report.report_id}")
    lines.append(f"Generated: {report.generated_at.isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("THESIS")
    lines.append(_THIN)
    lines.append(f"  Stance     : {report.overall_stance.value.upper()}")
    lines.append(
        f"  Confidence : {report.confidence.overall:.2f} ({report.confidence.band()})"
    )
    lines.append("")
    for chunk in _wrap(report.thesis):
        lines.append(f"  {chunk}")
    lines.append("")

    lines.append("CONFIDENCE DECOMPOSITION")
    lines.append(_THIN)
    confidence = report.confidence
    lines.append(
        f"  Evidence quality : {confidence.evidence_quality:.2f}  (source reliability + retrieval rank)"
    )
    lines.append(
        f"  Lens agreement   : {confidence.agreement:.2f}  (do the perspectives converge)"
    )
    lines.append(
        f"  Coverage         : {confidence.coverage:.2f}  (did every planned lens report)"
    )
    lines.append(
        f"  Corroboration    : {confidence.corroboration:.2f}  (multi-sourced vs single-source)"
    )
    if confidence.penalties:
        lines.append("  Penalties applied:")
        for penalty in confidence.penalties:
            lines.append(f"    - {penalty}")
    lines.append("")

    if report.sections:
        lines.append("ANALYSIS BY PERSPECTIVE")
        lines.append(_THIN)
        for section in report.sections:
            lines.append("")
            lines.append(f"  {section.heading.upper()}")
            lines.append(
                f"  stance={section.stance.value} confidence={section.confidence:.2f} "
                f"claims={len(section.claim_ids)}"
            )
            for chunk in _wrap(section.body, indent="  "):
                lines.append(chunk)
            if section.citations:
                cites = ", ".join(sorted({c.doc_id for c in section.citations}))
                lines.append(f"  evidence: {cites}")
        lines.append("")

    if report.conflicts:
        lines.append("CONFLICTS BETWEEN AGENTS")
        lines.append(_THIN)
        for conflict in report.conflicts:
            mark = "RESOLVED" if conflict.resolved else "UNRESOLVED"
            lines.append(
                f"  [{conflict.severity.value.upper():8s} {conflict.conflict_type.value:12s} {mark}] "
                f"{', '.join(conflict.agent_names)}"
            )
            for chunk in _wrap(conflict.description, indent="    "):
                lines.append(chunk)
            if conflict.resolution:
                for chunk in _wrap(f"Resolution: {conflict.resolution}", indent="    "):
                    lines.append(chunk)
        lines.append("")

    verified = [
        v
        for v in report.verified_claims
        if v.verdict in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)
    ]
    rejected = [
        v
        for v in report.verified_claims
        if v.verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)
    ]
    lines.append("CLAIM VERIFICATION")
    lines.append(_THIN)
    lines.append(f"  verified  : {len(verified)}")
    lines.append(f"  rejected  : {len(rejected)}")
    if rejected:
        lines.append("  Claims excluded from the conclusion:")
        for item in rejected:
            lines.append(
                f"    [{item.verdict.value:20s}] {item.claim_id}: "
                + (_first_sentence(item.rationale) or "no rationale recorded")
            )
    lines.append("")

    if report.key_metrics:
        lines.append("KEY METRICS")
        lines.append(_THIN)
        for name, value in report.key_metrics.items():
            label, suffix = _metric_label(name)
            lines.append(f"  {label:34s} {_fmt_number(value)}{suffix}")
        lines.append("")

    if report.risks:
        lines.append("RISK FACTORS IDENTIFIED")
        lines.append(_THIN)
        for risk in report.risks:
            for chunk in _wrap(f"- {risk}", indent="  "):
                lines.append(chunk)
        lines.append("")

    if include_evidence and report.evidence:
        lines.append("EVIDENCE AND SOURCES")
        lines.append(_THIN)
        for scored in report.evidence:
            doc = scored.document
            date = (
                doc.published_at.date().isoformat() if doc.published_at else "undated"
            )
            lines.append(
                f"  [{scored.score:.3f}] {doc.doc_id}  ({doc.doc_type.value}, {date}, "
                f"reliability {doc.reliability:.2f})"
            )
            lines.append(f"      {doc.title}")
            lines.append(f"      source: {doc.source}")
            if scored.explanation:
                for chunk in _wrap(scored.explanation, indent="      "):
                    lines.append(chunk)
        lines.append("")

    if report.data_sources:
        lines.append("DATA SOURCES")
        lines.append(_THIN)
        for label, value in sorted(report.data_sources.items()):
            lines.append(f"  {label.replace('_', ' ').title():<18}: {value}")
        lines.append("")

    if report.limitations:
        lines.append("LIMITATIONS AND CAVEATS")
        lines.append(_THIN)
        for limitation in report.limitations:
            for chunk in _wrap(f"- {limitation}", indent="  "):
                lines.append(chunk)
        lines.append("")

    if report.execution_trace:
        lines.append("EXECUTION TRACE")
        lines.append(_THIN)
        for entry in report.execution_trace:
            detail = " ".join(f"{k}={v}" for k, v in entry.items() if k != "node")
            lines.append(f"  {entry.get('node', '?'):12s} {detail}")
        lines.append("")

    lines.append(_RULE)
    lines.append(
        "Not investment advice. Verify figures against primary sources before acting."
    )
    lines.append(_RULE)
    return "\n".join(lines)


def render_json(report: FinancialResearchReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, default=str)


_UNITS = {
    "pct": " (%)",
    "percent": " (%)",
    "amount": " (USD)",
    "revenue_amount": " (USD)",
    "billion": " (USD bn)",
    "margin": " (%)",
    "volatility": " (%)",
    "ratio": " (x)",
    "multiple": " (x)",
    "usd": " (USD)",
    "eps": " (USD)",
    "fcf_amount": " (USD)",
    "free_cash_flow_amount": " (USD)",
    "monetary_amount": " (USD)",
}


def _metric_label(name: str) -> tuple:
    """`gross_margin` -> `Gross margin (%)`.

    The unit is appended, never substituted: stripping the unit token out of
    the label would turn "gross margin" into "gross" and "NVDA volatility"
    into "NVDA".
    """
    raw = str(name).strip()
    lower = raw.lower()
    suffix = _UNITS.get(lower, "")
    if not suffix:
        for key, value in sorted(_UNITS.items(), key=lambda kv: -len(kv[0])):
            if lower.endswith("_" + key):
                suffix = value
                break
    tokens = [t for t in lower.split("_") if t]
    label = " ".join(t.upper() if t in _ACRONYMS else t for t in tokens)
    return (label or raw)[:34], suffix


_ACRONYMS = {
    "rsi",
    "sma",
    "ema",
    "macd",
    "eps",
    "pct",
    "pe",
    "pb",
    "ps",
    "ev",
    "ebitda",
    "roi",
    "roe",
    "fcf",
    "yoy",
    "qoq",
    "qoq",
    "usd",
    "nvda",
    "aapl",
    "amd",
    "cfo",
    "capex",
    "ltm",
    "ttm",
    "px",
    "vol",
    "atr",
    "obv",
}


def _fmt_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e9 or magnitude < 1e-3):
        return f"{value:,.4g}"
    if magnitude >= 1000:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:,.4g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _wrap(text: str, width: int = 74, indent: str = "  ") -> List[str]:
    import textwrap

    if not text:
        return []
    return textwrap.wrap(
        text, width=width, initial_indent=indent, subsequent_indent=indent
    ) or [indent]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    for terminator in (". ", "; "):
        if terminator in text:
            return text.split(terminator, 1)[0]
    return text[:160]


__all__ = ["render", "render_json"]
