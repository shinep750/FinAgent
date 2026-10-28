"""A deterministic :class:`LLMClient` for tests.

Production code is backed by Ollama and there is no offline fallback. Tests need
the opposite trade: no daemon, no network, no flakiness, and byte-identical
output on every run so assertions can be exact. This module provides that, and
it lives under ``tests/`` on purpose - a deterministic client inside the package
would eventually be reachable from the CLI again, which is precisely the
fallback behaviour this project removed.

The responses are built from the *evidence actually present in the prompt*, not
from canned text: doc_ids, quotes and computed metrics are read back out of the
context blocks. A canned finding would let a citation bug pass unnoticed, because
the fake would be citing documents the retriever never returned.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from finagents.llm.base import LLMClient
from finagents.prompts import ROLE_MARKER

_DOC_RE = re.compile(
    r"^DOC (?P<doc_id>\S+) \| (?P<title>.*?) \| source=(?P<source>.*?) \| "
    r"date=(?P<date>.*?) \| type=(?P<dtype>.*?) \| reliability=(?P<rel>[\d.]+) \| "
    r"score=(?P<score>[\d.]+)\n(?P<text>.*?)(?=\n\nDOC |\Z)",
    re.DOTALL | re.MULTILINE,
)
_FINDING_RE = re.compile(
    r"^FINDING (?P<task_id>\S+) \| agent=(?P<agent>.*?) \| perspective=(?P<perspective>\S+) \| "
    r"stance=(?P<stance>\S+) \| confidence=(?P<conf>[\d.]+)\n"
    r"summary: (?P<summary>.*?)\nclaims:\n(?P<claims>.*?)(?=\n\nFINDING |\Z)",
    re.DOTALL | re.MULTILINE,
)
_METRIC_RE = re.compile(r"COMPUTED METRICS[^\n]*\n(.*)")
_CLAIM_LINE_RE = re.compile(
    r"^  - \[(?P<claim_id>[^\]]+)\] (?P<text>.*?) \(stance=", re.MULTILINE
)
_QUERY_RE = re.compile(r"USER QUERY: (?P<query>.*)")
_SUBJECT_RE = re.compile(r"SUBJECT: (?P<subject>.*)")
_TICKER_RE = re.compile(r"\$\$?([A-Z]{1,5})\b")


class FakeLLMClient(LLMClient):
    """Role-dispatching, evidence-grounded, fully deterministic client."""

    name = "fake"

    def __init__(
        self,
        responses: Optional[Dict[str, Any]] = None,
        *,
        malformed_first: bool = False,
    ) -> None:
        # Keyed by role; any role absent from this map is answered by the
        # default handler for that role.
        self.responses: Dict[str, Any] = dict(responses or {})
        self.calls: List[Dict[str, Any]] = []
        # Exercises `LLMClient.generate_json`'s repair path deterministically.
        self.malformed_first = malformed_first
        self._served_malformed = False

    # -- recording --------------------------------------------------------

    def calls_for(self, role: str) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["role"] == role]

    def reset(self) -> None:
        self.calls.clear()

    # -- contract ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        role = detect_role(prompt, system)
        self.calls.append(
            {
                "role": role,
                "prompt": prompt,
                "temperature": temperature,
                "system": system,
            }
        )
        if self.malformed_first and not self._served_malformed:
            self._served_malformed = True
            return "Sure! Here is the analysis you asked for:\nnot json at all"
        if role in self.responses:
            payload = self.responses[role]
            return payload if isinstance(payload, str) else json.dumps(payload)
        return json.dumps(self._default(role, prompt))

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "available": True,
            "detail": "deterministic test client",
        }

    # -- default handlers -------------------------------------------------

    def _default(self, role: str, prompt: str) -> Dict[str, Any]:
        handler = {
            "planner": self._plan,
            "verifier": self._verify,
            "synthesizer": self._synthesize,
            "financial": self._analyse,
            "news": self._analyse,
            "technical": self._analyse,
            "risk": self._analyse,
        }.get(role, self._analyse)
        return handler(prompt)

    def _plan(self, prompt: str) -> Dict[str, Any]:
        # One call analyses intent, the next decomposes; the step marker in the
        # instruction is what distinguishes them, exactly as in the real planner.
        if "STEP 1" in prompt:
            return self._intent(prompt)
        return self._decompose(prompt)

    def _intent(self, prompt: str) -> Dict[str, Any]:
        raw = _QUERY_RE.search(prompt)
        query = raw.group("query").strip() if raw else ""
        lowered = query.lower()
        tickers = _TICKER_RE.findall(query.upper()) or _TICKER_RE.findall(query)
        return {
            "goal": query or "assess the security",
            "question_type": _question_type(lowered),
            "tickers": sorted(set(tickers)),
            "time_horizon": _horizon(lowered),
            "entities": sorted(set(tickers)),
            "needs_market_data": True,
            "needs_news": bool(_NEWS_WORDS & _words(lowered)),
            "needs_filings": bool(_FILING_WORDS & _words(lowered)),
            "needs_risk": bool(_RISK_WORDS & _words(lowered)),
            "rationale": "classify the request into the lenses it requires",
        }

    def _decompose(self, prompt: str) -> Dict[str, Any]:
        wanted = re.findall(r"^REQUIRED PERSPECTIVES: (?P<v>.*)$", prompt, re.MULTILINE)
        if wanted:
            perspectives = [
                p.strip() for p in re.split(r"[,\s]+", wanted[0]) if p.strip()
            ]
        else:
            raw = _QUERY_RE.search(prompt)
            lowered = (raw.group("query") if raw else "").lower()
            words = _words(lowered)
            perspectives = [p for p, kws in _PERSPECTIVE_HINTS.items() if kws & words]
            if not perspectives:
                perspectives = ["fundamental", "risk"]
        tasks = []
        for priority, perspective in enumerate(perspectives, start=1):
            tasks.append(
                {
                    "description": f"{perspective} analysis of {perspective} factors",
                    "perspective": perspective,
                    "assigned_agent": f"{perspective} analyst",
                    "focus_metrics": _FOCUS.get(perspective, []),
                    "retrieval_queries": [
                        f"{perspective} {p}" for p in perspectives[:1]
                    ],
                    "metadata_filters": {},
                    "priority": priority,
                }
            )
        return {"tasks": tasks}

    def _analyse(self, prompt: str) -> Dict[str, Any]:
        docs = parse_documents(prompt)
        metrics = parse_metrics(prompt)
        perspective = _perspective_from_prompt(prompt)
        subject = _SUBJECT_RE.search(prompt)
        subject_text = subject.group("subject").strip() if subject else "the security"

        key_metrics = {name: value for name, value in list(metrics.items())[:6]}
        if not docs:
            # No evidence means no claims. A finding with zero citations would
            # enter verification as an unverifiable claim and dilute the report
            # with content that was never read from a source.
            return {
                "summary": (
                    f"No {perspective} evidence was retrieved for {subject}, so no conclusion "
                    "is drawn."
                ),
                "stance": "neutral",
                "confidence": 0.2,
                "claims": [],
                "key_metrics": dict(key_metrics),
                "risks": [],
                "caveats": ["no evidence retrieved; no claims made"],
                "reasoning_steps": ["retrieved 0 evidence documents"],
            }
        claims: List[Dict[str, Any]] = []
        for index, doc in enumerate(docs[:3]):
            claims.append(
                {
                    "text": f"{subject_text}: {doc['summary']}",
                    "claim_type": "quantitative" if doc["text"] else "qualitative",
                    "stance": "neutral",
                    "confidence": round(min(0.9, 0.45 + 0.1 * index), 2),
                    "citations": [doc["citation"]],
                    "metrics": dict(list(key_metrics.items())[:3]),
                    "caveats": ["drawn from retrieved evidence only"],
                }
            )
        return {
            "summary": _analyse_summary(perspective, subject_text, docs, metrics),
            "stance": "neutral",
            "confidence": 0.55 if docs else 0.2,
            "claims": claims,
            "key_metrics": key_metrics,
            "risks": [f"{perspective} risk not fully covered by retrieved evidence"],
            "caveats": ["deterministic test analysis; not model reasoning"],
            "reasoning_steps": [
                f"parsed {len(docs)} evidence document(s)",
                f"carried forward {len(key_metrics)} computed metric(s)",
            ],
        }

    def _verify(self, prompt: str) -> Dict[str, Any]:
        docs = {d["doc_id"]: d for d in parse_documents(prompt)}
        entries = []
        for claim_id, text, cited in parse_claims(prompt):
            usable = [c for c in cited if c in docs]
            if usable:
                verdict, score = "supported", 0.9
                supporting = [docs[c]["citation"] for c in usable[:2]]
                contradicting: List[Dict[str, Any]] = []
                rationale = "claim text is present in the cited evidence"
            else:
                verdict, score = "unsupported", 0.1
                supporting, contradicting = [], []
                rationale = "no cited document was present in the verification context"
            entries.append(
                {
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "support_score": score,
                    "rationale": rationale,
                    "supporting_citations": supporting,
                    "contradicting_citations": contradicting,
                }
            )
        return {"verifications": entries}

    def _synthesize(self, prompt: str) -> Dict[str, Any]:
        sections = []
        for finding in parse_findings(prompt):
            claims = _CLAIM_LINE_RE.findall(finding["claims"])
            body = finding["summary"].strip()
            if claims:
                body += (
                    " Supporting claims: " + "; ".join(c[1] for c in claims[:4]) + "."
                )
            sections.append(
                {
                    "task_id": finding["task_id"],
                    "heading": f"{finding['perspective'].title()} view",
                    "body": body,
                    "stance": finding["stance"],
                    "confidence": float(finding["confidence"]),
                }
            )
        return {"sections": sections}


# --------------------------------------------------------------------------
# Prompt parsing
# --------------------------------------------------------------------------


def detect_role(prompt: str, system: Optional[str] = None) -> str:
    for text in (system or "", prompt):
        match = re.search(re.escape(ROLE_MARKER) + r"\s*([a-z_]+)", text or "")
        if match:
            return match.group(1)
    return "unknown"


def parse_documents(prompt: str) -> List[Dict[str, Any]]:
    """Read back the `DOC ...` blocks the retriever actually supplied."""
    docs = []
    for match in _DOC_RE.finditer(prompt):
        text = " ".join(match.group("text").split())
        doc_id = match.group("doc_id")
        docs.append(
            {
                "doc_id": doc_id,
                "title": match.group("title").strip(),
                "source": match.group("source").strip(),
                "date": match.group("date").strip(),
                "doc_type": match.group("dtype").strip(),
                "reliability": float(match.group("rel")),
                "text": text,
                "summary": _first_sentence(text) or match.group("title").strip(),
                "citation": {
                    "doc_id": doc_id,
                    "title": match.group("title").strip(),
                    "source": match.group("source").strip(),
                    "quote": _first_sentence(text) or text[:180],
                },
            }
        )
    return docs


def parse_metrics(prompt: str) -> Dict[str, float]:
    match = _METRIC_RE.search(prompt)
    if not match:
        return {}
    out: Dict[str, float] = {}
    for pair in match.group(1).split(","):
        if "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            continue
    return out


def parse_claims(prompt: str) -> List[tuple]:
    """Return (claim_id, text, cited_doc_ids) for each claim under verification."""
    out = []
    for match in _CLAIM_LINE_RE.finditer(prompt):
        claim_id = match.group("claim_id")
        tail = prompt[match.end() : prompt.find("\n", match.end()) or len(prompt)]
        cited = re.search(r"cites=([^\)]*)", tail)
        ids = (
            [
                c.strip()
                for c in cited.group(1).split(",")
                if c.strip() and c.strip() != "none"
            ]
            if cited
            else []
        )
        out.append((claim_id, match.group("text").strip(), ids))
    return out


def parse_findings(prompt: str) -> List[Dict[str, Any]]:
    out = []
    for match in _FINDING_RE.finditer(prompt):
        out.append(
            {
                "task_id": match.group("task_id"),
                "agent": match.group("agent").strip(),
                "perspective": match.group("perspective").strip(),
                "stance": match.group("stance").strip(),
                "confidence": match.group("conf"),
                "summary": match.group("summary").strip(),
                "claims": match.group("claims"),
            }
        )
    return out


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _words(text: str) -> set:
    return set(re.findall(r"[a-z']+", text or ""))


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s", text)
    return (text[: match.start()] if match else text)[:300].strip()


def _question_type(lowered: str) -> str:
    if _DECISION & _words(lowered):
        return "investment decision"
    if "risk" in _words(lowered):
        return "risk assessment"
    if _NEWS_WORDS & _words(lowered):
        return "catalyst review"
    if "technical" in _words(lowered) or "chart" in _words(lowered):
        return "technical analysis"
    return "fundamental assessment"


def _horizon(lowered: str) -> str:
    if {"short", "week", "month", "near", "today"} & _words(lowered):
        return "short_term"
    if {"long", "decade", "2030", "2035"} & _words(lowered):
        return "long_term"
    return "medium_term"


def _perspective_from_prompt(prompt: str) -> str:
    match = re.search(r"^PERSPECTIVE: (\w+)$", prompt, re.MULTILINE)
    if match:
        return match.group(1)
    system = match.group(1) if match else ""
    for perspective in ("financial", "news", "technical", "risk"):
        if perspective in (system or "").lower():
            return "financial" if perspective == "financial" else perspective
    return "fundamental"


def _analyse_summary(
    perspective: str,
    subject: str,
    docs: List[Dict[str, Any]],
    metrics: Dict[str, float],
) -> str:
    if not docs:
        return f"No {perspective} evidence was retrieved for {subject}, so no conclusion is drawn."
    parts = [
        f"{perspective.title()} review of {subject} based on {len(docs)} retrieved "
        f"document(s) ({', '.join(d['source'] for d in docs[:3])})."
    ]
    if metrics:
        rendered = ", ".join(f"{k}={v:g}" for k, v in list(sorted(metrics.items()))[:5])
        parts.append(f"Computed from the price series: {rendered}.")
    parts.append(f"Top evidence: {docs[0]['summary']}")
    return " ".join(parts)


_NEWS_WORDS = {
    "news",
    "headline",
    "catalyst",
    "sentiment",
    "latest",
    "recent",
    "announcement",
    "flow",
}
_FILING_WORDS = {
    "filing",
    "filings",
    "10-k",
    "10-q",
    "8-k",
    "sec",
    "annual",
    "quarterly",
    "report",
}
_RISK_WORDS = {
    "risk",
    "risks",
    "downside",
    "exposure",
    "fragile",
    "bear",
    "worst",
    "tail",
    "safe",
}
_DECISION = {
    "buy",
    "sell",
    "hold",
    "worth",
    "invest",
    "should",
    "overvalued",
    "undervalued",
    "attractive",
    "recommend",
    "good",
    "price",
    "target",
    "valuation",
    "expensive",
    "cheap",
}
_PERSPECTIVE_HINTS = {
    "fundamental": _DECISION
    | {"revenue", "earnings", "margin", "eps", "profit", "guidance"},
    "news": _NEWS_WORDS,
    "technical": {
        "technical",
        "chart",
        "momentum",
        "trend",
        "moving",
        "average",
        "rsi",
        "oversold",
        "overbought",
    },
    "risk": _RISK_WORDS,
}
_FOCUS = {
    "fundamental": ["revenue", "gross_margin", "operating_margin", "eps"],
    "news": ["catalyst", "sentiment", "material_event"],
    "technical": ["price_trend", "sma_50", "sma_200", "rsi"],
    "risk": ["downside", "volatility", "regulatory", "concentration"],
}


__all__ = [
    "FakeLLMClient",
    "detect_role",
    "parse_claims",
    "parse_documents",
    "parse_findings",
    "parse_metrics",
]
