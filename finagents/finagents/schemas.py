"""Core domain schemas for the multi-agent financial reasoning system.

Every artifact that moves between agents is a validated Pydantic model, so a
malformed agent output fails loudly at the boundary instead of silently
corrupting downstream reasoning.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class Perspective(str, Enum):
    """Analytical lenses a planner can assign to specialized agents."""

    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    TECHNICAL = "technical"
    RISK = "risk"


class Stance(str, Enum):
    """Directional conclusion of a finding or a synthesis."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EvidenceType(str, Enum):
    FILING = "filing"
    EARNINGS = "earnings"
    NEWS = "news"
    TRANSCRIPT = "transcript"
    MARKET_DATA = "market_data"
    REGULATORY = "regulatory"
    ANALYST_NOTE = "analyst_note"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DERIVED = "derived"


class ClaimType(str, Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictType(str, Enum):
    DIRECTION = "directional"
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    SCOPE = "scope"


# --------------------------------------------------------------------------
# Retrieval primitives
# --------------------------------------------------------------------------


class MetadataFilters(BaseModel):
    """Structured constraints applied as a hard gate before fusion."""

    model_config = ConfigDict(extra="allow")

    tickers: List[str] = Field(default_factory=list)
    doc_types: List[EvidenceType] = Field(default_factory=list)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    sources: List[str] = Field(default_factory=list)
    perspectives: List[Perspective] = Field(default_factory=list)

    def matches(self, doc: "Document") -> bool:
        meta = doc.metadata
        if self.tickers and not (set(self.tickers) & set(meta.get("tickers", []))):
            return False
        if self.doc_types and doc.doc_type not in self.doc_types:
            return False
        if self.date_from and (
            doc.published_at is None or doc.published_at.date() < self.date_from
        ):
            return False
        if self.date_to and (
            doc.published_at is None or doc.published_at.date() > self.date_to
        ):
            return False
        if self.sources and doc.source not in self.sources:
            return False
        if self.perspectives:
            tags = set(meta.get("perspectives", []))
            if not (set(p.value for p in self.perspectives) & tags):
                return False
        return True

    def describe(self) -> str:
        bits = []
        if self.tickers:
            bits.append(f"tickers={','.join(self.tickers)}")
        if self.doc_types:
            bits.append(f"types={','.join(d.value for d in self.doc_types)}")
        if self.date_from or self.date_to:
            bits.append(f"dates={self.date_from or '*'}..{self.date_to or '*'}")
        if self.sources:
            bits.append(f"sources={','.join(self.sources)}")
        if self.perspectives:
            bits.append(f"perspectives={','.join(p.value for p in self.perspectives)}")
        return "; ".join(bits) if bits else "none"


class Document(BaseModel):
    """A single retrievable unit of financial evidence."""

    model_config = ConfigDict(extra="allow")

    doc_id: str = ""
    title: str
    text: str
    doc_type: EvidenceType
    source: str
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    reliability: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_id(self) -> "Document":
        if not self.doc_id:
            object.__setattr__(
                self, "doc_id", _stable_id("doc", self.title, self.source)
            )
        return self

    def recency(self, as_of: Optional[date] = None) -> float:
        """Exponential decay in [0,1]; unknown dates are treated as stale."""
        if self.published_at is None:
            return 0.0
        ref = as_of or self.published_at.date()
        days = max((ref - self.published_at.date()).days, 0)
        return float(0.5 ** (days / 365.0))

    def citation(self) -> str:
        stamp = self.published_at.date().isoformat() if self.published_at else "undated"
        return f"{self.title} ({self.source}, {stamp})"


class ScoredDocument(BaseModel):
    document: Document
    score: float
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    graph_score: float = 0.0
    metadata_score: float = 0.0
    rank_signals: Dict[str, float] = Field(default_factory=dict)
    kg_context: List[str] = Field(default_factory=list)
    explanation: str = ""


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


class UserQuery(BaseModel):
    raw_text: str
    tickers: List[str] = Field(default_factory=list)
    as_of: Optional[date] = None
    horizon: str = "medium_term"
    user_context: Dict[str, Any] = Field(default_factory=dict)


class Intent(BaseModel):
    """Output of intent analysis, before decomposition."""

    goal: str
    question_type: str
    tickers: List[str] = Field(default_factory=list)
    time_horizon: str = "medium_term"
    entities: List[str] = Field(default_factory=list)
    needs_market_data: bool = True
    needs_news: bool = False
    needs_filings: bool = False
    needs_risk: bool = False
    in_scope: Optional[bool] = None
    rationale: str = ""


class TaskSpec(BaseModel):
    """A unit of work handed to exactly one specialized agent."""

    task_id: str
    description: str
    perspective: Perspective
    assigned_agent: str
    focus_metrics: List[str] = Field(default_factory=list)
    retrieval_queries: List[str] = Field(default_factory=list)
    metadata_filters: MetadataFilters = Field(default_factory=MetadataFilters)
    priority: int = 5
    depends_on: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_id(self) -> "TaskSpec":
        if not self.task_id:
            object.__setattr__(
                self,
                "task_id",
                _stable_id("task", self.perspective.value, self.description),
            )
        return self


class ResearchPlan(BaseModel):
    plan_id: str
    query: UserQuery
    intent: Intent
    tasks: List[TaskSpec]
    decomposition_rationale: str = ""
    skipped_perspectives: List[Perspective] = Field(default_factory=list)
    llm_trace_id: Optional[str] = None

    def tasks_for(self, perspective: Perspective) -> List[TaskSpec]:
        return [t for t in self.tasks if t.perspective == perspective]

    def dependency_order(self) -> List[TaskSpec]:
        """Topologically order tasks, tolerating cycles and dangling deps."""
        by_id = {t.task_id: t for t in self.tasks}
        ordered: List[TaskSpec] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(tid: str) -> None:
            if tid in seen or tid in visiting:
                return
            visiting.add(tid)
            task = by_id.get(tid)
            if task is not None:
                for dep in task.depends_on:
                    visit(dep)
                visiting.discard(tid)
                seen.add(tid)
                ordered.append(task)

        for task in self.tasks:
            visit(task.task_id)
        return ordered


# --------------------------------------------------------------------------
# Agent outputs
# --------------------------------------------------------------------------


class Citation(BaseModel):
    doc_id: str
    title: str
    source: str
    quote: str = ""
    url: Optional[str] = None


class Claim(BaseModel):
    """An atomic, checkable assertion extracted from an agent's reasoning."""

    claim_id: str = ""
    text: str
    claim_type: ClaimType = ClaimType.QUALITATIVE
    stance: Stance = Stance.NEUTRAL
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    citations: List[Citation] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    caveats: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_id(self) -> "Claim":
        if not self.claim_id:
            object.__setattr__(self, "claim_id", _stable_id("claim", self.text))
        return self


class AgentFinding(BaseModel):
    """One agent's independent, evidence-backed conclusion."""

    task_id: str
    agent_name: str
    perspective: Perspective
    summary: str
    stance: Stance = Stance.NEUTRAL
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    claims: List[Claim] = Field(default_factory=list)
    evidence: List[ScoredDocument] = Field(default_factory=list)
    key_metrics: Dict[str, float] = Field(default_factory=dict)
    risks: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    llm_trace_id: Optional[str] = None

    def cited_doc_ids(self) -> set[str]:
        return {c.doc_id for cl in self.claims for c in cl.citations}

    def supported_claim_ratio(self) -> float:
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.citations) / len(self.claims)


class AgentError(BaseModel):
    task_id: str
    agent_name: str
    error: str
    recoverable: bool = True


# --------------------------------------------------------------------------
# Verification & synthesis
# --------------------------------------------------------------------------


class ClaimVerification(BaseModel):
    claim_id: str
    verdict: Verdict
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    supporting_citations: List[Citation] = Field(default_factory=list)
    contradicting_citations: List[Citation] = Field(default_factory=list)
    arithmetic_check: Optional[str] = None
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0)


class Conflict(BaseModel):
    conflict_id: str
    conflict_type: ConflictType
    description: str
    claim_ids: List[str] = Field(default_factory=list)
    agent_names: List[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM
    resolution: str = ""
    resolved: bool = False


class ConfidenceReport(BaseModel):
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quality: float = 0.0
    agreement: float = 0.0
    coverage: float = 0.0
    corroboration: float = 0.0
    penalties: List[str] = Field(default_factory=list)
    components: Dict[str, float] = Field(default_factory=dict)

    def band(self) -> str:
        if self.overall >= 0.75:
            return "high"
        if self.overall >= 0.5:
            return "moderate"
        if self.overall >= 0.3:
            return "low"
        return "very_low"


class SynthesizedSection(BaseModel):
    heading: str
    perspective: Perspective
    body: str
    stance: Stance = Stance.NEUTRAL
    confidence: float = 0.0
    claim_ids: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class FinancialResearchReport(BaseModel):
    report_id: str
    query: UserQuery
    generated_at: datetime = Field(default_factory=_utc_now)
    thesis: str
    overall_stance: Stance = Stance.NEUTRAL
    confidence: ConfidenceReport = Field(default_factory=ConfidenceReport)
    sections: List[SynthesizedSection] = Field(default_factory=list)
    verified_claims: List[ClaimVerification] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    findings: List[AgentFinding] = Field(default_factory=list)
    evidence: List[ScoredDocument] = Field(default_factory=list)
    key_metrics: Dict[str, float] = Field(default_factory=dict)
    risks: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    # Named so a reader can tell whether a figure came from a filing, a tagged
    # XBRL fact, an aggregator, or a price series. A report that does not say
    # where its numbers came from is asking to be trusted on faith.
    data_sources: Dict[str, str] = Field(default_factory=dict)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)

    def all_citations(self) -> List[Citation]:
        seen: Dict[str, Citation] = {}
        for section in self.sections:
            for cite in section.citations:
                seen.setdefault(cite.doc_id, cite)
        for finding in self.findings:
            for claim in finding.claims:
                for cite in claim.citations:
                    seen.setdefault(cite.doc_id, cite)
        return list(seen.values())

    def unsupported_claims(self) -> List[ClaimVerification]:
        return [
            v
            for v in self.verified_claims
            if v.verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)
        ]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class Settings(BaseModel):
    """Runtime configuration. Every field is overridable by env var."""

    model_config = ConfigDict(extra="allow")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0
    # Generous by default, because Ollama on a CPU-only machine prefills at
    # roughly 3ms per character: a 9,000-character evidence block costs about 80
    # seconds before a single token is generated, and the financial agent's prompt
    # is larger still. A 120s budget fails mid-run on hardware that is working
    # exactly as intended, and the agent drops out of the report. This only bites
    # a slow daemon - an unreachable one fails immediately on connection refused -
    # so the long timeout costs nothing when the service is actually down.
    llm_timeout_s: float = 300.0
    llm_max_tokens: int = 2048

    retrieval_top_k: int = 8
    retrieval_per_task_k: int = 6
    fusion_rrf_k: int = 60

    kg_max_hops: int = 2
    # Online, one 10-Q arrives as ~20 chunks sharing a source string. Without a
    # hard per-filing cap a single filing can occupy every retrieval slot.
    max_docs_per_collection: int = 2

    # Online data sources. `sec_user_agent` is not optional in practice: the SEC
    # requires a declared contact and throttles or blocks anonymous bulk access,
    # so the default is a placeholder that should be overridden in real use.
    # Kept in step with `data.sec.DEFAULT_SEC_USER_AGENT` by a test; `data`
    # imports this module, so the constant cannot be imported from there.
    sec_user_agent: str = "finagents/0.1 (research; set FINAGENTS_SEC_USER_AGENT)"
    cache_dir: str = ".cache/finagents"
    filings_per_ticker: int = 3
    max_chunks_per_filing: int = 20
    max_chunks_per_section: int = 3
    news_per_ticker: int = 10
    # Google News items shorter than this are a headline with no retrievable
    # body; keeping them produces citations that prove nothing.
    news_min_body_chars: int = 120
    http_timeout_s: float = 20.0

    parallel_agents: bool = True
    enable_verification: bool = True
    seed: int = 17

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        env_map = {
            "ollama_base_url": "FINAGENTS_OLLAMA_BASE_URL",
            "ollama_model": "FINAGENTS_OLLAMA_MODEL",
            "ollama_embedding_model": "FINAGENTS_OLLAMA_EMBEDDING_MODEL",
            "llm_temperature": "FINAGENTS_LLM_TEMPERATURE",
            "llm_timeout_s": "FINAGENTS_LLM_TIMEOUT_S",
            "llm_max_tokens": "FINAGENTS_LLM_MAX_TOKENS",
            "retrieval_top_k": "FINAGENTS_TOP_K",
            "retrieval_per_task_k": "FINAGENTS_TOP_K_PER_TASK",
            "kg_max_hops": "FINAGENTS_KG_MAX_HOPS",
            "max_docs_per_collection": "FINAGENTS_MAX_DOCS_PER_COLLECTION",
            "sec_user_agent": "FINAGENTS_SEC_USER_AGENT",
            "cache_dir": "FINAGENTS_CACHE_DIR",
            "filings_per_ticker": "FINAGENTS_FILINGS_PER_TICKER",
            "max_chunks_per_filing": "FINAGENTS_MAX_CHUNKS_PER_FILING",
            "news_per_ticker": "FINAGENTS_NEWS_PER_TICKER",
            "news_min_body_chars": "FINAGENTS_NEWS_MIN_BODY_CHARS",
            "max_chunks_per_section": "FINAGENTS_MAX_CHUNKS_PER_SECTION",
            "http_timeout_s": "FINAGENTS_HTTP_TIMEOUT_S",
            "parallel_agents": "FINAGENTS_PARALLEL",
            "enable_verification": "FINAGENTS_VERIFY",
        }
        import os

        values: Dict[str, Any] = {}
        for field, env_name in env_map.items():
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            default = cls.model_fields[field].default
            if isinstance(default, bool):
                values[field] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                values[field] = int(raw)
            elif isinstance(default, float):
                values[field] = float(raw)
            else:
                values[field] = raw
        values.update(overrides)
        return cls(**values)


__all__ = [
    "AgentError",
    "AgentFinding",
    "Citation",
    "Claim",
    "ClaimType",
    "ClaimVerification",
    "Conflict",
    "ConflictType",
    "Document",
    "EvidenceType",
    "FinancialResearchReport",
    "Intent",
    "MetadataFilters",
    "Perspective",
    "ResearchPlan",
    "ScoredDocument",
    "Severity",
    "Settings",
    "Stance",
    "SynthesizedSection",
    "TaskSpec",
    "UserQuery",
    "Verdict",
]
