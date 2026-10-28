"""Multi-Agent Financial Reasoning and Decision Support System.

An explainable, evidence-driven agentic framework: a planner decomposes a
question, specialised financial agents analyse it against hybrid-retrieved
evidence, and a verification layer checks, stress-tests and synthesises the
findings into a cited report.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .schemas import (  # noqa: F401
    AgentFinding,
    Claim,
    ClaimVerification,
    Conflict,
    ConfidenceReport,
    Document,
    EvidenceType,
    FinancialResearchReport,
    Intent,
    MetadataFilters,
    Perspective,
    ResearchPlan,
    ScoredDocument,
    Settings,
    Stance,
    TaskSpec,
    UserQuery,
    Verdict,
)

__all__ = [
    "__version__",
    "AgentFinding",
    "Claim",
    "ClaimVerification",
    "Conflict",
    "ConfidenceReport",
    "Document",
    "EvidenceType",
    "FinancialResearchReport",
    "Intent",
    "MetadataFilters",
    "Perspective",
    "ResearchPlan",
    "ScoredDocument",
    "Settings",
    "Stance",
    "TaskSpec",
    "UserQuery",
    "Verdict",
]
