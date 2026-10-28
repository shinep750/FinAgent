"""Workflow state for the LangGraph state machine."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..schemas import (
    AgentError,
    AgentFinding,
    ClaimVerification,
    Conflict,
    ConfidenceReport,
    FinancialResearchReport,
    ResearchPlan,
    ScoredDocument,
    Settings,
    TaskSpec,
    UserQuery,
)


class WorkflowState(BaseModel):
    """State threaded through the graph.

    Every node returns a partial state; LangGraph merges the returned dict into
    the running state. Lists are accumulated by the nodes themselves (by
    extending, never by relying on additive reducer behaviour) so that re-running
    a node in a resumed thread does not silently drop earlier work.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Input
    query: UserQuery
    settings: Settings = Field(default_factory=Settings)

    # Planning
    plan: Optional[ResearchPlan] = None

    # Retrieval
    evidence_by_task: Dict[str, List[ScoredDocument]] = Field(default_factory=dict)

    # Analysis
    findings: List[AgentFinding] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)

    # Verification
    verifications: List[ClaimVerification] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    confidence: Optional[ConfidenceReport] = None

    # Output
    report: Optional[FinancialResearchReport] = None

    # Observability. `operator.add` is required: a node receives a *copy* of the
    # state, so appending to `self.trace` in place is discarded when LangGraph
    # replaces the state with the node's return value. Without the additive
    # reducer the trace comes back empty and the run is unauditable.
    trace: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    status: str = "created"
    errors_are_fatal: bool = False

    def record(self, node: str, **detail: Any) -> Dict[str, Any]:
        """Build a trace entry to be returned by the node (do not mutate)."""
        entry: Dict[str, Any] = {"node": node}
        entry.update(detail)
        return entry

    def evidence_for(self, task: TaskSpec) -> List[ScoredDocument]:
        return self.evidence_by_task.get(task.task_id, [])


__all__ = ["WorkflowState"]
