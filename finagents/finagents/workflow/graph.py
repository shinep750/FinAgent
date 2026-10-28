"""LangGraph workflow.

    plan -> retrieve -> analyse -> verify -> synthesise -> report

`analyse` fans out across specialised agents (concurrently when
`settings.parallel_agents` is set) and re-folds their findings into the state.
`verify` runs the claim verifier, the conflict detector and the confidence
assessor, then routes to `synthesise` or, when a fault is both detected and
fatal, to `fail`.

Fault policy: an individual agent failing is recorded in `errors` and the run
continues, because a risk memo built from three lenses is still useful. Losing
the LLM entirely, or finding no evidence at all, is fatal -- a report with no
basis is worse than an error.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from ..agents import PlannerAgent, build_registry
from ..data.providers import MarketDataProvider, build_provider
from ..llm.base import LLMClient
from ..llm.factory import build_llm_client
from ..retrieval import FinancialKnowledgeGraph, HybridRetriever
from ..schemas import (
    AgentError,
    AgentFinding,
    Settings,
    TaskSpec,
    UserQuery,
)
from ..verification import (
    ClaimVerifier,
    ConfidenceAssessor,
    ConflictDetector,
    Synthesizer,
)
from .state import WorkflowState

logger = logging.getLogger(__name__)


class FinancialResearchWorkflow:
    """Composes the agents, retriever and verifiers into one runnable graph."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
        provider: Optional[MarketDataProvider] = None,
        knowledge_graph: Optional[FinancialKnowledgeGraph] = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.llm = llm or build_llm_client(self.settings)
        self.provider = provider or build_provider(self.settings)
        self.registry = build_registry(self.provider)
        self.planner = PlannerAgent(self.llm, self.settings, self.provider)
        self.kg = knowledge_graph
        self._plan: Optional[Any] = None
        self._retriever: Optional[HybridRetriever] = None
        self._retriever_key: Optional[tuple] = None
        self.verifier = ClaimVerifier(self.llm, self.settings)
        self.conflicts = ConflictDetector()
        self.confidence = ConfidenceAssessor()
        self.synthesizer = Synthesizer(self.llm, self.settings)
        self._graph: Optional[Any] = None
        # Optional caller-supplied hook, consulted between tasks inside the
        # long fan-out nodes. It may raise to abort the run (the API uses this
        # for its cancel button); leaving it None changes nothing.
        self.cancel_check: Optional[Callable[[], None]] = None

    def _check_cancel(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    # -- lazily built retriever ------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        """The retriever for the last planned ticker set, built on demand.

        Online there is no corpus to load up front: documents are fetched for
        the tickers the plan actually resolved, which is the difference between
        a few dozen documents and the entire EDGAR corpus. The knowledge graph is
        derived from whatever was retrieved, so its edges can only cite text the
        run actually read.
        """
        if self._retriever is not None and self._retriever_key == self._key():
            return self._retriever
        tickers = tuple(self._tickers())
        documents = self.provider.load_documents(tickers or None)
        knowledge_graph = self.kg or FinancialKnowledgeGraph.from_documents(documents)
        self._retriever = HybridRetriever(
            documents, knowledge_graph=knowledge_graph, settings=self.settings
        )
        self._retriever_key = self._key()
        logger.info(
            "retriever ready: %d documents, %d triples",
            len(documents),
            len(knowledge_graph.triples),
        )
        return self._retriever

    def _tickers(self) -> List[str]:
        plan = getattr(self, "_plan", None)
        return [t.upper() for t in (plan.intent.tickers if plan else [])]

    def _key(self) -> tuple:
        return tuple(self._tickers())

    def invalidate_retriever(self) -> None:
        self._retriever = None
        self._retriever_key = None

    # -- nodes ------------------------------------------------------------

    def node_plan(self, state: WorkflowState) -> Dict[str, Any]:
        started = time.perf_counter()
        plan = self.planner.plan(state.query)
        self._plan = plan
        # A new plan can name a different ticker set, so the cached retriever
        # from a previous run must not be reused against these documents.
        self.invalidate_retriever()
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("plan: %s", [t.perspective.value for t in plan.tasks])
        return {
            "plan": plan,
            "status": "planned",
            "trace": [
                state.record(
                    "plan",
                    tasks=len(plan.tasks),
                    perspectives=[t.perspective.value for t in plan.tasks],
                    skipped=[p.value for p in plan.skipped_perspectives],
                    ms=round(elapsed, 1),
                )
            ],
        }

    def node_retrieve(self, state: WorkflowState) -> Dict[str, Any]:
        started = time.perf_counter()
        plan = state.plan
        if plan is None:
            return {
                "status": "no_plan",
                "trace": [state.record("retrieve", skipped="no plan")],
            }
        by_task: Dict[str, List] = {}
        total = 0
        for task in plan.tasks:
            self._check_cancel()
            # Warm-start: if we already have evidence for this task in the state, reuse it.
            # In a follow-up, the Planner might reuse task IDs or we can match by perspective.
            existing = state.evidence_by_task.get(task.task_id, [])
            if existing:
                hits = existing
            else:
                hits = self.retriever.search_for_task(
                    task, top_k=self.settings.retrieval_per_task_k
                )
            by_task[task.task_id] = hits
            total += len(hits)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("retrieve: %d document(s) across %d task(s)", total, len(by_task))
        return {
            "evidence_by_task": by_task,
            "status": "retrieved",
            "trace": [
                state.record(
                    "retrieve",
                    documents=total,
                    tasks=len(by_task),
                    ms=round(elapsed, 1),
                )
            ],
        }

    def node_analyse(self, state: WorkflowState) -> Dict[str, Any]:
        started = time.perf_counter()
        plan = state.plan
        if plan is None:
            return {
                "status": "no_plan",
                "trace": [state.record("analyse", skipped="no plan")],
            }
        tasks = plan.dependency_order() or plan.tasks

        def run_one(task: TaskSpec) -> Dict[str, Any]:
            self._check_cancel()
            evidence = state.evidence_for(task)
            if not evidence:
                return {
                    "error": AgentError(
                        task_id=task.task_id,
                        agent_name=task.assigned_agent,
                        error="no evidence retrieved for this task",
                    )
                }
            try:
                agent = self.registry.get(task.assigned_agent, self.llm, self.settings)
                return {"finding": agent.run(task, evidence)}
            except Exception as exc:  # noqa: BLE001 - one agent must not sink the run
                logger.warning(
                    "agent %s failed on %s: %s", task.assigned_agent, task.task_id, exc
                )
                return {
                    "error": AgentError(
                        task_id=task.task_id,
                        agent_name=task.assigned_agent,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                }

        if self.settings.parallel_agents and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                results = list(pool.map(run_one, tasks))
        else:
            results = [run_one(task) for task in tasks]

        findings: List[AgentFinding] = []
        errors: List[AgentError] = []
        for result in results:
            if "finding" in result:
                findings.append(result["finding"])
            if "error" in result:
                errors.append(result["error"])
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("analyse: %d finding(s), %d error(s)", len(findings), len(errors))
        return {
            "findings": findings,
            "errors": errors,
            "status": "analysed" if findings else "analysis_failed",
            "trace": [
                state.record(
                    "analyse",
                    findings=len(findings),
                    errors=len(errors),
                    parallel=self.settings.parallel_agents,
                    ms=round(elapsed, 1),
                )
            ],
        }

    def node_verify(self, state: WorkflowState) -> Dict[str, Any]:
        started = time.perf_counter()
        if not state.findings:
            return {
                "status": "nothing_to_verify",
                "trace": [state.record("verify", claims=0, note="no findings")],
            }
        verifications = self.verifier.verify(state.findings)
        documents = {
            d.document.doc_id: d.document for f in state.findings for d in f.evidence
        }
        conflicts = self.conflicts.detect(state.findings, verifications, documents)
        confidence = self.confidence.assess(
            state.findings,
            verifications,
            conflicts,
            expected_tasks=len(state.plan.tasks) if state.plan else 0,
            errors=state.errors,
        )
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "verify: %d claim(s), %d conflict(s), confidence %.2f (%s)",
            len(verifications),
            len(conflicts),
            confidence.overall,
            confidence.band(),
        )
        return {
            "verifications": verifications,
            "conflicts": conflicts,
            "confidence": confidence,
            "status": "verified",
            "trace": [
                state.record(
                    "verify",
                    claims=len(verifications),
                    conflicts=len(conflicts),
                    confidence=confidence.overall,
                    band=confidence.band(),
                    ms=round(elapsed, 1),
                )
            ],
        }

    def node_synthesise(self, state: WorkflowState) -> Dict[str, Any]:
        started = time.perf_counter()
        confidence = state.confidence or self.confidence.assess(
            state.findings,
            state.verifications,
            state.conflicts,
            expected_tasks=len(state.plan.tasks) if state.plan else 0,
            errors=state.errors,
        )
        report = self.synthesizer.synthesise(
            query=state.query,
            plan=state.plan,
            findings=state.findings,
            verifications=state.verifications,
            conflicts=state.conflicts,
            confidence=confidence,
            errors=state.errors,
            execution_trace=list(state.trace),
            data_sources=self.provider.provenance(),
        )
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("synthesise: %d section(s)", len(report.sections))
        return {
            "report": report,
            "status": "complete",
            "trace": [
                state.record(
                    "synthesise",
                    sections=len(report.sections),
                    citations=len(report.all_citations()),
                    ms=round(elapsed, 1),
                )
            ],
        }

    def node_report(self, state: WorkflowState) -> Dict[str, Any]:
        report = state.report
        if report is None:
            return {
                "status": "failed",
                "trace": [state.record("report", note="no report")],
            }
        return {
            "status": "reported",
            "trace": [
                state.record(
                    "report",
                    report_id=report.report_id,
                    stance=report.overall_stance.value,
                )
            ],
        }

    def node_fail(self, state: WorkflowState) -> Dict[str, Any]:
        logger.error("workflow failed at %s", state.status)
        return {
            "status": "failed",
            "trace": [state.record("fail", reason=state.status)],
        }

    # -- edges ------------------------------------------------------------

    def build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", self.node_plan)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("analyse", self.node_analyse)
        graph.add_node("verify", self.node_verify)
        graph.add_node("synthesise", self.node_synthesise)
        graph.add_node("report", self.node_report)
        graph.add_node("fail", self.node_fail)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "analyse")
        graph.add_conditional_edges(
            "analyse",
            self._after_analyse,
            {"verify": "verify", "synthesise": "synthesise", "fail": "fail"},
        )
        graph.add_conditional_edges(
            "verify",
            self._after_verify,
            {"synthesise": "synthesise", "fail": "fail"},
        )
        graph.add_edge("synthesise", "report")
        graph.add_edge("report", END)
        graph.add_edge("fail", END)
        self._graph = graph.compile()
        return self._graph

    def _after_analyse(self, state: WorkflowState) -> str:
        """Findings go to verification; a barren run either fails or reports nothing."""
        if state.findings:
            if state.settings.enable_verification:
                return "verify"
            logger.warning(
                "verification disabled by settings; claims are reported unverified"
            )
            return "synthesise"
        if state.errors_are_fatal:
            logger.error("no findings produced and errors are fatal: %s", state.errors)
            return "fail"
        logger.warning("no findings produced; synthesising an empty report")
        return "synthesise"

    def _after_verify(self, state: WorkflowState) -> str:
        """Stop the run when verification finds the evidence unusable and
        `errors_are_fatal` is set. Otherwise an unverified report is still
        better than none, provided the report says which claims were dropped."""
        if (
            state.errors_are_fatal
            and state.confidence is not None
            and state.confidence.overall < 0.2
        ):
            logger.error(
                "confidence %.2f below the floor; aborting", state.confidence.overall
            )
            return "fail"
        return "synthesise"

    # -- entry point ------------------------------------------------------

    def run(
        self, query: UserQuery | str, thread_id: Optional[str] = None
    ) -> WorkflowState:
        if isinstance(query, str):
            query = UserQuery(raw_text=query)
        if query.tickers:
            # `--ticker` restricts the query, so the text is not scanned. But the
            # symbol still has to be canonicalised: passing `BRK.B` through
            # unresolved reaches EDGAR as a symbol it does not recognise, and the
            # report names a company whose filings were never read.
            query.tickers = (
                self.provider.canonical_tickers(query.tickers)
                if self.provider
                else [t.upper().strip() for t in query.tickers]
            )
        else:
            query.tickers = self.planner.resolve_tickers(query)
        graph = self._graph or self.build()
        state = WorkflowState(query=query, settings=self.settings)
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        result = graph.invoke(state, config=config)
        if isinstance(result, dict):
            return WorkflowState(**{**state.model_dump(), **result})
        return result


__all__ = ["FinancialResearchWorkflow"]
