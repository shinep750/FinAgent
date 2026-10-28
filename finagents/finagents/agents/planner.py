"""Planner agent.

Query -> intent -> task decomposition -> perspective identification ->
agent selection and assignment.

The planner is deliberately defensive. A local model will sometimes return a
malformed task list, an unknown perspective, or a plausible-looking ticker that
does not exist. Every one of those is repaired here, because a bad plan
propagates silently into every downstream agent and the final report.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from ..data import MarketDataProvider
from ..llm.base import LLMClient, build_prompt
from ..prompts import SYSTEM_PLANNER, role_header
from ..schemas import (
    EvidenceType,
    Intent,
    MetadataFilters,
    Perspective,
    ResearchPlan,
    Settings,
    TaskSpec,
    UserQuery,
)

logger = logging.getLogger(__name__)


class OutOfScopeError(ValueError):
    """The query is not about a supported equity/company, so no research run
    can produce a grounded answer. Surfaced to the CLI/API as a clean message
    instead of an empty or fabricated report."""


# Fallback retrieval focus per perspective, used when the model omits it.
_DEFAULT_FOCUS: Dict[Perspective, List[str]] = {
    Perspective.FUNDAMENTAL: [
        "revenue",
        "gross_margin",
        "operating_margin",
        "eps",
        "free_cash_flow",
    ],
    Perspective.NEWS: [
        "catalyst",
        "sentiment",
        "material_event",
        "management_commentary",
    ],
    Perspective.TECHNICAL: ["price_trend", "sma_50", "sma_200", "rsi", "volume"],
    Perspective.RISK: [
        "downside",
        "concentration",
        "regulatory",
        "valuation_fragility",
        "volatility",
    ],
}

# Document types worth retrieving for each perspective.
_DEFAULT_TYPES: Dict[Perspective, List[EvidenceType]] = {
    Perspective.FUNDAMENTAL: [
        EvidenceType.FILING,
        EvidenceType.EARNINGS,
        EvidenceType.TRANSCRIPT,
        EvidenceType.NEWS,
    ],
    Perspective.NEWS: [
        EvidenceType.NEWS,
        EvidenceType.REGULATORY,
        EvidenceType.EARNINGS,
    ],
    Perspective.TECHNICAL: [EvidenceType.MARKET_DATA, EvidenceType.NEWS],
    Perspective.RISK: [
        EvidenceType.FILING,
        EvidenceType.REGULATORY,
        EvidenceType.MARKET_DATA,
        EvidenceType.NEWS,
    ],
}

# Perspectives that treat narrative (news) as first-class evidence. A plan is
# only accepted if these always carry a news channel, regardless of what the
# model's own filters asked for. NEWS is included for the same reason it exists:
# a news task whose model-supplied filter said "filing" would, in a corpus with
# no filings (EDGAR down, or a company with none), retrieve nothing at all.
_NEWS_GROUNDED: Set[Perspective] = {
    Perspective.FUNDAMENTAL,
    Perspective.NEWS,
    Perspective.RISK,
}

_PERSPECTIVE_KEYWORDS: Dict[Perspective, tuple[str, ...]] = {
    Perspective.FUNDAMENTAL: (
        "fundamental",
        "revenue",
        "earnings",
        "valuation",
        "margin",
        "eps",
        "profit",
        "balance sheet",
        "cash flow",
        "guidance",
        "worth",
        "overvalued",
        "undervalued",
        "intrinsic",
        "expensive",
        "cheap",
        "fair value",
        "buy",
        "sell",
        "hold",
        "invest",
        "should i",
        "outlook",
        "forecast",
    ),
    Perspective.NEWS: (
        "news",
        "headline",
        "catalyst",
        "what happened",
        "sentiment",
        "recent",
        "announcement",
        "news flow",
        "latest",
        "today",
        "this week",
        "update",
    ),
    Perspective.TECHNICAL: (
        "technical",
        "chart",
        "momentum",
        "trend",
        "moving average",
        "support",
        "resistance",
        "price action",
        "rsi",
        "breakout",
        "overbought",
        "oversold",
    ),
    Perspective.RISK: (
        "risk",
        "volatility",
        "exposure",
        "downside",
        "safe",
        "worst case",
        "hedge",
        "bear case",
        "fragile",
        "concentration",
        "regulatory",
        "litigation",
        "tail risk",
    ),
}

# An investment recommendation cannot be answered from one lens. These phrases
# make a question a decision question, which forces fundamental and risk in
# regardless of which individual keywords happen to appear.
_DECISION_PHRASES: tuple[str, ...] = (
    "should i buy",
    "should i sell",
    "should i hold",
    "is it a buy",
    "worth buying",
    "worth investing",
    "good investment",
    "buy or sell",
    "worth it",
    "still a buy",
    "is now a good time",
    "overvalued or undervalued",
    "attractive",
    "any good",
)

# A ticker-shaped uppercase token ("ZZZZ") is a scope signal even when the data
# provider cannot resolve it: the user is asking about a company, so the run
# should degrade with an honest empty report rather than be refused. Question words
# and ordinary uppercase words are excluded so "WHAT IS THE CAPITAL OF FRANCE?"
# is not mistaken for a symbol.
_SYMBOL_TOKEN_RE = re.compile(r"\b([A-Z]{2,5}(?:[.-][A-Z]{1,2})?)\b")
_FOLLOW_UP_REFERENCE_RE = re.compile(
    r"\b(it|its|it's|they|their|them|this|that|these|those|company|stock|shares|same)\b"
    r"|^\s*(and|but|so|then|also|why|how about|what about|what else|anything else)\b"
)
_NON_SYMBOL_WORDS: Set[str] = {
    "A",
    "AN",
    "AND",
    "ANY",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BY",
    "CAN",
    "COULD",
    "DID",
    "DO",
    "DOES",
    "FOR",
    "FROM",
    "GET",
    "GIVE",
    "HAS",
    "HAVE",
    "HE",
    "HER",
    "HERE",
    "HIS",
    "HOW",
    "IF",
    "IN",
    "INTO",
    "IS",
    "IT",
    "ITS",
    "JUST",
    "MAY",
    "ME",
    "MIGHT",
    "MORE",
    "MOST",
    "MUST",
    "MY",
    "NEW",
    "NO",
    "NOT",
    "NOW",
    "OF",
    "ON",
    "ONLY",
    "OR",
    "OUR",
    "OUT",
    "OVER",
    "OWN",
    "PLEASE",
    "SHE",
    "SHOULD",
    "SO",
    "SOME",
    "TELL",
    "THAN",
    "THAT",
    "THE",
    "THEM",
    "THEN",
    "THERE",
    "THESE",
    "THEY",
    "THIS",
    "THOSE",
    "TO",
    "TOO",
    "UNDER",
    "UP",
    "US",
    "USE",
    "WAS",
    "WE",
    "WERE",
    "WHAT",
    "WHEN",
    "WHERE",
    "WHICH",
    "WHO",
    "WHY",
    "WILL",
    "WITH",
    "WOULD",
    "WRITE",
    "YES",
    "YOU",
    "YOUR",
    "AI",
    "CEO",
    "CFO",
    "IPO",
    "ETF",
    "SEC",
    "FDA",
    "GDP",
    "CPI",
    "FED",
    "EPS",
    "RSI",
    "SMA",
    "USA",
    "UK",
    "EU",
    "OK",
}


class PlannerAgent:
    def __init__(
        self,
        llm: LLMClient,
        settings: Optional[Settings] = None,
        provider: Optional[MarketDataProvider] = None,
        supported_tickers: Optional[Sequence[str]] = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or Settings()
        self.provider = provider
        self.supported_tickers = [
            t.upper()
            for t in (
                supported_tickers or (provider.supported_tickers() if provider else [])
            )
        ]

    # -- entity resolution ------------------------------------------------
    def resolve_tickers(self, query: UserQuery) -> List[str]:
        """Find the securities a query is about, from the text alone.

        Now supports Global and Indian markets. Validates against the provider
        rather than just the US SEC EDGAR map.
        """
        found: List[str] = []
        text = query.raw_text
        for candidate in re.findall(
            r"\$\s*([A-Za-z][A-Za-z.]{0,4}(?:[-.][A-Za-z]{1,2})?)", text
        ):
            ticker = self._accept(candidate, found)
            if ticker:
                found.append(ticker)
        for candidate in re.findall(r"\b([A-Z][A-Z.]{0,4}(?:[-.][A-Z]{1,2})?)\b", text):
            ticker = self._accept(candidate, found)
            if ticker:
                found.append(ticker)
        for raw in query.tickers:
            ticker = self._accept(raw, found)
            if ticker:
                found.append(ticker)
        return found

    def _accept(self, candidate: str, found: Sequence[str]) -> Optional[str]:
        """Validate one candidate against the provider, ignoring ones already found."""
        ticker = (
            self.provider.resolve_ticker(candidate)
            if self.provider
            else candidate.upper()
        )
        if ticker and ticker not in found:
            return ticker
        return None

    # -- intent -----------------------------------------------------------
    def analyse_intent(
        self, query: UserQuery, context: Optional[Dict[str, Any]] = None
    ) -> Intent:
        context_block = ""
        if context:
            prior_question = str(context.get("question", "")).strip()[:500]
            prior_thesis = str(context.get("thesis", "")).strip()[:700]
            prior_subject = ", ".join(
                str(t) for t in (context.get("tickers") or []) if str(t).strip()
            )
            context_block = (
                "\nCONVERSATION SO FAR (the new query may be a follow-up):\n"
                f"- PREVIOUS QUESTION: {prior_question or 'unknown'}\n"
                f"- PREVIOUS SUBJECT: {prior_subject or 'unknown'}\n"
                f"- PREVIOUS ANSWER SUMMARY: {prior_thesis or 'unknown'}\n"
                "If the new query does not name a company, treat it as being about "
                "PREVIOUS SUBJECT. Set in_scope to false when the new query is "
                "unrelated to that company, the market or investing (for example "
                "weather, coding, general knowledge or personal advice).\n"
            )
        prompt = build_prompt(
            system=SYSTEM_PLANNER,
            instruction=(
                f"{role_header('planner')}\n"
                f"USER QUERY: {query.raw_text}\n"
                f"COMMON TICKERS: {', '.join(self.supported_tickers) or 'none'}\n"
                f"{context_block}"
                "STEP 1 - INTENT ANALYSIS. Identify the user's goal, the type of "
                "question, the time horizon, and which data sources are needed. "
                "List the tickers the query is about. Tickers are resolved against "
                "market providers, so a company name is acceptable; do not invent "
                "symbols for private companies. Only return tickers when the query "
                "is about a publicly traded company, security, market or investing; "
                "for general knowledge, coding, weather, personal advice or any other "
                "topic, return an empty tickers list and set in_scope to false."
            ),
            output_schema=(
                '{"goal": "string", "question_type": "string", "tickers": ["TICKER"], '
                '"time_horizon": "short_term|medium_term|long_term", "entities": ["string"], '
                '"needs_market_data": true, "needs_news": false, "needs_filings": false, '
                '"needs_risk": false, "in_scope": true, "rationale": "string"}'
            ),
        )
        payload = self.llm.generate_json(
            prompt, SYSTEM_PLANNER, temperature=self.settings.llm_temperature
        )
        data = payload if isinstance(payload, dict) else {}

        # Deterministic tickers win: the text is authoritative, the model is not.
        resolved = self.resolve_tickers(query)
        model_tickers: List[str] = []
        for raw in data.get("tickers") or []:
            ticker = (
                self.provider.resolve_ticker(str(raw))
                if self.provider
                else str(raw).upper()
            )
            if ticker and ticker not in model_tickers:
                model_tickers.append(ticker)
        tickers = resolved or model_tickers
        lowered = query.raw_text.lower()
        needs_news = bool(data.get("needs_news")) or any(
            k in lowered for k in _PERSPECTIVE_KEYWORDS[Perspective.NEWS]
        )
        needs_filings = bool(data.get("needs_filings")) or any(
            k in lowered for k in _PERSPECTIVE_KEYWORDS[Perspective.FUNDAMENTAL]
        )
        needs_risk = bool(data.get("needs_risk")) or any(
            k in lowered for k in _PERSPECTIVE_KEYWORDS[Perspective.RISK]
        )
        raw_scope = data.get("in_scope")
        return Intent(
            goal=str(data.get("goal", "")).strip() or query.raw_text.strip()[:300],
            question_type=str(data.get("question_type", "investment_research")).strip()
            or "investment_research",
            tickers=tickers,
            time_horizon=str(data.get("time_horizon", query.horizon)) or query.horizon,
            entities=[str(e)[:80] for e in (data.get("entities") or [])][:12],
            needs_market_data=bool(data.get("needs_market_data", True)),
            needs_news=needs_news,
            needs_filings=needs_filings,
            needs_risk=needs_risk,
            in_scope=raw_scope if isinstance(raw_scope, bool) else None,
            rationale=str(data.get("rationale", "")).strip()[:600],
        )

    # -- perspective identification ---------------------------------------
    def _matched_perspectives(self, query: UserQuery) -> List[Perspective]:
        """Perspectives named explicitly by the question text."""
        lowered = query.raw_text.lower()
        chosen = [
            perspective
            for perspective, keywords in _PERSPECTIVE_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        if any(phrase in lowered for phrase in _DECISION_PHRASES):
            _add(chosen, Perspective.FUNDAMENTAL)
            _add(chosen, Perspective.RISK)
        return chosen

    def identify_perspectives(
        self, query: UserQuery, intent: Intent
    ) -> List[Perspective]:
        chosen = self._matched_perspectives(query)
        if intent.needs_news:
            _add(chosen, Perspective.NEWS)
        if intent.needs_filings:
            _add(chosen, Perspective.FUNDAMENTAL)
        if intent.needs_risk:
            _add(chosen, Perspective.RISK)
        if not chosen:
            chosen = [Perspective.FUNDAMENTAL, Perspective.NEWS, Perspective.RISK]
        return chosen

    # -- decomposition ----------------------------------------------------
    def decompose(
        self, query: UserQuery, intent: Intent, perspectives: Sequence[Perspective]
    ) -> List[TaskSpec]:
        ticker_hint = ", ".join(intent.tickers) or "the subject company"
        prompt = build_prompt(
            system=SYSTEM_PLANNER,
            instruction=(
                f"{role_header('planner')}\n"
                f"USER QUERY: {query.raw_text}\n"
                f"IDENTIFIED GOAL: {intent.goal}\n"
                f"SUBJECT: {ticker_hint}\n"
                f"SELECTED PERSPECTIVES: {', '.join(p.value for p in perspectives)}\n"
                "STEP 2 - TASK DECOMPOSITION AND AGENT ASSIGNMENT. Create one task per "
                "selected perspective. For each task give a description, the "
                "perspective, the assigned specialist agent, the metrics to focus on, "
                "one or two retrieval queries, and any document-type filter. Only use "
                "these agent names: financial_analysis, news_analysis, "
                "technical_analysis, risk_analysis."
            ),
            output_schema=(
                '{"tasks": [{"description": "string", "perspective": '
                '"fundamental|news|technical|risk", "assigned_agent": "string", '
                '"focus_metrics": ["string"], "retrieval_queries": ["string"], '
                '"metadata_filters": {"doc_types": ["filing|earnings|news|transcript|'
                'market_data|regulatory|analyst_note"], "tickers": ["TICKER"]}, '
                '"priority": 5}]}'
            ),
        )
        payload = self.llm.generate_json(
            prompt, SYSTEM_PLANNER, temperature=self.settings.llm_temperature
        )
        raw_tasks = payload.get("tasks") if isinstance(payload, dict) else None
        valid = {
            "financial_analysis": Perspective.FUNDAMENTAL,
            "news_analysis": Perspective.NEWS,
            "technical_analysis": Perspective.TECHNICAL,
            "risk_analysis": Perspective.RISK,
        }
        tasks: List[TaskSpec] = []
        seen: set[Perspective] = set()
        for raw in raw_tasks or []:
            if not isinstance(raw, dict):
                continue
            try:
                perspective = Perspective(str(raw.get("perspective", "")).lower())
            except ValueError:
                logger.warning(
                    "planner returned unknown perspective %r", raw.get("perspective")
                )
                continue
            if perspective not in perspectives or perspective in seen:
                continue
            agent = str(raw.get("assigned_agent", "")).strip()
            if agent not in valid or valid[agent] != perspective:
                agent = next(a for a, p in valid.items() if p == perspective)
            tasks.append(self._build_task(raw, perspective, agent, intent))
            seen.add(perspective)

        for perspective in perspectives:
            if perspective not in seen:
                logger.info(
                    "planner omitted the %s perspective; adding a default task",
                    perspective.value,
                )
                tasks.append(self._default_task(perspective, intent))
        tasks.sort(key=lambda t: -t.priority)
        return tasks

    def _build_task(
        self, raw: dict, perspective: Perspective, agent: str, intent: Intent
    ) -> TaskSpec:
        filters = raw.get("metadata_filters")
        doc_types: List[EvidenceType] = []
        for value in (filters or {}).get("doc_types", []) or []:
            try:
                doc_types.append(EvidenceType(str(value).lower()))
            except ValueError:
                continue
        tickers = [t.upper() for t in (filters or {}).get("tickers", []) or []]
        if intent.tickers:
            tickers = intent.tickers
        focus = [str(m)[:60] for m in (raw.get("focus_metrics") or [])][:8]
        queries = [
            str(q).strip()
            for q in (raw.get("retrieval_queries") or [])
            if str(q).strip()
        ][:3]
        description = (
            str(raw.get("description", "")).strip()
            or self._default_task(perspective, intent).description
        )
        return TaskSpec(
            task_id="",
            description=description[:500],
            perspective=perspective,
            assigned_agent=agent,
            focus_metrics=focus or list(_DEFAULT_FOCUS[perspective]),
            retrieval_queries=queries or [description[:200]],
            metadata_filters=MetadataFilters(
                tickers=tickers,
                doc_types=self._grounded_doc_types(perspective, doc_types),
            ),
            priority=_as_int(raw.get("priority"), 5),
        )

    @staticmethod
    def _grounded_doc_types(
        perspective: Perspective, chosen: List[EvidenceType]
    ) -> List[EvidenceType]:
        types = chosen or list(_DEFAULT_TYPES[perspective])
        if perspective in _NEWS_GROUNDED and EvidenceType.NEWS not in types:
            types = [EvidenceType.NEWS] + [t for t in types if t != EvidenceType.NEWS]
        return types

    def _default_task(self, perspective: Perspective, intent: Intent) -> TaskSpec:
        subject = ", ".join(intent.tickers) or "the subject company"
        descriptions = {
            Perspective.FUNDAMENTAL: f"Assess fundamentals for {subject}: revenue, margins, "
            "earnings quality, cash generation and valuation.",
            Perspective.NEWS: f"Identify material events, catalysts and news flow for {subject} "
            "and assess their directional impact.",
            Perspective.TECHNICAL: f"Evaluate price trend, momentum, moving-average structure "
            "and volume behaviour for {subject}.",
            Perspective.RISK: f"Assess downside risk for {subject}: concentration, regulatory "
            "exposure, valuation fragility and volatility.",
        }
        return TaskSpec(
            task_id="",
            description=descriptions[perspective],
            perspective=perspective,
            assigned_agent={
                Perspective.FUNDAMENTAL: "financial_analysis",
                Perspective.NEWS: "news_analysis",
                Perspective.TECHNICAL: "technical_analysis",
                Perspective.RISK: "risk_analysis",
            }[perspective],
            focus_metrics=list(_DEFAULT_FOCUS[perspective]),
            retrieval_queries=[descriptions[perspective][:200]],
            metadata_filters=MetadataFilters(
                tickers=list(intent.tickers),
                doc_types=list(_DEFAULT_TYPES[perspective]),
            ),
            priority=5,
        )

    # -- entry point ------------------------------------------------------
    def _in_scope(self, query: UserQuery, intent: Intent) -> bool:
        """A research run only makes sense when the question has a company in
        it. Resolved tickers are the strongest signal; an unresolved but
        ticker-shaped token (e.g., a delisted symbol) still counts so the run
        can degrade gracefully. Everything else is out of scope.

        Now supports Indian and Global companies.
        """
        if intent.tickers:
            return True
        tokens = _SYMBOL_TOKEN_RE.findall(query.raw_text)
        return any(token not in _NON_SYMBOL_WORDS for token in tokens)

    def _history(self, query: UserQuery) -> List[Dict[str, Any]]:
        raw = (query.user_context or {}).get("history")
        if not isinstance(raw, list):
            return []
        return [turn for turn in raw if isinstance(turn, dict)][-3:]

    @staticmethod
    def _prior_perspectives(context: Dict[str, Any]) -> List[Perspective]:
        values = context.get("perspectives")
        if not isinstance(values, list):
            return []
        chosen: List[Perspective] = []
        for value in values:
            try:
                perspective = Perspective(str(value).lower())
            except ValueError:
                continue
            _add(chosen, perspective)
        return chosen

    def _follow_up_in_scope(
        self, query: UserQuery, intent: Intent, context: Dict[str, Any]
    ) -> bool:
        if intent.tickers or query.tickers:
            return True
        if intent.in_scope is not None:
            return intent.in_scope
        if self._matched_perspectives(query):
            return True
        tokens = _SYMBOL_TOKEN_RE.findall(query.raw_text)
        if any(token not in _NON_SYMBOL_WORDS for token in tokens):
            return True
        return bool(_FOLLOW_UP_REFERENCE_RE.search(query.raw_text.lower()))

    def _follow_up_task(
        self,
        perspective: Perspective,
        query: UserQuery,
        intent: Intent,
        context: Dict[str, Any],
    ) -> TaskSpec:
        base = self._default_task(perspective, intent)
        subject = ", ".join(intent.tickers) or "the subject company"
        question = query.raw_text.strip()
        prior_question = str(context.get("question", "")).strip()
        description = f"Follow-up: {question}"
        if prior_question:
            description += f" (continuing from: {prior_question})"
        description += f" Focus on {perspective.value} evidence for {subject}."
        retrieval_queries = [f"{subject} {question}"]
        if prior_question:
            retrieval_queries.append(f"{subject} {prior_question}")
        return TaskSpec(
            task_id="",
            description=description[:500],
            perspective=perspective,
            assigned_agent=base.assigned_agent,
            focus_metrics=list(base.focus_metrics),
            retrieval_queries=[q[:200] for q in retrieval_queries[:3]],
            metadata_filters=MetadataFilters(
                tickers=list(intent.tickers),
                doc_types=list(base.metadata_filters.doc_types),
            ),
            priority=5,
        )

    def plan(self, query: UserQuery) -> ResearchPlan:
        history = self._history(query)
        context = history[-1] if history else None
        intent = self.analyse_intent(query, context=context)

        if context is not None:
            if not self._follow_up_in_scope(query, intent, context):
                raise OutOfScopeError(
                    "This chat is researching a company, and that question is "
                    "outside that subject. Ask about the company's filings, news, "
                    "fundamentals, valuation, risks or price action, or start a "
                    "new research run for another company."
                )
            prior_tickers = [
                str(t).strip().upper()
                for t in (
                    context.get("tickers")
                    if isinstance(context.get("tickers"), list)
                    else []
                )
                if str(t).strip()
            ]
            if not intent.tickers and not query.tickers and prior_tickers:
                intent.tickers = prior_tickers
        elif not self._in_scope(query, intent):
            raise OutOfScopeError(
                "This assistant only covers publicly traded companies. Name a "
                "company or ticker (for example NVDA, AAPL or MSFT) and ask about "
                "its filings, news, fundamentals, valuation, risks or price action."
            )

        if not intent.tickers and query.tickers:
            intent.tickers = list(query.tickers)
        if intent.tickers and not query.tickers:
            query.tickers = list(intent.tickers)

        if context is not None:
            perspectives = self._matched_perspectives(query)
            if not perspectives:
                perspectives = self._prior_perspectives(context) or [
                    Perspective.FUNDAMENTAL,
                    Perspective.NEWS,
                    Perspective.RISK,
                ]
            tasks = [
                self._follow_up_task(p, query, intent, context) for p in perspectives
            ]
            rationale_prefix = (
                "Follow-up: reused the subject and ran only the relevant lenses. "
            )
        else:
            perspectives = self.identify_perspectives(query, intent)
            tasks = self.decompose(query, intent, perspectives)
            rationale_prefix = ""

        skipped = [p for p in Perspective if p not in perspectives]
        rationale = (
            f"{rationale_prefix}"
            f"Perspectives selected: {', '.join(p.value for p in perspectives)}. "
            f"{'Omitted: ' + ', '.join(p.value for p in skipped) + '. ' if skipped else ''}"
            f"{intent.rationale}"
        )
        plan = ResearchPlan(
            plan_id="",
            query=query,
            intent=intent,
            tasks=tasks,
            decomposition_rationale=rationale.strip(),
            skipped_perspectives=skipped,
            llm_trace_id=getattr(self.llm, "name", None),
        )
        logger.info(
            "planned %s task(s) for %s: %s",
            len(plan.tasks),
            ",".join(intent.tickers) or "no ticker",
            [t.perspective.value for t in plan.tasks],
        )
        return plan


def _add(items: List[Perspective], value: Perspective) -> None:
    if value not in items:
        items.append(value)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["PlannerAgent", "OutOfScopeError"]
