"""Agent behaviour: routing, citation discipline, and no-evidence handling.

The rule under test throughout is that an agent may not assert anything it
cannot cite. Everything else is secondary.
"""

from __future__ import annotations

import pytest

from finagents.schemas import EvidenceType, Perspective, Stance, UserQuery
from finagents.agents.planner import OutOfScopeError

from tests.fake_llm import FakeLLMClient


@pytest.fixture(scope="module")
def planner(workflow):
    return workflow.planner


class TestPlanner:
    def test_valuation_question_selects_fundamentals(self, planner):
        plan = planner.plan(UserQuery(raw_text="Is NVDA's valuation justified?"))
        assert Perspective.FUNDAMENTAL in {t.perspective for t in plan.tasks}

    def test_chart_question_selects_technical(self, planner):
        plan = planner.plan(UserQuery(raw_text="What is NVDA's technical setup?"))
        assert Perspective.TECHNICAL in {t.perspective for t in plan.tasks}

    def test_risk_question_selects_risk(self, planner):
        plan = planner.plan(UserQuery(raw_text="What are the risks to NVDA?"))
        assert Perspective.RISK in {t.perspective for t in plan.tasks}

    def test_fundamental_and_risk_plans_always_retrieve_news(self, planner):
        """Regression: a filing-only risk view misses what the market is
        repricing. Fundamental and risk tasks keep a news channel even when the
        model's own filters omit it."""
        for question in (
            "Assess NVDA fundamentals.",
            "What are the risks to NVDA?",
        ):
            plan = planner.plan(UserQuery(raw_text=question))
            relevant = [
                t
                for t in plan.tasks
                if t.perspective in (Perspective.FUNDAMENTAL, Perspective.RISK)
            ]
            assert relevant, question
            for task in relevant:
                assert EvidenceType.NEWS in task.metadata_filters.doc_types

    def test_news_is_forced_even_when_the_model_omits_it(self, planner):
        types = planner._grounded_doc_types(Perspective.RISK, [EvidenceType.REGULATORY])
        assert types == [EvidenceType.NEWS, EvidenceType.REGULATORY]
        assert EvidenceType.NEWS in planner._grounded_doc_types(
            Perspective.FUNDAMENTAL, [EvidenceType.FILING]
        )
        assert EvidenceType.NEWS in planner._grounded_doc_types(
            Perspective.FUNDAMENTAL, []
        )

    def test_a_news_task_never_loses_its_news_channel(self, planner):
        """Regression: a news-perspective task whose model-supplied filter said
        "filing" retrieved nothing at all against a news-only corpus (EDGAR was
        unreachable), and the whole run produced an empty report. A news task
        must always be able to retrieve news."""
        types = planner._grounded_doc_types(Perspective.NEWS, [EvidenceType.FILING])
        assert EvidenceType.NEWS in types
        assert types[0] == EvidenceType.NEWS

    def test_every_task_is_assigned_to_a_known_agent(self, planner, workflow):
        plan = planner.plan(UserQuery(raw_text="Assess NVDA end to end."))
        for task in plan.tasks:
            workflow.registry.get(task.assigned_agent, workflow.llm, workflow.settings)

    def test_task_ids_are_unique(self, planner):
        plan = planner.plan(UserQuery(raw_text="Assess NVDA end to end."))
        ids = [t.task_id for t in plan.tasks]
        assert len(ids) == len(set(ids))

    def test_tasks_carry_retrieval_queries(self, planner):
        plan = planner.plan(UserQuery(raw_text="Assess NVDA fundamentals."))
        assert all(t.retrieval_queries for t in plan.tasks)

    def test_plan_records_skipped_perspectives(self, planner):
        plan = planner.plan(UserQuery(raw_text="What is NVDA's technical setup?"))
        assert Perspective.FUNDAMENTAL in {p for p in plan.skipped_perspectives}

    def test_non_equity_question_is_out_of_scope(self, planner):
        """The assistant is company research, not a general chatbot. A query
        with no resolvable public company must be rejected, not answered from
        unrelated evidence."""
        with pytest.raises(OutOfScopeError):
            planner.plan(UserQuery(raw_text="Write a poem about the sea."))

    def test_follow_up_reuses_subject_and_skips_decomposition(self, planner):
        """A follow-up about a known subject must not redo the whole plan: one
        intent call, no second decomposition call, only the requested lens."""
        history = [
            {
                "question": "What are the main risks facing NVDA?",
                "tickers": ["NVDA"],
                "perspectives": ["fundamental", "news", "risk"],
                "thesis": "Demand is strong, but export controls are a risk.",
            }
        ]
        before = len(planner.llm.calls)
        plan = planner.plan(
            UserQuery(
                raw_text="What about the margins?",
                user_context={"history": history},
            )
        )
        assert len(planner.llm.calls) - before == 1
        assert plan.intent.tickers == ["NVDA"]
        assert plan.query.tickers == ["NVDA"]
        assert [t.perspective for t in plan.tasks] == [Perspective.FUNDAMENTAL]

    def test_follow_up_without_a_lens_reuses_the_prior_lenses(self, planner):
        history = [
            {
                "question": "What are the main risks facing NVDA?",
                "tickers": ["NVDA"],
                "perspectives": ["fundamental", "news", "risk"],
                "thesis": "Demand is strong, but export controls are a risk.",
            }
        ]
        plan = planner.plan(
            UserQuery(raw_text="Why?", user_context={"history": history})
        )
        assert plan.intent.tickers == ["NVDA"]
        assert {t.perspective for t in plan.tasks} == {
            Perspective.FUNDAMENTAL,
            Perspective.NEWS,
            Perspective.RISK,
        }

    def test_follow_up_can_name_a_new_subject(self, planner):
        history = [
            {
                "question": "What are the main risks facing NVDA?",
                "tickers": ["NVDA"],
                "perspectives": ["fundamental", "news", "risk"],
                "thesis": "Demand is strong, but export controls are a risk.",
            }
        ]
        plan = planner.plan(
            UserQuery(raw_text="What about AMD?", user_context={"history": history})
        )
        assert plan.intent.tickers == ["AMD"]
        assert all(t.metadata_filters.tickers == ["AMD"] for t in plan.tasks)

    def test_off_topic_follow_up_is_rejected(self, planner):
        history = [
            {
                "question": "What are the main risks facing NVDA?",
                "tickers": ["NVDA"],
                "perspectives": ["fundamental", "news", "risk"],
                "thesis": "Demand is strong, but export controls are a risk.",
            }
        ]
        with pytest.raises(OutOfScopeError):
            planner.plan(
                UserQuery(
                    raw_text="What's the weather tomorrow?",
                    user_context={"history": history},
                )
            )

    def test_ticker_is_resolved_from_text(self, planner):
        assert "NVDA" in planner.resolve_tickers(
            UserQuery(raw_text="How is NVDA doing?")
        )

    def test_english_words_that_happen_to_be_tickers_are_not_tickers(self):
        """EDGAR lists ~500 ordinary words as symbols. Scanning an upper-cased
        copy of the query resolves them, and the report then reasons about the
        wrong company using the right company's filings.

        `ARE` is Alexandria Real Estate Equities and `MAIN` is Main Street
        Capital, both real filers. "What are the main risks facing NVDA?" must
        not pick them up.
        """
        planner = self._planner_for({"ARE", "MAIN", "NVDA", "AAPL", "BRK-B"})
        found = planner.resolve_tickers(
            UserQuery(raw_text="What are the main risks facing NVDA?")
        )
        assert found == ["NVDA"], found

    def test_a_dollar_sign_makes_a_ticker_at_any_casing(self):
        """`$nvda` is unambiguous; a bare `nvda` is a word until the model
        names the security."""
        planner = self._planner_for({"NVDA", "AAPL", "BRK-B"})
        assert planner.resolve_tickers(UserQuery(raw_text="is $nvda cheap?")) == [
            "NVDA"
        ]
        assert planner.resolve_tickers(UserQuery(raw_text="is nvda cheap?")) == []

    def test_share_classes_resolve_from_bare_text(self):
        planner = self._planner_for({"NVDA", "BRK-B"})
        assert planner.resolve_tickers(UserQuery(raw_text="how is BRK-B trading?")) == [
            "BRK-B"
        ]
        assert planner.resolve_tickers(UserQuery(raw_text="how is BRK.B trading?")) == [
            "BRK-B"
        ]

    def test_caller_supplied_tickers_are_resolved_and_deduplicated(self):
        planner = self._planner_for({"NVDA", "BRK-B"})
        found = planner.resolve_tickers(
            UserQuery(
                raw_text="compare these", tickers=["nvda", "NVDA", "brk.b", "ZZZZ"]
            )
        )
        assert found == ["NVDA", "BRK-B"], found

    @staticmethod
    def _planner_for(known):
        """A planner over a fixed EDGAR map, so resolution is testable offline.

        Uses the real `OnlineProvider.resolve_ticker` against a map-only SEC
        stub, so share-class normalisation is exercised rather than re-implemented
        here with different rules.
        """
        from finagents.agents.planner import PlannerAgent
        from finagents.data.providers import OnlineProvider
        from finagents.schemas import Settings

        class _MapOnlySEC:
            def ticker_map(self):
                return {symbol: (1, symbol) for symbol in known}

            def is_known_ticker(self, symbol):
                return symbol in known

            def close(self):
                pass

        provider = OnlineProvider(sec=_MapOnlySEC())
        return PlannerAgent(FakeLLMClient(), Settings(), provider)

    def test_no_tasks_is_impossible(self, planner):
        """A plan with zero tasks would silently produce an empty report."""
        plan = planner.plan(UserQuery(raw_text="NVDA"))
        assert plan.tasks


class TestAgentCitations:
    def _run(self, workflow, task):
        hits = workflow.retriever.search_for_task(task, top_k=4)
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        return agent.run(task, hits)

    @pytest.mark.parametrize(
        "question",
        [
            "Analyse NVDA fundamentals.",
            "What is the NVDA news flow?",
            "Describe NVDA's technical setup.",
            "What are the risks to NVDA?",
        ],
    )
    def test_every_claim_cites_retrieved_evidence(self, workflow, question):
        plan = workflow.planner.plan(UserQuery(raw_text=question))
        retrieved = {
            d.document.doc_id
            for t in plan.tasks
            for d in workflow.retriever.search_for_task(t, top_k=4)
        }
        for task in plan.tasks:
            finding = self._run(workflow, task)
            for claim in finding.claims:
                assert claim.citations, f"{claim.claim_id} has no citation"
                for citation in claim.citations:
                    assert (
                        citation.doc_id in retrieved
                    ), f"{claim.claim_id} cites {citation.doc_id}, which was not retrieved"

    def test_agent_refuses_to_claim_without_evidence(self, workflow):
        """No evidence in, no claims out -- never a plausible invention."""
        plan = workflow.planner.plan(UserQuery(raw_text="Analyse NVDA fundamentals."))
        task = plan.tasks[0]
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        finding = agent.run(task, [])
        assert finding.claims == []
        assert finding.confidence <= 0.3

    def test_finding_stance_is_consistent_with_its_claims(self, workflow):
        plan = workflow.planner.plan(UserQuery(raw_text="Analyse NVDA fundamentals."))
        task = plan.tasks[0]
        hits = workflow.retriever.search_for_task(task, top_k=4)
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        finding = agent.run(task, hits)
        stances = {c.stance for c in finding.claims}
        assert finding.stance in stances or finding.stance == Stance.NEUTRAL

    def test_technical_agent_uses_computed_indicators(self, workflow):
        """Indicators come from the price series, not from document prose.

        The agent prefixes them with the ticker, because a multi-ticker run that
        emitted two bare `sma_50` keys would silently overwrite one with the
        other.
        """
        plan = workflow.planner.plan(
            UserQuery(raw_text="What is NVDA's technical setup?")
        )
        task = next(t for t in plan.tasks if t.perspective == Perspective.TECHNICAL)
        hits = workflow.retriever.search_for_task(task, top_k=4)
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        finding = agent.run(task, hits)
        indicators = {k for k in finding.key_metrics}
        assert indicators, "the technical agent reported no computed metrics"
        assert all(k.startswith("nvda_") for k in indicators), sorted(indicators)
        assert {"nvda_rsi_14", "nvda_sma_50", "nvda_sma_200"} <= indicators

    def test_indicators_come_from_real_bars(self, workflow, provider):
        """The agent's numbers must match the provider's, not a model's memory."""
        from finagents.data.providers import compute_indicators

        plan = workflow.planner.plan(
            UserQuery(raw_text="What is NVDA's technical setup?")
        )
        task = next(t for t in plan.tasks if t.perspective == Perspective.TECHNICAL)
        hits = workflow.retriever.search_for_task(task, top_k=4)
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        finding = agent.run(task, hits)
        expected = compute_indicators("NVDA", provider)
        assert finding.key_metrics["nvda_last_price"] == pytest.approx(
            expected["last_price"], rel=1e-6
        )
        assert finding.key_metrics["nvda_rsi_14"] == pytest.approx(
            expected["rsi_14"], rel=1e-3
        )


class TestAgentCaveats:
    def _finding(self, workflow, payload):
        plan = workflow.planner.plan(UserQuery(raw_text="Analyse NVDA fundamentals."))
        task = plan.tasks[0]
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        return agent.coerce(task, payload, [], {})

    def test_an_uncited_claim_is_not_blamed_on_dangling_citations(self, workflow):
        """Regression: coverage correctly distinguishes two failures.

        The old code appended "claim without citation" entries to the
        dangling-citation list, so a model that merely returned claims with no
        citations produced a caveat accusing the extractor of citing documents
        that were never retrieved.
        """
        finding = self._finding(
            workflow, {"claims": [{"text": "Revenue concentration is a risk."}]}
        )
        caveats = " ".join(finding.caveats)
        assert "returned without a citable reference" in caveats
        assert "absent from the retrieved" not in caveats
        assert "named documents" not in caveats

    def test_a_hallucinated_document_id_is_reported_as_dangling(self, workflow):
        finding = self._finding(
            workflow,
            {
                "claims": [
                    {
                        "text": "Revenue concentration is a risk.",
                        "citations": [{"doc_id": "ghost-doc", "quote": "..."}],
                    }
                ]
            },
        )
        caveats = " ".join(finding.caveats)
        assert "named documents absent from the retrieved" in caveats
        assert "ghost-doc" in caveats


class TestRiskMetrics:
    def test_volatility_is_reported_in_percent_units(self, workflow, provider):
        """Regression: the renderer labels *_volatility as (%), but the risk
        agent emitted the raw annualised fraction, so 0.3737 displayed as
        0.3737 (%) instead of 37.37 (%)."""
        from finagents.data.providers import compute_indicators

        plan = workflow.planner.plan(UserQuery(raw_text="What are the risks to NVDA?"))
        task = next(t for t in plan.tasks if t.perspective == Perspective.RISK)
        hits = workflow.retriever.search_for_task(task, top_k=4)
        agent = workflow.registry.get(
            task.assigned_agent, workflow.llm, workflow.settings
        )
        finding = agent.run(task, hits)
        expected = compute_indicators("NVDA", provider)["volatility_60d_annualised"]
        assert "nvda_volatility" in finding.key_metrics
        assert finding.key_metrics["nvda_volatility"] == pytest.approx(
            expected * 100.0, rel=1e-6
        )
