"""End-to-end: the invariants a reader of the report is entitled to rely on.

These are the tests that would have caught the two worst bugs found during
development -- the graph routing straight past the verification node, and
neutral lenses being excluded from the agreement score.
"""

from __future__ import annotations

import json

import pytest

from finagents.cli import main
from finagents.reporting import render, render_json
from finagents.schemas import Stance, UserQuery, Verdict


class TestWorkflow:
    def test_run_completes(self, nvda_state):
        assert nvda_state.status in ("reported", "complete")

    def test_report_is_produced(self, nvda_state):
        assert nvda_state.report is not None
        assert nvda_state.report.sections

    def test_verification_node_actually_runs(self, nvda_state):
        """Regression guard: the graph once jumped analyse -> synthesise."""
        assert nvda_state.verifications, "the verification node was skipped"
        assert any(e["node"] == "verify" for e in nvda_state.trace)

    def test_every_node_in_the_graph_is_recorded(self, nvda_state):
        nodes = [e["node"] for e in nvda_state.trace]
        assert {"plan", "retrieve", "analyse", "verify", "synthesise", "report"} <= set(
            nodes
        )

    def test_every_claim_was_verified(self, nvda_state):
        claim_ids = {c.claim_id for f in nvda_state.findings for c in f.claims}
        verified = {v.claim_id for v in nvda_state.verifications}
        assert claim_ids == verified

    def test_report_carries_the_execution_trace(self, nvda_state):
        """A report with no trace cannot be audited after the fact."""
        nodes = [e["node"] for e in nvda_state.report.execution_trace]
        assert {"plan", "retrieve", "analyse", "verify"} <= set(nodes)

    def test_no_unverified_claim_reaches_the_report(self, nvda_state):
        rejected = {
            v.claim_id
            for v in nvda_state.verifications
            if v.verdict in (Verdict.UNSUPPORTED, Verdict.CONTRADICTED)
        }
        report = nvda_state.report
        for section in report.sections:
            assert not (set(section.claim_ids) & rejected)

    def test_findings_cite_only_retrieved_documents(self, nvda_state):
        for finding in nvda_state.findings:
            retrieved = {d.document.doc_id for d in finding.evidence}
            for claim in finding.claims:
                for citation in claim.citations:
                    assert citation.doc_id in retrieved

    def test_confidence_is_decomposed(self, nvda_state):
        confidence = nvda_state.report.confidence
        assert 0.0 <= confidence.overall <= 1.0
        assert set(confidence.components) == {
            "evidence_quality",
            "agreement",
            "coverage",
            "corroboration",
        }

    def test_agent_failure_does_not_sink_the_run(self, workflow, settings):
        """One lens failing must not lose the other lenses' work."""

        registry = workflow.registry
        original = registry.get

        def exploding(name, llm, s):
            if name == "news_analysis":
                raise RuntimeError("simulated agent crash")
            return original(name, llm, s)

        registry.get = exploding
        try:
            state = workflow.run("What are the risks to NVDA and the news flow?")
        finally:
            registry.get = original
        assert state.report is not None
        assert state.errors
        assert any(e.agent_name == "news_analysis" for e in state.errors)

    def test_run_is_deterministic(self, workflow):
        """Two identical runs must agree, or the offline mode is a lie."""
        a = workflow.run("Assess NVDA fundamentals.")
        b = workflow.run("Assess NVDA fundamentals.")
        assert a.report.thesis == b.report.thesis
        assert a.confidence.overall == b.confidence.overall
        assert len(a.findings) == len(b.findings)

    def test_sequential_and_parallel_agree(self, settings, llm, provider):
        """Concurrency must not change the answer."""
        from finagents.workflow import FinancialResearchWorkflow

        seq = FinancialResearchWorkflow(
            settings.model_copy(update={"parallel_agents": False}),
            llm=llm,
            provider=provider,
        )
        par = FinancialResearchWorkflow(
            settings.model_copy(update={"parallel_agents": True}),
            llm=llm,
            provider=provider,
        )
        question = "What are the risks to NVDA?"
        assert sorted(f.task_id for f in seq.run(question).findings) == sorted(
            f.task_id for f in par.run(question).findings
        )

    def test_unknown_ticker_still_answers(self, workflow):
        """A question about a company not in the corpus must degrade, not crash."""
        state = workflow.run("What are the risks facing ZZZZ?")
        assert state.status in ("reported", "complete")

    def test_follow_up_reuses_the_ticker_and_runs_one_lens(self, workflow):
        """A follow-up in a chat carries the subject forward. It must not need
        the ticker in the text, and it must not fan out to every specialist."""
        history = [
            {
                "question": "What are the main risks facing NVDA?",
                "tickers": ["NVDA"],
                "perspectives": ["fundamental", "news", "risk"],
                "thesis": "Demand is strong, but export controls are a risk.",
            }
        ]
        state = workflow.run(
            UserQuery(
                raw_text="What about the margins?",
                user_context={"history": history},
            )
        )
        assert state.query.tickers == ["NVDA"]
        assert [t.perspective.value for t in state.plan.tasks] == ["fundamental"]
        assert state.report is not None


class TestReport:
    def test_report_renders_as_text(self, nvda_state):
        text = render(nvda_state.report)
        assert "THESIS" in text
        assert "EVIDENCE AND SOURCES" in text
        assert "Not investment advice" in text

    def test_report_renders_as_json(self, nvda_state):
        payload = json.loads(render_json(nvda_state.report))
        assert payload["thesis"]
        assert payload["confidence"]["overall"] >= 0

    def test_report_shows_confidence_breakdown(self, nvda_state):
        text = render(nvda_state.report)
        for label in (
            "Evidence quality",
            "Lens agreement",
            "Coverage",
            "Corroboration",
        ):
            assert label in text

    def test_report_lists_every_retrieved_document(self, nvda_state):
        text = render(nvda_state.report)
        for scored in nvda_state.report.evidence:
            assert scored.document.doc_id in text

    def test_report_does_not_leak_pipeline_bookkeeping(self, nvda_state):
        """`channel_*` and `event_count_*` are debug signals, not financials."""
        text = render(nvda_state.report)
        assert "channel_customer_concentration" not in text
        assert "event_count_product" not in text
        assert "document_count" not in text

    def test_report_names_its_data_sources(self, nvda_state):
        """A report that does not say where its numbers came from asks to be
        trusted on faith. The reader is entitled to know the primary source."""
        report = nvda_state.report
        assert report.data_sources
        joined = " ".join(report.data_sources.values()).lower()
        assert "sec" in joined or "edgar" in joined

    def test_report_discloses_aggregated_news(self, nvda_state):
        """Google News is an aggregator, and the report must not imply otherwise."""
        text = render(nvda_state.report)
        assert "DATA SOURCES" in text
        assert "google news" in text.lower()

    def test_report_never_claims_to_be_synthetic(self, nvda_state):
        """The product has no synthetic corpus; nothing may suggest it does."""
        report = nvda_state.report
        blob = " ".join(
            [report.thesis, *report.limitations, *report.data_sources.values()]
        ).lower()
        assert "synthetic" not in blob
        assert "offline corpus" not in blob


class TestCli:
    @pytest.fixture
    def cli_workflow(self, monkeypatch, workflow):
        """Point the CLI at the fixture workflow.

        The CLI builds its own workflow from settings, which means it would
        demand a live Ollama daemon. What these tests are about is argument
        handling and rendering, so the construction seam is patched rather than
        the LLM inside it.
        """
        monkeypatch.setattr(
            "finagents.cli.FinancialResearchWorkflow", lambda *a, **k: workflow
        )
        return workflow

    def test_requires_a_question(self, capsys):
        assert main([]) == 2

    def test_a_missing_llm_explains_itself(self, monkeypatch, capsys):
        """A missing daemon is a configuration problem, not a crash.

        `build_llm_client` deliberately raises rather than degrading, so the CLI
        has to catch it. Otherwise the first thing a new user sees for the most
        likely misconfiguration - no Ollama running - is a traceback.
        """
        from finagents.llm.base import LLMError

        def _boom(*_a, **_k):
            raise LLMError("Ollama is unavailable at http://localhost:11434")

        monkeypatch.setattr("finagents.cli.FinancialResearchWorkflow", _boom)
        assert main(["what are NVDA's risks?"]) == 1
        err = capsys.readouterr().err
        assert "Ollama is unavailable" in err
        assert "ollama serve" in err

    def test_doctor_reports_provider(self, capsys):
        main(["--doctor"])
        out = capsys.readouterr().out
        assert "provider available" in out
        assert "ollama endpoint" in out

    def test_doctor_names_the_live_data_sources(self, capsys):
        main(["--doctor"])
        assert "SEC EDGAR" in capsys.readouterr().out

    def test_prints_a_report(self, capsys, cli_workflow):
        code = main(["What are the risks to NVDA?"])
        assert "MULTI-AGENT FINANCIAL RESEARCH REPORT" in capsys.readouterr().out
        assert code == 0

    def test_json_output_is_parseable(self, capsys, cli_workflow):
        main(["--json", "What are the risks to NVDA?"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["overall_stance"] in {s.value for s in Stance}

    def test_ticker_flag_is_honoured(self, capsys, cli_workflow):
        main(["--json", "--ticker", "AAPL", "How is AAPL doing?"])
        assert json.loads(capsys.readouterr().out)["query"]["tickers"] == ["AAPL"]

    def test_a_ticker_flag_restricts_rather_than_unions(self, capsys, cli_workflow):
        """`--ticker NVDA` means "only NVDA", even if the text names another."""
        main(["--json", "--ticker", "AAPL", "How is NVDA doing?"])
        assert json.loads(capsys.readouterr().out)["query"]["tickers"] == ["AAPL"]

    def test_an_explicit_ticker_flag_is_canonicalised(self, capsys, cli_workflow):
        """The report must name the security it actually read.

        `BRK.B` reaches EDGAR as an unknown symbol, so filings come back empty
        while the report still claims Berkshire was analysed.
        """
        main(["--json", "--ticker", "BRK.B", "What are the risks?"])
        assert json.loads(capsys.readouterr().out)["query"]["tickers"] == ["BRK-B"]

    def test_offline_flags_are_gone(self, capsys):
        """`--provider offline` and `--data-source` must not come back."""
        with pytest.raises(SystemExit):
            main(["--provider", "offline", "anything"])
        assert "unrecognized arguments" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            main(["--data-source", "offline", "anything"])
        assert "unrecognized arguments" in capsys.readouterr().err

    def test_the_provider_flag_is_gone_entirely(self, capsys):
        """Ollama is the only LLM, so `--provider ollama` was a no-op.

        A flag that accepts exactly one value and ignores it is worse than no
        flag: it implies a choice that does not exist. It must be rejected.
        """
        with pytest.raises(SystemExit):
            main(["--provider", "ollama", "anything"])
        assert "unrecognized arguments" in capsys.readouterr().err
