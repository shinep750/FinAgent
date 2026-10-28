"""The JSON contract, the JSON parser, and the retry path.

Two separate concerns are tested here. `extract_json` and `generate_json` are
product code: local models wrap JSON in prose and fences constantly, so the
forgiveness of the parser is a real correctness property. `FakeLLMClient` is a
test double, but it is treated as an API rather than a stub - the pipeline's
correctness claims only hold if the client it is tested against honours the same
contract Ollama does.
"""

from __future__ import annotations

import json
from typing import List, Optional

import pytest

from finagents.llm.base import LLMClient, LLMError, build_prompt, extract_json
from finagents.llm.factory import build_llm_client
from finagents.prompts import role_header
from finagents.schemas import Settings

from tests.fake_llm import FakeLLMClient, detect_role, parse_documents, parse_metrics

_EVIDENCE_BLOCK = (
    "DOC nvda-10k-fy2025 | NVIDIA CORP FY2025 10-K | source=SEC EDGAR | "
    "date=2025-02-26 | type=filing | reliability=0.95 | score=0.940\n"
    "For fiscal year 2025 total revenue reached $130.5 billion, an increase of 114 "
    "percent from the prior year. Gross margin was 75.0% for the fiscal year.\n\n"
    "DOC market-nvda-snapshot | NVIDIA CORP price statistics | source=Yahoo Finance | "
    "date=2025-02-26 | type=market_data | reliability=0.85 | score=0.610\n"
    "NVDA last traded at 181.20 with a 50-day moving average of 169.40."
)

_METRICS_BLOCK = (
    f"{_EVIDENCE_BLOCK}\n\n"
    "COMPUTED METRICS (calculated from the market data, not from the text):\n"
    "last_price=181.2, rsi_14=58.2, sma_200=151.05, volatility_60d_annualised=0.413"
)

_EXTRACT_PROMPT = build_prompt(
    system="You are a specialist analyst.",
    instruction=f"{role_header('financial')}\nAnalyse NVDA fundamentals.",
    context=_METRICS_BLOCK,
    output_schema='{"summary": "string", "claims": []}',
)


class _GarbageClient(LLMClient):
    """Returns unparseable text forever, to exercise the retry path."""

    name = "garbage"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        self.calls += 1
        return "I am afraid I cannot do that."


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_around_object(self):
        assert extract_json('Here you go: {"a": 1} hope that helps') == {"a": 1}

    def test_nested_objects(self):
        assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}

    def test_empty_response(self):
        with pytest.raises(LLMError):
            extract_json("   ")

    def test_not_json(self):
        with pytest.raises(LLMError):
            extract_json("no json here")


class TestRetryLogic:
    def test_retries_then_fails_loudly(self):
        client = _GarbageClient()
        with pytest.raises(LLMError):
            client.generate_json("prompt", "system", retries=2)
        assert client.calls == 3

    def test_recovers_on_a_later_attempt(self):
        class Flaky(_GarbageClient):
            def complete(self, prompt, system=None, **kw):
                self.calls += 1
                return "nope" if self.calls == 1 else '{"ok": true}'

        client = Flaky()
        assert client.generate_json("prompt", "system", retries=2) == {"ok": True}
        assert client.calls == 2

    def test_malformed_first_response_is_repaired(self):
        """A chatty model must not break a run; the retry has to recover."""
        client = FakeLLMClient(malformed_first=True)
        payload = client.generate_json(
            build_prompt(
                system="s",
                instruction=f"{role_header('financial')}\nAnalyse.",
                context=_METRICS_BLOCK,
            )
        )
        assert payload["claims"]


class TestFakeClientContract:
    def test_health_reports_available(self, llm):
        assert llm.health()["available"] is True

    def test_complete_returns_text(self, llm):
        assert isinstance(llm.complete("Say hello", "You are terse."), str)

    @pytest.mark.parametrize(
        "role",
        [
            "planner",
            "financial",
            "news",
            "technical",
            "risk",
            "verifier",
            "synthesizer",
        ],
    )
    def test_output_is_raw_json_not_python_repr(self, llm, role):
        """Regression guard: True/None in the output once broke json.loads."""
        raw = llm.complete(
            f"{role_header(role)}\nDo the {role} task for NVDA.", "system"
        )
        json.loads(raw)  # raises if the client emits a Python repr

    def test_role_dispatch(self, llm):
        for role in ("planner", "financial", "verifier", "synthesizer"):
            assert detect_role(f"{role_header(role)}\nGo") == role

    def test_generate_json_returns_mapping(self, llm):
        payload = llm.generate_json(_EXTRACT_PROMPT, "You are a specialist analyst.")
        assert isinstance(payload, dict)
        assert payload[
            "claims"
        ], "claims should be extracted from the supplied evidence"
        claim = payload["claims"][0]
        assert claim["citations"][0]["doc_id"] == "nvda-10k-fy2025"

    def test_computed_metrics_are_carried_through(self, llm):
        """Metrics come from the price series, not from the document text."""
        payload = llm.generate_json(_EXTRACT_PROMPT, "You are a specialist analyst.")
        assert payload["key_metrics"]["rsi_14"] == 58.2
        assert payload["key_metrics"]["volatility_60d_annualised"] == 0.413

    def test_no_evidence_yields_no_claims(self, llm):
        """The client must not invent a figure that is not in the evidence.

        This is the property that makes test runs trustworthy: the deterministic
        client can only report what it was shown, so a failure here means
        retrieval or parsing broke, not hallucination.
        """
        bare = build_prompt(
            system="s",
            instruction=f"{role_header('financial')}\nAnalyse NVDA fundamentals.",
            context="NVDA did very well last year and the outlook is bright.",
        )
        payload = llm.generate_json(bare, "s")
        assert payload["claims"] == []

    def test_citations_reference_only_supplied_documents(self, llm):
        payload = llm.generate_json(_EXTRACT_PROMPT, "system")
        supplied = {d["doc_id"] for d in parse_documents(_EXTRACT_PROMPT)}
        cited = {c["doc_id"] for claim in payload["claims"] for c in claim["citations"]}
        assert cited <= supplied

    def test_prompt_has_stable_sections(self):
        prompt = build_prompt(
            system="s", instruction="i", context="c", output_schema="o"
        )
        assert prompt.index("### INSTRUCTION") < prompt.index("### CONTEXT")
        assert prompt.index("### CONTEXT") < prompt.index("### REQUIRED OUTPUT FORMAT")
        assert prompt.rstrip().endswith("No prose, no markdown fences.")

    def test_parse_metrics_ignores_malformed_pairs(self):
        assert parse_metrics("COMPUTED METRICS:\na=1, broken, b=2") == {
            "a": 1.0,
            "b": 2.0,
        }


class TestFactory:
    def test_unreachable_ollama_is_fatal(self, settings: Settings):
        """No fallback client exists, so a dead daemon must stop the run.

        The previous behaviour - warn and return a deterministic client - would
        let a financial report be produced with no model behind it. That is the
        failure mode this project exists to avoid, so it is now an error.
        """
        dead = settings.model_copy(update={"ollama_base_url": "http://127.0.0.1:1"})
        with pytest.raises(LLMError) as excinfo:
            build_llm_client(dead)
        message = str(excinfo.value)
        assert "ollama serve" in message, "the error must say how to fix it"
        assert "no offline fallback" in message.lower()

    def test_error_names_both_models_needed(self, settings: Settings):
        dead = settings.model_copy(update={"ollama_base_url": "http://127.0.0.1:1"})
        with pytest.raises(LLMError) as excinfo:
            build_llm_client(dead)
        assert settings.ollama_model in str(excinfo.value)


class TestHealthModelTagging:
    """Ollama tags every model name; users configure the bare name."""

    def test_a_pulled_model_is_recognised_despite_the_tag(self):
        from finagents.llm.ollama_client import _has_model

        installed = ["nomic-embed-text:latest", "llama3.1:latest"]
        assert _has_model(installed, "llama3.1") is True
        assert _has_model(installed, "nomic-embed-text") is True

    def test_an_explicit_tag_still_matches(self):
        from finagents.llm.ollama_client import _has_model

        assert _has_model(["llama3.1:8b"], "llama3.1:8b") is True

    def test_a_prefix_is_not_a_match(self):
        """`llama` must not be satisfied by `llama3.1:latest`."""
        from finagents.llm.ollama_client import _has_model

        assert _has_model(["llama3.1:latest"], "llama") is False
        assert _has_model(["llama3.1:latest"], "llama3") is False

    def test_a_model_that_is_genuinely_absent_is_reported_missing(self):
        from finagents.llm.ollama_client import _has_model

        assert _has_model(["llama3.1:latest"], "mistral") is False

    def test_health_reports_a_pulled_model_as_available(
        self, settings: Settings, monkeypatch
    ):
        """Regression: `llama3.1` was reported missing while `llama3.1:latest`
        was installed, sending the user to re-download 4.9GB."""

        from finagents.llm.ollama_client import OllamaClient

        class _StubResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"models": [{"name": "llama3.1:latest"}]}

        class _StubClient:
            def get(self, *_a, **_k):
                return _StubResponse()

        client = OllamaClient(settings)
        monkeypatch.setattr(client, "_client", _StubClient())
        health = client.health()
        assert health["available"] is True
        assert health["detail"] is None
        assert health["installed_models"] == ["llama3.1:latest"]

    def test_error_names_both_models_needed(self, settings: Settings):
        dead = settings.model_copy(update={"ollama_base_url": "http://127.0.0.1:1"})
        with pytest.raises(LLMError) as excinfo:
            build_llm_client(dead)
        assert settings.ollama_embedding_model in str(excinfo.value)
