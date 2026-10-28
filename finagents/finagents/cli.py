"""Command-line entry point.

python -m finagents "What are the main risks facing NVDA?"
python -m finagents --json --ticker NVDA "Is the valuation justified?"
python -m finagents --doctor          # check the LLM provider
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .agents.planner import OutOfScopeError
from .llm import LLMError, build_llm_client
from .reporting import render, render_json
from .schemas import Settings, UserQuery
from .workflow import FinancialResearchWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finagents",
        description="Multi-Agent Financial Reasoning and Decision Support System.",
    )
    parser.add_argument("query", nargs="*", help="The financial question to research.")
    parser.add_argument(
        "--ticker", action="append", default=[], help="Restrict to a ticker."
    )
    parser.add_argument(
        "--model", help="Ollama model name (default: settings/llama3.1)."
    )
    parser.add_argument("--top-k", type=int, help="Documents retrieved per task.")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON."
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the verification layer."
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Run agents serially."
    )
    parser.add_argument(
        "--doctor", action="store_true", help="Check provider health and exit."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    overrides = {
        "enable_verification": not args.no_verify,
        "parallel_agents": not args.sequential,
    }
    if args.model:
        overrides["ollama_model"] = args.model
    if args.top_k:
        overrides["retrieval_per_task_k"] = args.top_k
    settings = Settings.from_env(**overrides)

    if args.doctor:
        return _doctor(settings)

    question = " ".join(args.query).strip()
    if not question:
        print(
            'error: provide a question, e.g. finagents "What are NVDA\'s main risks?"',
            file=sys.stderr,
        )
        return 2

    query = UserQuery(raw_text=question, tickers=[t.upper() for t in args.ticker])
    try:
        workflow = FinancialResearchWorkflow(settings)
    except LLMError as exc:
        # A missing LLM is a configuration problem, not a bug. Print what to do
        # about it instead of a traceback: the caller gets exit code 1 either way,
        # but only one of them tells them how to fix it.
        print(f"error: {exc}", file=sys.stderr)
        print(_OLLAMA_HELP.format(model=settings.ollama_model), file=sys.stderr)
        return 1
    try:
        state = workflow.run(query)
    except OutOfScopeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if state.report is None:
        print(
            f"error: workflow produced no report (status={state.status})",
            file=sys.stderr,
        )
        for error in state.errors:
            print(f"  {error.agent_name}: {error.error}", file=sys.stderr)
        return 1

    print(render_json(state.report) if args.as_json else render(state.report))
    # Non-zero exit signals a completed-but-untrustworthy run, so a caller can
    # gate on it without parsing the report.
    return 0 if state.report.confidence.overall >= 0.3 else 3


def _doctor(settings: Settings) -> int:
    print(f"ollama endpoint     : {settings.ollama_base_url}")
    print(f"ollama model        : {settings.ollama_model}")
    print(f"data sources        : SEC EDGAR, Google News RSS, Yahoo Finance")
    try:
        client = build_llm_client(settings)
    except LLMError as exc:
        print(f"provider available  : False")
        print(f"detail              : {exc}")
        print(_OLLAMA_HELP.format(model=settings.ollama_model), file=sys.stderr)
        return 1
    health = client.health()
    print(f"provider available  : {health.get('available')}")
    if health.get("detail"):
        print(f"detail              : {health['detail']}")
    if health.get("installed_models"):
        print(f"installed models    : {', '.join(health['installed_models'])}")
    if not health.get("available"):
        print(_OLLAMA_HELP.format(model=settings.ollama_model), file=sys.stderr)
        return 1
    return 0


_OLLAMA_HELP = """
To enable model-backed reasoning:
  1. Install Ollama from https://ollama.com/download
  2. ollama serve
  3. ollama pull {model}
"""


if __name__ == "__main__":
    raise SystemExit(main())
