"""Local web front end for FinAgents.

A run takes minutes on a CPU-only box, so `/ask` never blocks: it registers a
job and returns its id immediately. The browser polls `/jobs/{id}` until the
report is ready.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from finagents.agents.planner import OutOfScopeError
from finagents.llm import LLMError
from finagents.reporting import render
from finagents.schemas import Settings, UserQuery
from finagents.workflow import FinancialResearchWorkflow

logger = logging.getLogger("finagents.api")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s: %(message)s",
)

_WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="FinAgents API")
app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
# One worker on purpose. Ollama runs the model on the CPU, so a second
# concurrent job does not run faster - it starves the first one and pushes the
# retrieval HTTP calls (20s timeout) past their deadline, which yields an empty
# report. Queued jobs are honest about the wait; parallel ones silently fail.
_executor = ThreadPoolExecutor(max_workers=1)


class JobCancelled(Exception):
    """Raised inside a workflow node when the caller asked the run to stop."""


def _is_cancelled(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return bool(job and job.get("cancel_requested"))


def _check_cancel(job_id: str) -> None:
    if _is_cancelled(job_id):
        raise JobCancelled(job_id)


def _settings(no_verify: bool, sequential: bool) -> Settings:
    # Env vars are essential here: the SEC user agent and cache dir live in
    # FINAGENTS_*, and Settings() would silently ignore every one of them.
    return Settings.from_env(
        enable_verification=not no_verify,
        parallel_agents=not sequential,
    )


def _set_stage(job_id: str, stage: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["stage"] = stage


def _instrument(workflow: FinancialResearchWorkflow, job_id: str) -> None:
    """Report progress by wrapping the node methods before `run()` builds the graph.

    `build()` resolves `self.node_*` into the compiled graph, so the instance
    attributes have to be replaced before the first `run()` call.
    """
    for stage in ("plan", "retrieve", "analyse", "verify", "synthesise", "report"):
        original: Callable[..., Any] = getattr(workflow, f"node_{stage}")

        def wrapped(
            state: Any, _orig: Callable[..., Any] = original, _stage: str = stage
        ) -> Any:
            _check_cancel(job_id)
            _set_stage(job_id, _stage)
            return _orig(state)

        setattr(workflow, f"node_{stage}", wrapped)
    # The node wrapper only runs between stages; this lets long fan-out stages
    # (retrieve/analyse) abort between tasks instead of after the whole stage.
    workflow.cancel_check = lambda: _check_cancel(job_id)


def _clean_history(raw: Any) -> list:
    """Keep only the compact context a follow-up needs.

    The browser sends the last few successful turns. Anything malformed is
    dropped rather than trusted: this data ends up in a model prompt. The list
    is capped so a chatty client cannot make the prompt unbounded.
    """
    if not isinstance(raw, list):
        return []
    turns: list = []
    for item in raw[-3:]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()[:2000]
        if not question:
            continue
        raw_tickers = item.get("tickers")
        tickers = [
            str(t).strip().upper()[:12]
            for t in (raw_tickers if isinstance(raw_tickers, list) else [])
            if str(t).strip()
        ][:8]
        raw_perspectives = item.get("perspectives")
        perspectives = [
            str(p).strip().lower()
            for p in (raw_perspectives if isinstance(raw_perspectives, list) else [])
            if str(p).strip().lower() in ("fundamental", "news", "technical", "risk")
        ][:4]
        turns.append(
            {
                "question": question,
                "tickers": tickers,
                "perspectives": perspectives,
                "thesis": str(item.get("thesis", "")).strip()[:1500],
            }
        )
    return turns


def _run_job(
    job_id: str,
    question: str,
    ticker: Optional[str],
    no_verify: bool,
    sequential: bool,
    history: Optional[list] = None,
    thread_id: Optional[str] = None,
) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        if job.get("cancel_requested"):
            job.update(
                {
                    "state": "cancelled",
                    "stage": "cancelled",
                    "status": "cancelled",
                    "detail": "Stopped by user",
                }
            )
            return
        job["state"] = "running"
    try:
        settings = _settings(no_verify, sequential)
        workflow = FinancialResearchWorkflow(settings)
        _instrument(workflow, job_id)
        query = UserQuery(
            raw_text=question,
            tickers=[ticker.upper()] if ticker else [],
            user_context={"history": history} if history else {},
        )
        # A ticker-less query resolves symbols inside planner before the graph
        # starts; without this the UI would claim the job is still "queued"
        # during that first (slow) model call.
        _set_stage(job_id, "plan")
        state = workflow.run(query, thread_id=thread_id)
        result: Dict[str, Any]

        if state.report is None:
            result = {
                "status": state.status,
                "errors": [e.model_dump(mode="json") for e in state.errors],
            }
        else:
            result = {
                "status": state.status,
                "report": state.report.model_dump(mode="json"),
                "text": render(state.report),
            }
        result["tickers"] = list(state.query.tickers)
        result["perspectives"] = [
            task.perspective.value for task in (state.plan.tasks if state.plan else [])
        ]
        result["follow_up"] = bool(history)
        with _jobs_lock:
            _jobs[job_id].update(result)
            _jobs[job_id]["state"] = "done"
    except JobCancelled:
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "state": "cancelled",
                    "stage": "cancelled",
                    "status": "cancelled",
                    "detail": "Stopped by user",
                }
            )
    except OutOfScopeError as exc:
        with _jobs_lock:
            _jobs[job_id].update(
                {
                    "state": "error",
                    "status": "out_of_scope",
                    "detail": str(exc),
                }
            )
    except LLMError as exc:
        with _jobs_lock:
            _jobs[job_id].update(
                {"state": "error", "status": "llm_error", "detail": str(exc)}
            )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the caller
        logger.exception("job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(
                {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
            )


def _start(
    question: str,
    ticker: Optional[str],
    no_verify: bool,
    sequential: bool,
    history: Optional[list] = None,
) -> str:
    job_id = uuid.uuid4().hex
    # Find the last successful job to use as a thread_id for follow-ups
    thread_id = None
    if history:
        # History contains turns; we try to find the most recent job_id from it
        # but the current history structure is a list of turns, not jobs.
        # For now, we'll use the current session's job_id as a thread if provided.
        # However, we can simply use the job_id of the last turn if we tracked it.
        pass

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "stage": "queued",
            "question": question,
            "ticker": ticker,
            "follow_up": bool(history),
            "cancel_requested": False,
        }
    _executor.submit(_run_job, job_id, question, ticker, no_verify, sequential, history, thread_id=job_id)
    return job_id


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "jobs": len(_jobs)}


@app.get("/ask")
def ask(
    question: str = Query(...),
    ticker: Optional[str] = Query(None),
    no_verify: bool = Query(False),
    sequential: bool = Query(True),
) -> Dict[str, Any]:
    return {"job_id": _start(question, ticker, no_verify, sequential)}


@app.post("/ask")
def ask_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    if not question:
        return JSONResponse({"detail": "question is required"}, status_code=400)
    return {
        "job_id": _start(
            question,
            payload.get("ticker") or None,
            bool(payload.get("no_verify", False)),
            bool(payload.get("sequential", True)),
            _clean_history(payload.get("history")),
        )
    }


@app.get("/jobs/{job_id}")
def job(job_id: str) -> JSONResponse:
    with _jobs_lock:
        data = _jobs.get(job_id)
    if data is None:
        return JSONResponse({"detail": "unknown job"}, status_code=404)
    return JSONResponse(data)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> JSONResponse:
    """Ask a run to stop at the next safe boundary.

    The workflow is synchronous, so this is cooperative: a running job stops
    after the agent (or stage) it is currently in returns. A queued job is
    marked cancelled immediately and never starts.
    """
    with _jobs_lock:
        data = _jobs.get(job_id)
        if data is None:
            return JSONResponse({"detail": "unknown job"}, status_code=404)
        if data["state"] in ("done", "error", "cancelled"):
            return JSONResponse({"job_id": job_id, "state": data["state"]})
        data["cancel_requested"] = True
        if data["state"] == "queued":
            data.update(
                {
                    "state": "cancelled",
                    "stage": "cancelled",
                    "status": "cancelled",
                    "detail": "Stopped by user",
                }
            )
        return JSONResponse(
            {
                "job_id": job_id,
                "state": data["state"],
                "cancel_requested": True,
            }
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
