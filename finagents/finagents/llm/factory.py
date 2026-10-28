"""LLM construction.

Ollama is the only backend. There is no fallback client: a financial agent that
silently falls back to canned text would produce a confident report with no
reasoning behind it, which is the worst possible failure mode for this
application. If Ollama is unreachable the run stops and says how to fix it.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..schemas import Settings
from .base import LLMClient, LLMError
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def build_llm_client(settings: Optional[Settings] = None) -> LLMClient:
    """Return a ready Ollama client, raising `LLMError` if it cannot serve."""
    settings = settings or Settings()
    client = OllamaClient(settings)
    status = client.health()
    if status.get("available"):
        return client
    detail = status.get("detail") or "no response"
    client.close()
    raise LLMError(
        f"Ollama is unavailable at {settings.ollama_base_url} ({detail}). "
        f"Start it with `ollama serve` and pull the models with "
        f"`ollama pull {settings.ollama_model}` and "
        f"`ollama pull {settings.ollama_embedding_model}`. "
        "There is no offline fallback: model-backed reasoning is required."
    )


__all__ = ["build_llm_client"]
