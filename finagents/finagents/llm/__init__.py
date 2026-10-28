"""LLM abstraction and the Ollama provider.

The rest of the system never imports an HTTP client directly; it asks for
structured JSON via :class:`LLMClient`. Production code is backed by Ollama.
Tests substitute a deterministic client (see ``tests/fake_llm.py``) that
implements the same contract without a daemon.
"""

from __future__ import annotations

from .base import LLMClient, LLMError, build_prompt, extract_json
from .factory import build_llm_client
from .ollama_client import OllamaClient

__all__ = [
    "LLMClient",
    "LLMError",
    "OllamaClient",
    "build_llm_client",
    "build_prompt",
    "extract_json",
]
