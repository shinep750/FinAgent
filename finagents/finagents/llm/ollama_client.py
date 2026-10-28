"""Ollama-backed LLM client (the production path).

Talks to a local Ollama daemon over its REST API with `format="json"`, so the
model is constrained to emit parseable JSON. No cloud calls, no API key.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from ..schemas import Settings
from .base import LLMClient, LLMError

logger = logging.getLogger(__name__)


def _has_model(models: List[str], wanted: str) -> bool:
    """Whether `wanted` names an installed model.

    Ollama reports a tag with every model - `llama3.1:latest` - while a user
    configures the bare name. Comparing them literally reports a model that was
    pulled a second ago as missing, and sends the user off to re-run a 4.9 GB
    download that is already on disk. Matching on the `name:tag` boundary keeps
    this exact: `llama` does not match `llama3.1:latest`.
    """
    if wanted in models:
        return True
    return any(name.startswith(f"{wanted}:") for name in models)


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.ollama_base_url.rstrip("/"),
            timeout=settings.llm_timeout_s,
        )

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": (
                    self.settings.llm_temperature
                    if temperature is None
                    else temperature
                ),
                "num_predict": max_tokens or self.settings.llm_max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        try:
            response = self._client.post("/api/generate", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"cannot reach Ollama at {self.settings.ollama_base_url}: {exc}. "
                "Is `ollama serve` running?"
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"Ollama returned {response.status_code}: {response.text[:300]}"
            )

        body = response.json()
        content = body.get("response", "")
        if not content.strip():
            raise LLMError("Ollama returned an empty response")
        return content

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed via a local embedding model; used for the semantic channel."""
        vectors: List[List[float]] = []
        for text in texts:
            response = self._client.post(
                "/api/embeddings",
                json={"model": self.settings.ollama_embedding_model, "prompt": text},
            )
            if response.status_code >= 400:
                raise LLMError(f"Ollama embeddings failed: {response.status_code}")
            vectors.append(response.json()["embedding"])
        return vectors

    def health(self) -> Dict[str, Any]:
        try:
            response = self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except httpx.HTTPError as exc:
            return {
                "provider": self.name,
                "available": False,
                "detail": f"unreachable at {self.settings.ollama_base_url}: {exc}",
            }
        return {
            "provider": self.name,
            "available": _has_model(models, self.settings.ollama_model),
            "configured_model": self.settings.ollama_model,
            "installed_models": models,
            "detail": (
                None
                if _has_model(models, self.settings.ollama_model)
                else f"model '{self.settings.ollama_model}' not pulled; run: ollama pull {self.settings.ollama_model}"
            ),
        }

    def close(self) -> None:
        self._client.close()
