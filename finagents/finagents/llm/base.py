"""LLM abstraction.

The rest of the system never imports an HTTP client directly. It asks for
structured JSON via :class:`LLMClient`, which lets the same pipeline run
against a real local model (Ollama) or a deterministic test double that
implements the same JSON contract.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a usable completion."""


def extract_json(raw: str) -> Any:
    """Parse JSON from a model response that may be wrapped in prose or fences.

    Local models frequently ignore "reply with JSON only", so this is
    deliberately forgiving: it tries the whole string, then fenced blocks,
    then the first balanced object/array.
    """
    if raw is None:
        raise LLMError("empty response")
    text = raw.strip()
    if not text:
        raise LLMError("empty response")

    attempts: List[str] = [text]
    for match in _JSON_BLOCK.finditer(text):
        attempts.append(match.group(1).strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        attempts.append(text[start : idx + 1])
                        break
            start = text.find(opener, start + 1)

    for attempt in attempts:
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue

    raise LLMError(f"no parseable JSON in response: {text[:400]!r}")


class LLMClient(ABC):
    """Minimal structured-output contract required by every agent."""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Return raw text for `prompt`."""

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 2,
    ) -> Any:
        """Return a parsed JSON value, repairing/retrying on parse failure."""
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            extra = ""
            if attempt:
                extra = (
                    "\n\nIMPORTANT: your previous reply could not be parsed as JSON. "
                    "Reply with a single valid JSON object and nothing else."
                )
            try:
                raw = self.complete(
                    prompt + extra,
                    system,
                    temperature=(
                        temperature
                        if attempt == 0
                        else min(0.3, (temperature or 0.0) + 0.1)
                    ),
                    max_tokens=max_tokens,
                )
                return extract_json(raw)
            except (LLMError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "LLM JSON parse failed (attempt %s): %s", attempt + 1, exc
                )
        raise LLMError(
            f"failed to obtain valid JSON after {retries + 1} attempts: {last_error}"
        )

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "available": True}


def build_prompt(
    system: str, instruction: str, context: str = "", output_schema: str = ""
) -> str:
    """Assemble a prompt with a stable section order.

    Local instruction-tuned models follow structure more reliably when the
    instructions, then the data, then the required output shape are separated
    by explicit headers.
    """
    sections = [f"### INSTRUCTION\n{instruction.strip()}"]
    if context:
        sections.append(f"### CONTEXT\n{context.strip()}")
    if output_schema:
        sections.append(f"### REQUIRED OUTPUT FORMAT (JSON)\n{output_schema.strip()}")
    sections.append(
        "### RESPONSE\nRespond with JSON only. No prose, no markdown fences."
    )
    return "\n\n".join(sections)
