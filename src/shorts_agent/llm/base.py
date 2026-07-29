"""Provider-agnostic LLM interface.

The pipeline only ever needs two things from a model: prose, and a JSON object
matching a schema. Keeping the surface that small means adding a provider is one
small class, and tests can substitute a fake without touching pipeline code.

Note there is deliberately no ``temperature`` knob: current Anthropic models
reject sampling parameters outright, so variety is steered by prompting instead
(see ``ideation/prompts.py``). Providers that still accept sampling parameters
configure them internally.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from shorts_agent.exceptions import ProviderError


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(ABC):
    """Minimal LLM surface used by the pipeline."""

    def __init__(self, model: str, *, max_tokens: int = 8192):
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a free-form text completion."""

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int | None = None,
        attempts: int = 3,
    ) -> Any:
        """Return JSON conforming to ``schema``.

        Subclasses that support native structured output should override this;
        the fallback here re-prompts with the parse error fed back, which is far
        more reliable than a single strict attempt against a plain text model.
        """
        instruction = (
            f"{prompt}\n\nRespond with ONLY valid JSON matching this JSON Schema, "
            f"no prose, no markdown fences:\n{json.dumps(schema, indent=2)}"
        )
        last_error: Exception | None = None
        for _ in range(attempts):
            response = self.complete(instruction, system=system, max_tokens=max_tokens)
            try:
                return extract_json(response.text)
            except ValueError as exc:
                last_error = exc
                instruction = (
                    f"{prompt}\n\nYour previous reply could not be parsed as JSON "
                    f"({exc}). Respond with ONLY valid JSON matching:\n"
                    f"{json.dumps(schema, indent=2)}"
                )
        raise ProviderError(f"LLM returned no valid JSON after {attempts} attempts: {last_error}")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON out of a model reply, tolerating fences and surrounding prose."""
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1))

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in response")


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Build a structured-output-compatible object schema.

    Structured outputs require ``additionalProperties: false`` and an explicit
    ``required`` list on every object, so this helper keeps call sites honest.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }
