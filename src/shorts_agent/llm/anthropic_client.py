"""Anthropic implementation of :class:`LLMClient`.

Two details worth knowing if you extend this:

* Current Anthropic models reject ``temperature``/``top_p``/``top_k`` with a 400,
  so response variety is steered by prompting, and depth by ``output_config.effort``.
* ``output_config.format`` gives schema-guaranteed JSON, so ``complete_json``
  needs no retry loop — the parse cannot fail on a successful response.
"""

from __future__ import annotations

import json
from typing import Any

from shorts_agent.exceptions import ProviderError
from shorts_agent.llm.base import LLMClient, LLMResponse

# Above this, the SDK can hit HTTP timeouts on non-streaming requests.
_STREAM_THRESHOLD = 16000


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 8192,
        effort: str = "medium",
    ):
        super().__init__(model, max_tokens=max_tokens)
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "The anthropic package is required for the Anthropic provider. "
                "Install with: pip install 'shorts-agent[llm-anthropic]'"
            ) from exc

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.effort = effort

    def _request(self, **kwargs: Any) -> Any:
        try:
            if kwargs.get("max_tokens", 0) > _STREAM_THRESHOLD:
                with self._client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()
            return self._client.messages.create(**kwargs)
        except self._anthropic.RateLimitError as exc:
            raise ProviderError(f"Anthropic rate limit hit: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise ProviderError(f"Could not reach the Anthropic API: {exc}") from exc

    def _build(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int | None,
        output_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": self.effort}
        if output_format:
            output_config["format"] = output_format

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            kwargs["system"] = system
        return kwargs

    @staticmethod
    def _text_of(message: Any) -> str:
        return "".join(block.text for block in message.content if block.type == "text")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        message = self._request(**self._build(prompt, system, max_tokens))
        if message.stop_reason == "refusal":
            raise ProviderError("Anthropic declined the request (stop_reason=refusal)")
        return LLMResponse(
            text=self._text_of(message),
            raw=message,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int | None = None,
        attempts: int = 3,
    ) -> Any:
        message = self._request(
            **self._build(
                prompt,
                system,
                max_tokens,
                output_format={"type": "json_schema", "schema": schema},
            )
        )
        if message.stop_reason == "refusal":
            raise ProviderError("Anthropic declined the request (stop_reason=refusal)")
        if message.stop_reason == "max_tokens":
            raise ProviderError(
                "Anthropic response was truncated before the JSON was complete; "
                "raise max_tokens for this call."
            )
        text = self._text_of(message)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # schema-guaranteed, so this is a real fault
            raise ProviderError(f"Structured output was not valid JSON: {exc}") from exc
