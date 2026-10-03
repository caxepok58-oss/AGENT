"""OpenAI-compatible implementation of :class:`LLMClient`.

Works against any endpoint that speaks the OpenAI chat-completions API: OpenAI
itself, or a local server (Ollama, LM Studio, vLLM) or hosted gateway via
``base_url``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from shorts_agent.exceptions import ProviderError
from shorts_agent.llm.base import LLMClient, LLMResponse, extract_json, is_loopback_url

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

DEFAULT_ENDPOINT = "https://api.openai.com/v1"


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.9,
    ):
        super().__init__(model, max_tokens=max_tokens)
        try:
            from openai import DefaultHttpxClient, OpenAI
        except ImportError as exc:
            raise ProviderError(
                "The openai package is required for the OpenAI provider. "
                "Install with: pip install 'shorts-agent[llm-openai]'"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if is_loopback_url(base_url):
            # System proxy variables (routine with VPN tools) rarely exclude
            # localhost and would swallow requests meant for a server on this
            # machine, surfacing as a baffling "connection error".
            kwargs["http_client"] = DefaultHttpxClient(trust_env=False)
        self._client = OpenAI(**kwargs)
        self.base_url = base_url
        self.temperature = temperature

    def _messages(self, prompt: str, system: str | None) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=self._messages(prompt, system),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as our own error type
            # Naming the endpoint matters for local servers, where the usual
            # cause is simply that nothing is listening there.
            endpoint = self.base_url or DEFAULT_ENDPOINT
            raise ProviderError(f"OpenAI-compatible request to {endpoint} failed: {exc}") from exc

        return LLMResponse(
            text=response.choices[0].message.content or "",
            raw=response,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
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
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=self._messages(prompt, system),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True, "schema": schema},
                },
            )
        except Exception:  # noqa: BLE001 - older models/endpoints lack json_schema support
            return super().complete_json(
                prompt, schema=schema, system=system, max_tokens=max_tokens, attempts=attempts
            )

        text = response.choices[0].message.content or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # OpenAI guarantees schema-valid JSON, but many OpenAI-compatible servers
        # (small local models especially) accept response_format and then ignore
        # it, replying with fenced or chatty JSON. Parse leniently, then fall back
        # to re-prompting with the parse error, rather than failing a run over
        # formatting.
        try:
            return extract_json(text)
        except ValueError:
            return super().complete_json(
                prompt, schema=schema, system=system, max_tokens=max_tokens, attempts=attempts
            )
