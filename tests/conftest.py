from __future__ import annotations

from typing import Any

import pytest

from shorts_agent.config import AppConfig, ChannelConfig
from shorts_agent.llm.base import LLMClient, LLMResponse


class FakeLLM(LLMClient):
    """Scripted LLM stand-in.

    ``json_responses`` is consumed in order by ``complete_json``, so a test can
    script a multi-call sequence (e.g. ideas, then script, then metadata).
    """

    def __init__(
        self,
        *,
        text: str = "fake text",
        json_responses: list[Any] | None = None,
        error: Exception | None = None,
    ):
        super().__init__("fake-model")
        self.text = text
        self.json_responses = list(json_responses or [])
        self.error = error
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        return LLMResponse(text=self.text)

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int | None = None,
        attempts: int = 3,
    ) -> Any:
        self.prompts.append(prompt)
        self.systems.append(system)
        if self.error is not None:
            raise self.error
        if not self.json_responses:
            raise AssertionError("FakeLLM ran out of scripted JSON responses")
        return self.json_responses.pop(0)


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return AppConfig(
        channel=ChannelConfig(
            name="Test Channel",
            niche="personal finance for beginners",
            persona="A concise, upbeat finance coach.",
            default_hashtags=["#Shorts", "#money"],
        ),
        base_dir=tmp_path,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
