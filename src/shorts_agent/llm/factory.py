from __future__ import annotations

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.llm.base import LLMClient


def build_llm_client(config: AppConfig) -> LLMClient:
    settings = get_settings()
    provider = config.providers.llm

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set (see .env.example)")
        from shorts_agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=settings.llm_model, api_key=settings.anthropic_api_key)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set (see .env.example)")
        from shorts_agent.llm.openai_client import OpenAIClient

        model = settings.llm_model if not settings.llm_model.startswith("claude") else "gpt-4o"
        return OpenAIClient(model=model, api_key=settings.openai_api_key)

    raise ConfigError(f"Unknown LLM provider: {provider}")
