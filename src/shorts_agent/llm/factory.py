from __future__ import annotations

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.llm.base import LLMClient

# Ollama serves an OpenAI-compatible API under /v1 on this port by default.
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Local OpenAI-compatible servers ignore the API key, but the OpenAI SDK refuses
# to build a client without one.
PLACEHOLDER_API_KEY = "not-needed"


def build_llm_client(config: AppConfig) -> LLMClient:
    settings = get_settings()
    provider = config.providers.llm

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is not set (see .env.example)")
        from shorts_agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model=settings.llm_model, api_key=settings.anthropic_api_key)

    if provider in ("openai", "ollama"):
        from shorts_agent.llm.openai_client import OpenAIClient

        base_url = settings.openai_base_url
        api_key = settings.openai_api_key
        model = settings.llm_model

        if provider == "ollama":
            base_url = base_url or OLLAMA_BASE_URL
            if model.startswith("claude"):
                # The default LLM_MODEL is a Claude name; Ollama has no such model.
                raise ConfigError(
                    f"LLM_MODEL is {model!r}, which is not a model Ollama serves. Pull one "
                    "(for example `ollama pull qwen2.5:7b`) and set LLM_MODEL to its name."
                )

        if base_url:
            # A custom endpoint decides which model names are valid (Ollama, LM
            # Studio, a gateway in front of other vendors), so the name passes
            # through untouched, and most such servers need no real key.
            api_key = api_key or PLACEHOLDER_API_KEY
        else:
            if not api_key:
                raise ConfigError("OPENAI_API_KEY is not set (see .env.example)")
            if model.startswith("claude"):
                # A leftover Claude name is certainly invalid for the OpenAI API.
                model = "gpt-4o"

        return OpenAIClient(model=model, api_key=api_key, base_url=base_url)

    raise ConfigError(f"Unknown LLM provider: {provider}")
