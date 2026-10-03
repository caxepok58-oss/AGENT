from __future__ import annotations

import pytest

from shorts_agent.config import get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.llm.factory import OLLAMA_BASE_URL, PLACEHOLDER_API_KEY, build_llm_client
from shorts_agent.llm.openai_client import OpenAIClient

ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Settings are cached and read the real environment; start each test clean."""
    get_settings.cache_clear()
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # keep a developer's real .env out of the picture
    yield
    get_settings.cache_clear()


@pytest.fixture
def env(monkeypatch):
    def set_env(**values: str) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name.upper(), value)
        get_settings.cache_clear()

    return set_env


def endpoint(client: object) -> str:
    assert isinstance(client, OpenAIClient)
    return str(client._client.base_url).rstrip("/")


def api_key(client: object) -> str:
    assert isinstance(client, OpenAIClient)
    return client._client.api_key


# --- openai ----------------------------------------------------------------


def test_openai_without_a_key_or_endpoint_is_a_config_error(config):
    config.providers.llm = "openai"

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_llm_client(config)


def test_openai_replaces_a_leftover_claude_model_name(config, env):
    """The default LLM_MODEL is a Claude name, which is certainly invalid for OpenAI."""
    config.providers.llm = "openai"
    env(openai_api_key="sk-test")

    client = build_llm_client(config)

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"
    assert client.base_url is None


def test_openai_keeps_an_explicit_model(config, env):
    config.providers.llm = "openai"
    env(openai_api_key="sk-test", llm_model="gpt-4.1")

    assert build_llm_client(config).model == "gpt-4.1"


def test_a_custom_endpoint_needs_no_key_and_passes_the_model_through(config, env):
    config.providers.llm = "openai"
    env(openai_base_url="http://localhost:1234/v1", llm_model="my-local-model")

    client = build_llm_client(config)

    assert isinstance(client, OpenAIClient)
    assert endpoint(client) == "http://localhost:1234/v1"
    assert api_key(client) == PLACEHOLDER_API_KEY
    assert client.model == "my-local-model"


def test_a_custom_endpoint_uses_a_real_key_when_one_is_given(config, env):
    """Hosted OpenAI-compatible services (DeepSeek and the like) do need their key."""
    config.providers.llm = "openai"
    env(openai_base_url="https://api.example.com/v1", openai_api_key="sk-real", llm_model="m")

    assert api_key(build_llm_client(config)) == "sk-real"


def test_a_custom_endpoint_does_not_rewrite_a_claude_model_name(config, env):
    """A gateway in front of Claude legitimately serves model names like this."""
    config.providers.llm = "openai"
    env(openai_base_url="http://localhost:4000/v1", llm_model="claude-sonnet-5-5")

    assert build_llm_client(config).model == "claude-sonnet-5-5"


# --- ollama ----------------------------------------------------------------


def test_ollama_defaults_to_the_local_server_without_a_key(config, env):
    config.providers.llm = "ollama"
    env(llm_model="qwen2.5:7b")

    client = build_llm_client(config)

    assert isinstance(client, OpenAIClient)
    assert endpoint(client) == OLLAMA_BASE_URL
    assert api_key(client) == PLACEHOLDER_API_KEY
    assert client.model == "qwen2.5:7b"


def test_ollama_can_point_at_another_machine(config, env):
    config.providers.llm = "ollama"
    env(llm_model="qwen2.5:7b", openai_base_url="http://192.168.1.50:11434/v1")

    assert endpoint(build_llm_client(config)) == "http://192.168.1.50:11434/v1"


def test_ollama_refuses_the_default_claude_model_name_with_a_fix(config):
    config.providers.llm = "ollama"

    with pytest.raises(ConfigError, match="ollama pull"):
        build_llm_client(config)


# --- unchanged behaviour ---------------------------------------------------


def test_anthropic_still_requires_its_own_key(config, env):
    env(openai_base_url="http://localhost:11434/v1")  # irrelevant to this provider

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_llm_client(config)


# --- settings ----------------------------------------------------------------


def test_a_blank_base_url_line_means_unset(env):
    env(openai_base_url="   ")

    assert get_settings().openai_base_url is None


def test_a_base_url_without_a_scheme_is_a_readable_config_error(env):
    env(openai_base_url="localhost:11434/v1")

    with pytest.raises(ConfigError, match="http://"):
        get_settings()


def test_a_mistyped_provider_is_a_readable_config_error(env):
    env(llm_provider="olama")

    with pytest.raises(ConfigError, match="llm_provider"):
        get_settings()


# --- local-inference profile -------------------------------------------------


def is_local(client: object) -> bool:
    assert isinstance(client, OpenAIClient)
    return client._client.max_retries == 0 and client.max_tokens == 4096


def test_ollama_gets_the_local_profile(config, env):
    config.providers.llm = "ollama"
    env(llm_model="qwen2.5:7b")

    assert is_local(build_llm_client(config))


def test_ollama_on_another_machine_still_gets_the_local_profile(config, env):
    config.providers.llm = "ollama"
    env(llm_model="qwen2.5:7b", openai_base_url="http://192.168.1.50:11434/v1")

    assert is_local(build_llm_client(config))


def test_an_openai_compatible_server_on_this_machine_gets_the_local_profile(config, env):
    config.providers.llm = "openai"
    env(openai_base_url="http://127.0.0.1:1234/v1", llm_model="m")

    assert is_local(build_llm_client(config))


def test_a_hosted_openai_compatible_service_keeps_the_cloud_defaults(config, env):
    config.providers.llm = "openai"
    env(openai_base_url="https://api.example.com/v1", openai_api_key="sk-x", llm_model="m")

    assert not is_local(build_llm_client(config))


def test_plain_openai_keeps_the_cloud_defaults(config, env):
    config.providers.llm = "openai"
    env(openai_api_key="sk-test")

    assert not is_local(build_llm_client(config))
