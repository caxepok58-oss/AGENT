from __future__ import annotations

import pytest
import yaml

from shorts_agent.config import (
    AppConfig,
    ContentConfig,
    PublishingConfig,
    get_settings,
    load_config,
)
from shorts_agent.exceptions import ConfigError

MINIMAL = {
    "channel": {
        "name": "C",
        "niche": "n",
        "persona": "p",
    }
}


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Settings are cached and read the real environment, and LLM_PROVIDER and
    TTS_PROVIDER now change what load_config returns, so start each test clean."""
    get_settings.cache_clear()
    for var in ("LLM_PROVIDER", "TTS_PROVIDER", "SHORTS_AGENT_DB", "SHORTS_AGENT_OUTPUT_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # keep a developer's real .env out of the picture
    yield
    get_settings.cache_clear()


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_config_applies_defaults(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)

    assert config.channel.name == "C"
    assert config.providers.llm == "anthropic"
    # Publishing must default to the safe side: private, no auto-publish.
    assert config.publishing.privacy_status == "private"
    assert config.publishing.auto_publish is False
    assert config.content.ai_disclosure is True


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_config_raises_config_error(tmp_path):
    path = _write(tmp_path, {"channel": {"name": "C"}})  # missing niche/persona
    with pytest.raises(ConfigError, match="Invalid config"):
        load_config(path, base_dir=tmp_path)


def test_duration_over_platform_limit_is_rejected():
    with pytest.raises(ConfigError, match="cannot exceed 180s"):
        ContentConfig(max_duration_seconds=240)


def test_target_duration_must_sit_within_bounds():
    with pytest.raises(ConfigError, match="must be between"):
        ContentConfig(min_duration_seconds=30, target_duration_seconds=10, max_duration_seconds=60)


def test_negative_upload_cap_is_rejected():
    with pytest.raises(ConfigError, match="cannot be negative"):
        PublishingConfig(max_uploads_per_day=-1)


def test_relative_paths_resolve_against_base_dir(tmp_path):
    config = load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)
    resolved = config.resolve_path("data/agent.db")

    assert resolved == tmp_path / "data" / "agent.db"


def test_absolute_paths_are_left_alone(tmp_path):
    config = AppConfig(channel=MINIMAL["channel"], base_dir=tmp_path)  # type: ignore[arg-type]

    assert config.resolve_path("/var/lib/agent.db").as_posix() == "/var/lib/agent.db"


def test_llm_provider_from_the_environment_overrides_the_yaml(tmp_path, monkeypatch):
    """SETUP.md tells users to set LLM_PROVIDER in .env; it used to be silently ignored."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()

    config = load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)

    assert config.providers.llm == "ollama"


def test_tts_provider_from_the_environment_overrides_the_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "elevenlabs")
    get_settings.cache_clear()

    config = load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)

    assert config.providers.tts == "elevenlabs"


def test_the_yaml_provider_wins_when_the_environment_says_nothing(tmp_path):
    """Settings' own default ("anthropic") must never beat an explicit YAML choice."""
    data = {**MINIMAL, "providers": {"llm": "openai", "tts": "elevenlabs"}}

    config = load_config(_write(tmp_path, data), base_dir=tmp_path)

    assert config.providers.llm == "openai"
    assert config.providers.tts == "elevenlabs"


def test_provider_override_from_a_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text("LLM_PROVIDER=ollama\n")

    config = load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)

    assert config.providers.llm == "ollama"


def test_an_override_is_logged_when_it_changes_the_yaml_value(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    get_settings.cache_clear()

    with caplog.at_level("INFO", logger="shorts_agent.config"):
        load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)

    assert "overrides providers.llm" in caplog.text


def test_an_override_that_matches_the_yaml_is_not_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()

    with caplog.at_level("INFO", logger="shorts_agent.config"):
        load_config(_write(tmp_path, MINIMAL), base_dir=tmp_path)

    assert caplog.text == ""
