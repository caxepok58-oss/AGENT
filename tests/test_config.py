from __future__ import annotations

import pytest
import yaml

from shorts_agent.config import AppConfig, ContentConfig, PublishingConfig, load_config
from shorts_agent.exceptions import ConfigError

MINIMAL = {
    "channel": {
        "name": "C",
        "niche": "n",
        "persona": "p",
    }
}


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
