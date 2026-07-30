from __future__ import annotations

import pytest
from typer.testing import CliRunner

from shorts_agent.cli import app
from shorts_agent.config import get_settings

CONFIG_YAML = """\
channel:
  name: Test Channel
  niche: personal finance for beginners
  persona: A concise, upbeat finance coach who explains things simply.
  language: en
providers:
  visuals: generated
  tts: edge
  edge_tts_voice: en-US-AndrewNeural
"""


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Settings are cached and read the real environment; start each test clean
    and off the real repo's cwd/.env, same isolation as test_diagnostics.py."""
    get_settings.cache_clear()
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "YOUTUBE_API_KEY",
        "PEXELS_API_KEY",
        "PIXABAY_API_KEY",
        "ELEVENLABS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    yield
    get_settings.cache_clear()


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_path(tmp_path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML)
    return str(path)


def test_version_prints_the_installed_version(runner):
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "shorts-agent" in result.output


def test_init_creates_config_files(runner, tmp_path):
    target = tmp_path / "project"

    result = runner.invoke(app, ["init", "--dir", str(target)])

    assert result.exit_code == 0
    assert (target / "config" / "config.yaml").exists()
    assert (target / ".env").exists()


def test_init_does_not_overwrite_without_force(runner, tmp_path):
    target = tmp_path / "project"
    runner.invoke(app, ["init", "--dir", str(target)])
    custom = "# my hand-edited config\n"
    (target / "config" / "config.yaml").write_text(custom)

    result = runner.invoke(app, ["init", "--dir", str(target)])

    assert result.exit_code == 0
    assert (target / "config" / "config.yaml").read_text() == custom


def test_init_overwrites_with_force(runner, tmp_path):
    target = tmp_path / "project"
    runner.invoke(app, ["init", "--dir", str(target)])
    (target / "config" / "config.yaml").write_text("# my hand-edited config\n")

    result = runner.invoke(app, ["init", "--dir", str(target), "--force"])

    assert result.exit_code == 0
    assert "my hand-edited config" not in (target / "config" / "config.yaml").read_text()


def test_doctor_fails_without_an_llm_key(runner, config_path):
    result = runner.invoke(app, ["doctor", "--config", config_path])

    assert result.exit_code == 1
    assert "problem(s)" in result.output


def test_doctor_passes_with_an_llm_key(runner, config_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    get_settings.cache_clear()

    result = runner.invoke(app, ["doctor", "--config", config_path])

    assert result.exit_code == 0
    assert "0 problem(s)" in result.output


def test_trends_escapes_markup_in_external_topic_keywords(runner, config_path, monkeypatch):
    """Rich interprets [...] as markup by default, including [link=...] which
    some terminals render as a clickable hyperlink. Topic keywords are raw
    external content (video titles anyone with a trending video controls), so
    a crafted one must survive as literal text, not a hidden link."""
    from shorts_agent.models import TrendTopic
    from shorts_agent.pipeline import Pipeline

    malicious = TrendTopic(
        keyword="Check this [link=https://evil.example/phish]click here[/link] out",
        source="youtube:most_popular",
    )
    monkeypatch.setattr(Pipeline, "research", lambda self, limit=None: [malicious])

    result = runner.invoke(app, ["trends", "--config", config_path])

    assert result.exit_code == 0
    # Unescaped, Rich would silently consume the tags and show only "click
    # here" — with no visible trace that a link was ever there.
    assert "[link=https://evil.example/phish]" in result.output


def test_trends_reports_when_no_signals_are_found(runner, config_path):
    """Default providers (youtube, manual) are both unavailable with no key and
    no keywords file configured, so this must degrade cleanly, not crash."""
    result = runner.invoke(app, ["trends", "--config", config_path])

    assert result.exit_code == 0
    assert "No trend signals found" in result.output


def test_ideate_fails_cleanly_without_an_llm_key(runner, config_path):
    result = runner.invoke(app, ["ideate", "--config", config_path])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_run_fails_cleanly_without_an_llm_key(runner, config_path):
    result = runner.invoke(app, ["run", "--config", config_path])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_preview_reports_a_missing_run(runner, config_path):
    result = runner.invoke(app, ["preview", "does-not-exist", "--config", config_path])

    assert result.exit_code == 1
    assert "No run found" in result.output


def test_publish_reports_a_missing_run(runner, config_path):
    result = runner.invoke(app, ["publish", "does-not-exist", "--config", config_path])

    assert result.exit_code == 1
    assert "No run found" in result.output


def test_ideate_prints_generated_ideas_with_a_scripted_llm(runner, config_path, monkeypatch):
    from tests.conftest import FakeLLM

    fake = FakeLLM(
        json_responses=[
            {
                "ideas": [
                    {
                        "title": "The subscription quietly draining your account",
                        "hook": "You are probably paying for something you forgot.",
                        "premise": "A short walkthrough of finding forgotten subscriptions.",
                        "target_emotion": "urgency",
                        "trend_keywords": [],
                        "virality_reasoning": "Specific, actionable, and mildly alarming.",
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("shorts_agent.pipeline.orchestrator.build_llm_client", lambda config: fake)

    result = runner.invoke(app, ["ideate", "--config", config_path, "--count", "1"])

    assert result.exit_code == 0
    assert "subscription quietly draining" in result.output
    assert "ok" in result.output
