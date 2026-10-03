from __future__ import annotations

import json
import logging

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
        "LLM_PROVIDER",
        "LLM_MODEL",
        "TTS_PROVIDER",
        "OPENAI_BASE_URL",
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


def test_init_creates_a_footage_folder_that_says_what_it_is_for(runner, tmp_path):
    target = tmp_path / "project"

    result = runner.invoke(app, ["init", "--dir", str(target)])

    assert result.exit_code == 0
    readme = target / "config" / "footage" / "README.txt"
    assert "Your own footage goes here" in readme.read_text(encoding="utf-8")
    assert "Положите сюда" in readme.read_text(encoding="utf-8")
    # The note is not footage, so it must not make an empty library look populated.
    from shorts_agent.visuals.local import list_footage

    assert list_footage(target / "config" / "footage") == []


def test_init_leaves_an_existing_footage_folder_alone(runner, tmp_path):
    target = tmp_path / "project"
    footage = target / "config" / "footage"
    footage.mkdir(parents=True)
    (footage / "mine.mp4").write_bytes(b"clip")

    result = runner.invoke(app, ["init", "--dir", str(target)])

    assert result.exit_code == 0
    assert [p.name for p in footage.iterdir()] == ["mine.mp4"]


def test_init_points_at_the_two_ways_to_get_real_footage(runner, tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path / "project")])

    assert "PIXABAY_API_KEY" in result.output
    assert "config/footage" in result.output


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


def _use_local_server(monkeypatch, server, model="qwen2.5:7b"):
    """Configure a run the way a user without any API key would: Ollama via .env."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("OPENAI_BASE_URL", server.url)
    get_settings.cache_clear()


def test_doctor_passes_for_a_local_model_setup_with_no_api_key_at_all(
    runner, config_path, server, monkeypatch
):
    server.models = ["qwen2.5:7b"]
    _use_local_server(monkeypatch, server)

    result = runner.invoke(app, ["doctor", "--config", config_path])

    assert result.exit_code == 0
    assert "0 problem(s)" in result.output


def test_doctor_tells_you_when_the_local_model_has_not_been_pulled(
    runner, config_path, server, monkeypatch
):
    server.models = ["llama3.1:latest"]
    _use_local_server(monkeypatch, server)

    result = runner.invoke(app, ["doctor", "--config", config_path])

    assert result.exit_code == 1
    assert "ollama pull qwen2.5:7b" in result.output


def test_ideate_runs_end_to_end_against_a_local_model_with_no_api_key(
    runner, config_path, server, monkeypatch
):
    """The whole chain a no-API-key user depends on: .env -> provider override ->
    factory -> OpenAI-compatible client -> a real HTTP round trip -> ideas on screen."""
    server.reply_with(
        json.dumps(
            {
                "ideas": [
                    {
                        "title": "Idea written by a local model",
                        "hook": "Your phone is quietly costing you money.",
                        "premise": "A short look at forgotten subscriptions.",
                        "target_emotion": "urgency",
                        "trend_keywords": [],
                        "virality_reasoning": "Specific and slightly alarming.",
                    }
                ]
            }
        )
    )
    _use_local_server(monkeypatch, server)

    result = runner.invoke(app, ["ideate", "--config", config_path, "--count", "1"])

    assert result.exit_code == 0, result.output
    assert "Idea written by a local model" in result.output
    request = server.requests[0]
    assert request.path == "/v1/chat/completions"
    assert request.body["model"] == "qwen2.5:7b"
    # The persona and niche from config.yaml really reach the local model.
    assert "personal finance for beginners" in request.body["messages"][-1]["content"]


def test_http_client_request_logs_are_silenced(runner):
    """Current SDKs log every request at INFO through httpx2, which the old
    httpx-only silencing missed, so each model call printed a noise line."""
    from shorts_agent.cli import _setup_logging

    names = ("httpx", "httpx2", "urllib3")
    saved = {name: logging.getLogger(name).level for name in names}
    try:
        _setup_logging(False)
        for name in names:
            assert logging.getLogger(name).level == logging.WARNING
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
