from __future__ import annotations

import pytest

from shorts_agent.config import get_settings
from shorts_agent.diagnostics import (
    check_config,
    check_ffmpeg,
    check_keys,
    check_youtube_auth,
    run_all,
    summarize,
)


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Settings are cached and read the real environment; start each test clean."""
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
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", str(tmp_path / "absent_secrets.json"))
    monkeypatch.setenv("YOUTUBE_TOKEN_FILE", str(tmp_path / "absent_token.json"))
    # Prevent a developer's real .env from leaking into assertions.
    monkeypatch.chdir(tmp_path)
    yield
    get_settings.cache_clear()


def by_name(checks, fragment):
    return next(c for c in checks if fragment in c.name)


def test_missing_llm_key_is_a_failure(config):
    check = by_name(check_keys(config), "LLM")

    assert check.status == "fail"
    assert "ANTHROPIC_API_KEY" in check.detail


def test_present_llm_key_passes(config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()

    assert by_name(check_keys(config), "LLM").status == "ok"


def test_missing_youtube_key_is_only_a_warning(config):
    """Trend research degrades to the curated list; it must not block a run."""
    assert by_name(check_keys(config), "YouTube trend").status == "warn"


def test_selected_stock_provider_without_a_key_warns(config):
    config.providers.visuals = "pexels"

    check = by_name(check_keys(config), "visuals")
    assert check.status == "warn"
    assert "generated card" in check.detail


def test_generated_visuals_need_no_key(config):
    assert by_name(check_keys(config), "visuals").status == "ok"


def test_elevenlabs_without_a_key_is_a_failure(config):
    config.providers.tts = "elevenlabs"

    assert by_name(check_keys(config), "TTS").status == "fail"


def test_unauthorized_youtube_is_a_warning_not_a_failure():
    """Uploading is opt-in, so an unauthorized machine is a normal state."""
    assert check_youtube_auth()[0].status == "warn"


def test_authorized_youtube_passes(monkeypatch, tmp_path):
    secrets = tmp_path / "client_secret.json"
    token = tmp_path / "youtube_token.json"
    secrets.write_text("{}")
    token.write_text("{}")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRETS_FILE", str(secrets))
    monkeypatch.setenv("YOUTUBE_TOKEN_FILE", str(token))
    get_settings.cache_clear()

    assert check_youtube_auth()[0].status == "ok"


def test_voice_language_mismatch_is_reported_as_a_failure(config):
    config.channel.language = "ru"
    config.providers.edge_tts_voice = "en-US-AndrewNeural"

    assert by_name(check_config(config), "voice language").status == "fail"


def test_matching_voice_language_passes(config):
    config.channel.language = "ru"
    config.providers.edge_tts_voice = "ru-RU-DmitryNeural"

    assert by_name(check_config(config), "voice language").status == "ok"


def test_matching_voice_language_passes_with_incidental_whitespace(config):
    """Regression: this check and the runtime warning in tts/factory.py used to
    normalize channel.language differently, so stray whitespace could make a
    correctly matching voice/language pair fail here only."""
    config.channel.language = " ru "
    config.providers.edge_tts_voice = "ru-RU-DmitryNeural"

    assert by_name(check_config(config), "voice language").status == "ok"


def test_vague_niche_and_persona_warn(config):
    config.channel.niche = "money"
    config.channel.persona = "a coach"

    assert by_name(check_config(config), "niche/persona").status == "warn"


def test_overlong_target_duration_warns(config):
    config.content.target_duration_seconds = 150
    config.content.max_duration_seconds = 180

    assert by_name(check_config(config), "target duration").status == "warn"


def test_auto_publishing_to_public_warns(config):
    config.publishing.privacy_status = "public"
    config.publishing.auto_publish = True

    check = by_name(check_config(config), "publishing safety")
    assert check.status == "warn"
    assert "docs/POLICY.md" in check.fix


def test_safe_publishing_defaults_pass(config):
    assert by_name(check_config(config), "publishing safety").status == "ok"


def test_ffmpeg_and_libass_are_detected():
    """The bundled imageio-ffmpeg build must support burning captions."""
    checks = check_ffmpeg()

    assert by_name(checks, "ffmpeg").status == "ok"
    assert by_name(checks, "libass").status == "ok"


def test_run_all_reports_every_category(config):
    checks = run_all(config)
    names = " ".join(c.name for c in checks)

    for expected in ("core packages", "ffmpeg", "caption font", "LLM", "publishing safety"):
        assert expected in names


def test_summarize_counts_by_status(config):
    ok, warn, fail = summarize(run_all(config))

    assert ok > 0
    assert fail >= 1  # no LLM key in this isolated environment
    assert ok + warn + fail == len(run_all(config))
