from __future__ import annotations

import pytest

from shorts_agent.config import get_settings
from shorts_agent.diagnostics import (
    _probe_models,
    _small_model_note,
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
        "OPENAI_BASE_URL",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "TTS_PROVIDER",
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


# --- local / OpenAI-compatible LLM ----------------------------------------------


@pytest.fixture
def ollama(config, server, monkeypatch):
    """Config set up for the ollama provider, pointed at the fake local server."""
    config.providers.llm = "ollama"
    monkeypatch.setenv("OPENAI_BASE_URL", server.url)
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    get_settings.cache_clear()
    return config


def test_ollama_needs_no_api_key_and_passes_when_the_model_is_pulled(ollama, server):
    server.models = ["qwen2.5:7b", "llama3.1:latest"]

    check = by_name(check_keys(ollama), "LLM")

    assert check.status == "ok"
    assert "qwen2.5:7b" in check.detail


def test_ollama_with_the_model_missing_says_what_it_has_and_how_to_fix_it(ollama, server):
    server.models = ["llama3.1:latest"]

    check = by_name(check_keys(ollama), "LLM")

    assert check.status == "fail"
    assert "llama3.1:latest" in check.detail
    assert "ollama pull qwen2.5:7b" in check.fix


def test_an_untagged_model_name_matches_the_latest_tag(ollama, server, monkeypatch):
    server.models = ["llama3.1:latest"]
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    get_settings.cache_clear()

    assert by_name(check_keys(ollama), "LLM").status == "ok"


def test_ollama_that_is_not_running_is_a_failure_that_says_how_to_start_it(config, monkeypatch):
    config.providers.llm = "ollama"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens here
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    get_settings.cache_clear()

    check = by_name(check_keys(config), "LLM")

    assert check.status == "fail"
    assert "cannot reach Ollama" in check.detail
    assert "ollama serve" in check.fix


def test_ollama_with_the_default_claude_model_name_is_a_failure(config):
    config.providers.llm = "ollama"

    check = by_name(check_keys(config), "LLM")

    assert check.status == "fail"
    assert "ollama pull" in check.fix


def test_custom_openai_endpoint_needs_no_key_and_is_not_probed(config, monkeypatch):
    """Only Ollama is probed; a hosted endpoint may be metered, and doctor stays offline."""
    config.providers.llm = "openai"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    get_settings.cache_clear()

    def boom(url):
        raise AssertionError("doctor must not contact a non-Ollama endpoint")

    monkeypatch.setattr("shorts_agent.diagnostics._probe_models", boom)

    check = by_name(check_keys(config), "LLM")

    assert check.status == "ok"
    assert "no API key needed" in check.detail


def test_custom_openai_endpoint_with_a_claude_model_name_is_only_a_warning(config, monkeypatch):
    """A gateway in front of Claude is legitimate, so this can't be a failure."""
    config.providers.llm = "openai"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:4000/v1")
    get_settings.cache_clear()

    assert by_name(check_keys(config), "LLM").status == "warn"


def test_probe_returns_the_model_ids_the_server_reports(server):
    server.models = ["a:1", "b:2"]

    assert _probe_models(server.url) == ["a:1", "b:2"]


def test_probe_returns_none_for_an_unreachable_server():
    assert _probe_models("http://127.0.0.1:9/v1") is None


def test_probe_ignores_a_system_proxy_for_local_servers(server, monkeypatch):
    """VPN tools often export HTTP_PROXY without excluding localhost."""
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    server.models = ["a:1"]

    assert _probe_models(server.url) == ["a:1"]


@pytest.mark.parametrize(
    "model", ["qwen2.5:1.5b", "qwen2.5:0.5b-instruct", "llama3.2:3b", "gemma3:4b", "phi3:3.8b"]
)
def test_small_models_get_a_warning(model):
    assert _small_model_note(model) is not None


@pytest.mark.parametrize(
    "model",
    ["qwen2.5:7b", "llama3.1:8b", "gemma3:12b", "qwen2.5:14b-instruct", "llama3.1:70b", "mistral"],
)
def test_seven_billion_parameters_and_up_are_not_warned_about(model):
    assert _small_model_note(model) is None


def test_doctor_warns_but_does_not_fail_for_a_small_ollama_model(config, server, monkeypatch):
    config.providers.llm = "ollama"
    server.models = ["qwen2.5:1.5b"]
    monkeypatch.setenv("OPENAI_BASE_URL", server.url)
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:1.5b")
    get_settings.cache_clear()

    check = by_name(check_keys(config), "LLM")

    assert check.status == "warn"
    assert "qwen2.5:7b" in check.fix
