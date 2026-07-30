"""Environment checks behind the ``doctor`` command.

Nearly every integration here is optional and degrades quietly by design, which
is good for robustness and bad for understanding what your setup will actually
do. These checks answer that in one place: what works, what will silently fall
back, and what will fail outright.

Every check is read-only and local: no API call, no upload, no quota spent. That
makes ``doctor`` safe to run at any time, including from CI, where a non-zero
exit means some part of the pipeline genuinely cannot run.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.tts.factory import language_prefix

Status = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_python_packages() -> list[Check]:
    checks: list[Check] = []
    required = {"pydantic": "core", "yaml": "core", "typer": "CLI", "requests": "HTTP"}
    missing = [name for name in required if not _module_present(name)]
    if missing:
        checks.append(
            Check(
                "core packages",
                "fail",
                f"missing: {', '.join(missing)}",
                "pip install -e .",
            )
        )
    else:
        checks.append(Check("core packages", "ok", "installed"))

    optional = {
        "moviepy": ("video rendering", "pip install -e '.[video]'"),
        "PIL": ("generated visuals", "pip install -e '.[video]'"),
        "numpy": ("generated visuals", "pip install -e '.[video]'"),
        "edge_tts": ("keyless voiceover", "pip install -e '.[tts]'"),
        "googleapiclient": ("YouTube upload", "pip install -e '.[youtube]'"),
        "pytrends": ("Google Trends signals", "pip install -e '.[trends]'"),
        "apscheduler": ("the daemon command", "pip install -e '.[scheduler]'"),
    }
    for module, (purpose, fix) in optional.items():
        if _module_present(module):
            checks.append(Check(f"{module}", "ok", purpose))
        else:
            checks.append(Check(f"{module}", "warn", f"absent — {purpose} unavailable", fix))

    return checks


def _ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall through to PATH
        return shutil.which("ffmpeg")


def check_ffmpeg() -> list[Check]:
    """ffmpeg must exist and be built with libass, or captions cannot be burned."""
    path = _ffmpeg_path()
    if not path:
        return [
            Check(
                "ffmpeg",
                "fail",
                "not found",
                "pip install -e '.[video]' (bundles ffmpeg), or install ffmpeg",
            )
        ]

    try:
        result = subprocess.run(
            [path, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [Check("ffmpeg", "fail", f"present but not runnable: {exc}")]

    checks = [Check("ffmpeg", "ok", Path(path).name)]
    if " ass " in result.stdout:
        checks.append(Check("libass (caption burning)", "ok", "ass filter available"))
    else:
        checks.append(
            Check(
                "libass (caption burning)",
                "fail",
                "this ffmpeg has no 'ass' filter, so captions cannot be burned in",
                "install an ffmpeg built with --enable-libass, or rely on the bundled one",
            )
        )
    return checks


def check_font(config: AppConfig) -> list[Check]:
    """The caption font is resolved by name at render time, so it must be installed."""
    font = config.captions.font
    if not shutil.which("fc-match"):
        return [
            Check(
                f"caption font {font!r}",
                "warn",
                "cannot verify (fontconfig tools not installed)",
                "install fontconfig, or check the font manually",
            )
        ]

    try:
        result = subprocess.run(["fc-match", font], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return [Check(f"caption font {font!r}", "warn", f"could not run fc-match: {exc}")]

    resolved = result.stdout.strip()
    # fc-match always returns *something*; a mismatch means it substituted.
    if font.split()[0].lower() in resolved.lower():
        return [Check(f"caption font {font!r}", "ok", resolved.split(":")[0])]
    return [
        Check(
            f"caption font {font!r}",
            "warn",
            f"not installed; fontconfig would substitute {resolved.split(':')[0]}",
            "set captions.font to an installed font (see: fc-list : family)",
        )
    ]


def check_keys(config: AppConfig) -> list[Check]:
    settings = get_settings()
    checks: list[Check] = []

    provider = config.providers.llm
    key = settings.anthropic_api_key if provider == "anthropic" else settings.openai_api_key
    env_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if key:
        checks.append(
            Check(f"LLM ({provider})", "ok", f"{env_name} set, model {settings.llm_model}")
        )
    else:
        checks.append(
            Check(
                f"LLM ({provider})",
                "fail",
                f"{env_name} is not set — ideas, scripts and metadata cannot be generated",
                f"add {env_name} to .env (see docs/SETUP.md)",
            )
        )

    if settings.youtube_api_key:
        checks.append(Check("YouTube trend signals", "ok", "YOUTUBE_API_KEY set"))
    else:
        checks.append(
            Check(
                "YouTube trend signals",
                "warn",
                "no YOUTUBE_API_KEY — live trending data will be skipped",
                "add YOUTUBE_API_KEY, or rely on config/manual_trends.yaml",
            )
        )

    visuals = config.providers.visuals
    if visuals == "generated":
        checks.append(
            Check("visuals", "ok", "generated cards (no key needed; stock footage performs better)")
        )
    else:
        key = settings.pexels_api_key if visuals == "pexels" else settings.pixabay_api_key
        if key:
            checks.append(Check(f"visuals ({visuals})", "ok", "key set"))
        else:
            checks.append(
                Check(
                    f"visuals ({visuals})",
                    "warn",
                    f"{visuals} selected but its key is missing — every scene will "
                    "fall back to a generated card",
                    f"add {visuals.upper()}_API_KEY, or set providers.visuals: generated",
                )
            )

    tts = config.providers.tts
    if tts == "elevenlabs" and not settings.elevenlabs_api_key:
        checks.append(
            Check(
                "TTS (elevenlabs)",
                "fail",
                "ELEVENLABS_API_KEY is not set",
                "add the key, or set providers.tts: edge",
            )
        )
    else:
        checks.append(Check(f"TTS ({tts})", "ok", "configured"))

    return checks


def check_youtube_auth() -> list[Check]:
    """Reports whether uploading is possible, without contacting YouTube."""
    settings = get_settings()
    secrets = Path(settings.youtube_client_secrets_file)
    token = Path(settings.youtube_token_file)

    if not secrets.exists():
        return [
            Check(
                "YouTube upload",
                "warn",
                f"no OAuth client secrets at {secrets} — uploading is unavailable",
                "see docs/SETUP.md section 5, or keep using local-only runs",
            )
        ]
    if not token.exists():
        return [
            Check(
                "YouTube upload",
                "warn",
                "client secrets present but this machine is not authorized yet",
                "run: shorts-agent auth",
            )
        ]
    return [Check("YouTube upload", "ok", f"authorized (token at {token})")]


def check_config(config: AppConfig) -> list[Check]:
    checks: list[Check] = []

    persona = config.channel.persona.strip()
    niche = config.channel.niche.strip()
    if len(niche) < 12 or len(persona) < 30:
        checks.append(
            Check(
                "channel niche/persona",
                "warn",
                "very short — vague values produce generic, interchangeable videos",
                "describe the niche and persona specifically (see docs/SETUP.md)",
            )
        )
    else:
        checks.append(Check("channel niche/persona", "ok", niche[:48]))

    duration = config.content.target_duration_seconds
    if duration > 90:
        checks.append(
            Check(
                "target duration",
                "warn",
                f"{duration:.0f}s — 30-60s consistently retains better on Shorts",
            )
        )
    else:
        checks.append(Check("target duration", "ok", f"{duration:.0f}s"))

    voice = config.providers.edge_tts_voice
    language = language_prefix(config.channel.language)
    if config.providers.tts == "edge" and "-" in voice:
        if language_prefix(voice) != language:
            checks.append(
                Check(
                    "voice language",
                    "fail",
                    f"voice {voice!r} does not speak {config.channel.language!r}; "
                    "narration would be mispronounced",
                    f"pick a {language}-* voice (edge-tts --list-voices)",
                )
            )
        else:
            checks.append(Check("voice language", "ok", voice))

    if config.publishing.privacy_status == "public" and config.publishing.auto_publish:
        checks.append(
            Check(
                "publishing safety",
                "warn",
                "auto-publishing straight to public — no human reviews any video",
                "read docs/POLICY.md; consider privacy_status: private",
            )
        )
    else:
        checks.append(
            Check(
                "publishing safety",
                "ok",
                f"{config.publishing.privacy_status}, max {config.publishing.max_uploads_per_day}/day",
            )
        )

    music_dir = config.resolve_path(config.visuals.music_dir)
    tracks = (
        [
            p
            for p in music_dir.iterdir()
            if p.suffix.lower() in {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
        ]
        if music_dir.exists()
        else []
    )
    checks.append(
        Check("background music", "ok", f"{len(tracks)} track(s) in {music_dir}")
        if tracks
        else Check("background music", "ok", "none configured (optional)")
    )

    return checks


def run_all(config: AppConfig) -> list[Check]:
    return [
        *check_python_packages(),
        *check_ffmpeg(),
        *check_font(config),
        *check_keys(config),
        *check_youtube_auth(),
        *check_config(config),
    ]


def summarize(checks: list[Check]) -> tuple[int, int, int]:
    """Return ``(ok, warn, fail)`` counts."""
    return (
        sum(1 for c in checks if c.status == "ok"),
        sum(1 for c in checks if c.status == "warn"),
        sum(1 for c in checks if c.status == "fail"),
    )
