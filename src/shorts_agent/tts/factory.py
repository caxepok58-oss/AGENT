from __future__ import annotations

import logging

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.tts.base import TTSProvider

logger = logging.getLogger(__name__)


def language_prefix(value: str) -> str:
    """First BCP-47 subtag, lowercased: the part that identifies the language itself.

    Shared by the runtime mismatch warning below and the ``doctor`` command's
    equivalent check, so both compare voice and channel language the same way
    instead of maintaining two copies that can (and did) drift apart.
    """
    return value.strip().split("-", 1)[0].lower()


def _warn_on_language_mismatch(config: AppConfig) -> None:
    """Warn when the configured voice does not speak the channel's language.

    This is a quiet, expensive mistake: an English voice reading Russian text
    produces confident-sounding nonsense, and nothing else in the pipeline can
    detect it. edge-tts voice names start with a BCP-47 locale, so the check is
    a simple prefix comparison.
    """
    voice = config.providers.edge_tts_voice
    language = config.channel.language.strip()
    if not language or "-" not in voice:
        return

    voice_language = language_prefix(voice)
    channel_language = language_prefix(language)
    if voice_language != channel_language:
        logger.warning(
            "Voice %r speaks %r but channel.language is %r. The narration will be "
            "read with the wrong language's pronunciation — pick a %s-* voice "
            "(`edge-tts --list-voices`).",
            voice,
            voice_language,
            language.lower(),
            channel_language,
        )


def build_tts_provider(config: AppConfig) -> TTSProvider:
    settings = get_settings()
    provider = config.providers.tts

    if provider == "edge":
        _warn_on_language_mismatch(config)
        from shorts_agent.tts.edge_tts_provider import EdgeTTSProvider

        return EdgeTTSProvider(voice=config.providers.edge_tts_voice)

    if provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            raise ConfigError("ELEVENLABS_API_KEY is not set (see .env.example)")
        from shorts_agent.tts.elevenlabs_provider import ElevenLabsProvider

        return ElevenLabsProvider(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
        )

    raise ConfigError(f"Unknown TTS provider: {provider}")
