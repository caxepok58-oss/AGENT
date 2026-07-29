from __future__ import annotations

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.tts.base import TTSProvider


def build_tts_provider(config: AppConfig) -> TTSProvider:
    settings = get_settings()
    provider = config.providers.tts

    if provider == "edge":
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
