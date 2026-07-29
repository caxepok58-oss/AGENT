from __future__ import annotations

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.visuals.base import VisualProvider


def build_visual_provider(config: AppConfig) -> VisualProvider:
    settings = get_settings()
    provider = config.providers.visuals
    width = config.visuals.width
    height = config.visuals.height

    if provider == "generated":
        from shorts_agent.visuals.generated import GeneratedVisualProvider

        return GeneratedVisualProvider(width=width, height=height)

    if provider == "pexels":
        from shorts_agent.visuals.stock import PexelsVisualProvider

        return PexelsVisualProvider(settings.pexels_api_key, width=width, height=height)

    if provider == "pixabay":
        from shorts_agent.visuals.stock import PixabayVisualProvider

        return PixabayVisualProvider(settings.pixabay_api_key, width=width, height=height)

    raise ConfigError(f"Unknown visuals provider: {provider}")
