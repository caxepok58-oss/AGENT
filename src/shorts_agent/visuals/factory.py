"""Choosing the visual source.

``providers.visuals`` names one, or ``auto`` (the default) picks the best that is
available, so that adding a stock key to ``.env`` or dropping clips into the
footage folder is enough: no second setting to find and change. In ``auto``:

1. Pixabay, if ``PIXABAY_API_KEY`` is set;
2. else Pexels, if ``PEXELS_API_KEY`` is set;
3. else the footage folder, if it holds any clips;
4. else generated gradient cards, which need nothing but are plain.

Whatever is chosen, a scene it cannot supply (no result, a network error) falls
back to the footage folder when that holds clips, and to a generated card otherwise.
"""

from __future__ import annotations

from pathlib import Path

from shorts_agent.config import AppConfig, Settings, get_settings
from shorts_agent.exceptions import ConfigError
from shorts_agent.visuals.base import VisualProvider
from shorts_agent.visuals.local import LocalFootageProvider, list_footage


def footage_dir(config: AppConfig) -> Path:
    return config.resolve_path(config.visuals.footage_dir)


def resolve_visual_provider(config: AppConfig, settings: Settings) -> str:
    """The concrete provider name behind ``providers.visuals``, resolving ``auto``."""
    choice = config.providers.visuals
    if choice != "auto":
        return choice
    if settings.pixabay_api_key:
        return "pixabay"
    if settings.pexels_api_key:
        return "pexels"
    if list_footage(footage_dir(config)):
        return "local"
    return "generated"


def build_visual_provider(config: AppConfig) -> VisualProvider:
    settings = get_settings()
    provider = resolve_visual_provider(config, settings)
    width = config.visuals.width
    height = config.visuals.height

    if provider == "generated":
        from shorts_agent.visuals.generated import GeneratedVisualProvider

        return GeneratedVisualProvider(width=width, height=height)

    local = LocalFootageProvider(footage_dir(config), width=width, height=height)
    if provider == "local":
        return local

    # A stock provider's fallback: the owner's own clips when there are any, which
    # beats a gradient card; otherwise it builds the generated card itself.
    fallback = local if local.clips else None

    if provider == "pexels":
        from shorts_agent.visuals.stock import PexelsVisualProvider

        return PexelsVisualProvider(
            settings.pexels_api_key, width=width, height=height, fallback=fallback
        )

    if provider == "pixabay":
        from shorts_agent.visuals.stock import PixabayVisualProvider

        return PixabayVisualProvider(
            settings.pixabay_api_key, width=width, height=height, fallback=fallback
        )

    raise ConfigError(f"Unknown visuals provider: {provider}")
