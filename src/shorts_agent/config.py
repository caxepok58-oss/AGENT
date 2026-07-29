"""Configuration loading.

Two layers, deliberately kept separate:

* :class:`Settings` — secrets and machine-specific file paths, sourced from
  environment variables / ``.env``. Never checked into git.
* :class:`AppConfig` — content and behavior settings (niche, persona, safety
  limits, provider selection), sourced from a YAML file that *is* safe to
  version-control per channel.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shorts_agent.exceptions import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    tts_provider: Literal["edge", "elevenlabs"] = "edge"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None

    pexels_api_key: str | None = None
    pixabay_api_key: str | None = None

    google_trends_enabled: bool = True

    # Read-only trend research needs only an API key; uploading needs OAuth.
    youtube_api_key: str | None = None
    youtube_client_secrets_file: str = "./client_secret.json"
    youtube_token_file: str = "./youtube_token.json"

    shorts_agent_config: str = "./config/config.yaml"
    shorts_agent_db: str | None = None
    shorts_agent_output_dir: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


class ChannelConfig(BaseModel):
    name: str
    niche: str
    language: str = "en"
    audience: str = "general"
    persona: str
    made_for_kids: bool = False
    category_id: str = "22"
    default_hashtags: list[str] = Field(default_factory=lambda: ["#Shorts"])


class ContentConfig(BaseModel):
    target_duration_seconds: float = 45
    min_duration_seconds: float = 15
    max_duration_seconds: float = 90
    ideas_per_run: int = 5
    words_per_second: float = 2.3
    add_outro_cta: bool = True
    outro_text: str = "Follow for more!"
    ai_disclosure: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> ContentConfig:
        if self.max_duration_seconds > 180:
            raise ConfigError("max_duration_seconds cannot exceed 180s (YouTube Shorts limit)")
        if not (self.min_duration_seconds <= self.target_duration_seconds <= self.max_duration_seconds):
            raise ConfigError(
                "target_duration_seconds must be between min_duration_seconds and max_duration_seconds"
            )
        return self


class TrendsConfig(BaseModel):
    providers: list[Literal["youtube", "google_trends", "manual"]] = Field(
        default_factory=lambda: ["youtube", "manual"]
    )
    youtube_region_code: str = "US"
    manual_keywords_file: str | None = None
    lookback_days: int = 5
    max_topics: int = 15


class ProvidersConfig(BaseModel):
    llm: Literal["anthropic", "openai"] = "anthropic"
    tts: Literal["edge", "elevenlabs"] = "edge"
    visuals: Literal["pexels", "pixabay", "generated"] = "generated"
    edge_tts_voice: str = "en-US-AndrewNeural"


class VisualsConfig(BaseModel):
    width: int = 1080
    height: int = 1920
    music_dir: str = "config/music"
    music_volume_db: float = -18


class CaptionsConfig(BaseModel):
    enabled: bool = True
    # Resolved by fontconfig at render time, so it must name a font installed on
    # the machine doing the render. DejaVu Sans ships with essentially every
    # Linux distribution; "Arial" or "Helvetica" suit macOS and Windows.
    font: str = "DejaVu Sans"
    font_size: int = 90
    highlight_color: str = "&H0000D7FF"
    base_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    position: Literal["top", "middle", "bottom"] = "middle"


class PolicyConfig(BaseModel):
    blocklist_file: str | None = None
    use_llm_moderation: bool = True
    disallow_duplicate_topics_days: int = 30


class PublishingConfig(BaseModel):
    privacy_status: Literal["private", "unlisted", "public"] = "private"
    auto_publish: bool = False
    max_uploads_per_day: int = 2
    default_publish_delay_hours: float = 4
    playlist_id: str = ""

    @field_validator("max_uploads_per_day")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 0:
            raise ConfigError("max_uploads_per_day cannot be negative")
        return v


class StorageConfig(BaseModel):
    db_path: str = "data/shorts_agent.db"
    output_dir: str = "output"


class AppConfig(BaseModel):
    channel: ChannelConfig
    content: ContentConfig = Field(default_factory=ContentConfig)
    trends: TrendsConfig = Field(default_factory=TrendsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # Relative paths inside the YAML resolve against this, which defaults to
    # the process working directory — so a config written with paths like
    # "data/agent.db" behaves the same however the config file is located.
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    def resolve_path(self, relative: str) -> Path:
        p = Path(relative).expanduser()
        return p if p.is_absolute() else (self.base_dir / p)


def load_config(path: str | Path | None = None, base_dir: Path | None = None) -> AppConfig:
    settings = get_settings()
    config_path = Path(path or settings.shorts_agent_config)
    if not config_path.exists():
        raise ConfigError(
            f"Config file not found: {config_path}. Copy config/config.example.yaml to "
            "config/config.yaml (or set SHORTS_AGENT_CONFIG) and edit it."
        )

    raw = yaml.safe_load(config_path.read_text()) or {}
    try:
        config = AppConfig.model_validate({**raw, "base_dir": base_dir or Path.cwd()})
    except ConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as our own error type
        raise ConfigError(f"Invalid config at {config_path}: {exc}") from exc

    # Environment overrides for deployment-specific paths.
    if settings.shorts_agent_db:
        config.storage.db_path = settings.shorts_agent_db
    if settings.shorts_agent_output_dir:
        config.storage.output_dir = settings.shorts_agent_output_dir

    return config


def ensure_runtime_dirs(config: AppConfig) -> None:
    db_dir = config.resolve_path(config.storage.db_path).parent
    output_dir = config.resolve_path(config.storage.output_dir)
    for d in (db_dir, output_dir):
        os.makedirs(d, exist_ok=True)
