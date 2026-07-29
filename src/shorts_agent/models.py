"""Shared data models passed between pipeline stages.

Pydantic models are used throughout (rather than plain dataclasses) because
almost every one of these needs to round-trip through JSON at some point:
LLM structured output, SQLite storage, or the CLI's ``--format json``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


class TrendTopic(BaseModel):
    keyword: str
    score: float = 0.0
    source: str
    region: str | None = None
    sample_titles: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Idea(BaseModel):
    id: str = Field(default_factory=_uuid)
    title: str
    hook: str
    premise: str
    target_emotion: str = ""
    trend_keywords: list[str] = Field(default_factory=list)
    virality_reasoning: str = ""
    niche: str = ""


class Scene(BaseModel):
    index: int
    text: str
    visual_keyword: str
    on_screen_text: str | None = None
    duration_seconds: float | None = None


class Script(BaseModel):
    idea_id: str
    title: str
    scenes: list[Scene]
    cta: str | None = None

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.scenes)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


class WordTiming(BaseModel):
    word: str
    start: float
    end: float


class SceneAudio(BaseModel):
    scene_index: int
    audio_path: str
    word_timings: list[WordTiming]
    duration_seconds: float


class VisualAsset(BaseModel):
    scene_index: int
    path: str
    kind: Literal["video", "image"]
    source: str


class VideoMetadata(BaseModel):
    title: str
    description: str
    tags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    category_id: str = "22"
    privacy_status: Literal["private", "unlisted", "public"] = "private"
    publish_at: datetime | None = None
    made_for_kids: bool = False
    contains_synthetic_media: bool = True
    playlist_id: str | None = None


class PolicyCheckResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class GeneratedVideo(BaseModel):
    run_id: str
    video_path: str
    metadata: VideoMetadata
    script: Script
    idea: Idea
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PublishResult(BaseModel):
    youtube_video_id: str
    url: str
    privacy_status: str
    publish_at: datetime | None = None
