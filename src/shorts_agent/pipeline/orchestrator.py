"""The pipeline that turns trend signals into a published Short.

Stage order and why it matters:

``trends -> ideas -> policy -> script -> policy -> audio -> visuals -> captions
-> video -> metadata -> publish``

Policy runs twice, and both times *before* anything expensive. Rejecting an idea
costs one LLM call; rejecting a finished video wastes a full render and, worse,
risks an upload that damages the channel.

Publishing is opt-in at three levels: ``publish=False`` by default, uploads land
as ``private`` unless configured otherwise, and a daily upload cap is enforced
against real upload history. See docs/POLICY.md for why those defaults are what
they are.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from shorts_agent.captions import write_ass
from shorts_agent.config import AppConfig, ensure_runtime_dirs, get_settings
from shorts_agent.exceptions import PolicyViolation, RateLimitExceeded, ShortsAgentError
from shorts_agent.ideation import IdeaGenerator, ScriptWriter
from shorts_agent.llm import build_llm_client
from shorts_agent.metadata import MetadataGenerator
from shorts_agent.models import (
    GeneratedVideo,
    Idea,
    PublishResult,
    SceneAudio,
    Script,
    TrendTopic,
    VideoMetadata,
    VisualAsset,
    WordTiming,
)
from shorts_agent.policy import PolicyGuard
from shorts_agent.storage import Storage, dump_json
from shorts_agent.trends import TrendAggregator, build_trend_providers
from shorts_agent.tts import build_tts_provider
from shorts_agent.video import VideoAssembler
from shorts_agent.visuals import build_visual_provider

logger = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    run_id: str
    idea: Idea | None = None
    script: Script | None = None
    metadata: VideoMetadata | None = None
    video_path: str | None = None
    publish: PublishResult | None = None
    topics: list[TrendTopic] = Field(default_factory=list)
    skipped_ideas: list[dict] = Field(default_factory=list)
    estimated_duration_seconds: float | None = None


class Pipeline:
    def __init__(self, config: AppConfig, *, llm: Any = None, storage: Storage | None = None):
        self.config = config
        ensure_runtime_dirs(config)
        self.storage = storage or Storage(config.resolve_path(config.storage.db_path))
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Built lazily so trend-only commands work without an LLM key."""
        if self._llm is None:
            self._llm = build_llm_client(self.config)
        return self._llm

    # --- stages -------------------------------------------------------

    def research(self, limit: int | None = None) -> list[TrendTopic]:
        aggregator = TrendAggregator(build_trend_providers(self.config))
        return aggregator.collect(
            self.config.channel.niche, limit=limit or self.config.trends.max_topics
        )

    def ideate(self, topics: list[TrendTopic], count: int | None = None) -> list[Idea]:
        recent = self.storage.recent_topics(self.config.policy.disallow_duplicate_topics_days)
        generator = IdeaGenerator(self.llm, self.config)
        return generator.generate(
            topics,
            count=count,
            recent_topics=recent,
            top_performers=self.storage.top_performers(),
        )

    def select_idea(self, ideas: list[Idea]) -> tuple[Idea, list[dict]]:
        """Return the first idea that clears policy, plus what was rejected."""
        guard = self._guard()
        skipped: list[dict] = []

        for idea in ideas:
            result = guard.check_idea(idea)
            if result.passed:
                return idea, skipped
            logger.warning("Idea %r rejected: %s", idea.title, "; ".join(result.reasons))
            skipped.append({"title": idea.title, "reasons": result.reasons})

        raise PolicyViolation(
            [f"all {len(ideas)} generated ideas were rejected"]
            + [r for s in skipped for r in s["reasons"]]
        )

    def write_script(self, idea: Idea) -> Script:
        script = ScriptWriter(self.llm, self.config).write(idea)

        result = self._guard().check_script(script, idea)
        if not result.passed:
            raise PolicyViolation(result.reasons)

        return script

    def produce(self, script: Script, run_dir: Path) -> tuple[Path, float]:
        """Render the video for ``script`` and return its path and duration."""
        tts = build_tts_provider(self.config)
        visuals_provider = build_visual_provider(self.config)

        audios: list[SceneAudio] = []
        visuals: list[VisualAsset] = []
        timings: list[WordTiming] = []
        offset = 0.0

        for scene in script.scenes:
            audio = tts.synthesize(
                scene.text, run_dir / "audio" / f"scene_{scene.index:02d}.mp3", scene.index
            )
            audios.append(audio)

            # Scene timings are relative to their own audio file; the caption
            # track spans the whole video, so shift each scene by the running
            # offset before merging.
            timings.extend(
                w.model_copy(update={"start": w.start + offset, "end": w.end + offset})
                for w in audio.word_timings
            )
            offset += audio.duration_seconds

            visuals.append(
                visuals_provider.fetch(
                    scene.visual_keyword,
                    run_dir / "visuals",
                    scene.index,
                    text=scene.on_screen_text,
                )
            )
            logger.info(
                "Scene %d: %.2fs audio, visual from %s",
                scene.index,
                audio.duration_seconds,
                visuals[-1].source,
            )

        captions_path = None
        if self.config.captions.enabled and timings:
            captions_path = write_ass(
                timings,
                run_dir / "captions.ass",
                self.config.captions,
                width=self.config.visuals.width,
                height=self.config.visuals.height,
            )

        video_path = VideoAssembler(self.config).assemble(
            visuals,
            audios,
            run_dir / "video.mp4",
            captions_path=captions_path,
            work_dir=run_dir / "work",
        )
        return video_path, offset

    def build_metadata(self, idea: Idea, script: Script) -> VideoMetadata:
        return MetadataGenerator(self.llm, self.config).generate(idea, script)

    def publish(self, video: GeneratedVideo, *, schedule: bool = True) -> PublishResult:
        """Upload ``video``, enforcing the configured daily cap."""
        from shorts_agent.youtube import YouTubeUploader, get_youtube_service

        self._check_upload_quota()

        metadata = video.metadata
        if schedule and self.config.publishing.default_publish_delay_hours > 0:
            metadata.publish_at = self._next_slot()
            # publishAt is only honoured on private videos, so scheduling
            # implies private regardless of the configured status.
            metadata.privacy_status = "private"
            logger.info("Scheduling publication for %s", metadata.publish_at.isoformat())

        settings = get_settings()
        service = get_youtube_service(
            settings.youtube_client_secrets_file, settings.youtube_token_file
        )
        result = YouTubeUploader(service).upload(Path(video.video_path), metadata)

        self.storage.record_upload(
            youtube_video_id=result.youtube_video_id,
            run_id=video.run_id,
            privacy_status=result.privacy_status,
            title=metadata.title,
            topic=video.idea.title,
            publish_at=metadata.publish_at,
        )
        logger.info("Uploaded: %s", result.url)
        return result

    # --- full run -----------------------------------------------------

    def run(
        self,
        *,
        publish: bool = False,
        schedule: bool = True,
        idea_count: int | None = None,
    ) -> PipelineResult:
        run_id = uuid.uuid4().hex[:12]
        run_dir = self.config.resolve_path(self.config.storage.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self.storage.start_run(run_id, self.config.channel.niche)
        result = PipelineResult(run_id=run_id)

        try:
            if publish:
                # Check before spending anything: discovering the cap after a
                # full render wastes the whole run.
                self._check_upload_quota()

            result.topics = self.research()
            logger.info("Collected %d trend topics", len(result.topics))

            ideas = self.ideate(result.topics, count=idea_count)
            if not ideas:
                raise ShortsAgentError("No usable ideas were generated")

            idea, skipped = self.select_idea(ideas)
            result.idea, result.skipped_ideas = idea, skipped
            self.storage.update_run(
                run_id, idea_title=idea.title, topic=idea.title, status="ideated"
            )
            logger.info("Selected idea: %s", idea.title)

            script = self.write_script(idea)
            result.script = script
            result.estimated_duration_seconds = round(
                script.word_count / self.config.content.words_per_second, 2
            )
            self.storage.update_run(run_id, script_json=dump_json(script), status="scripted")

            video_path, _ = self.produce(script, run_dir)
            result.video_path = str(video_path)
            self.storage.update_run(run_id, video_path=str(video_path), status="rendered")

            metadata = self.build_metadata(idea, script)
            result.metadata = metadata
            self.storage.update_run(run_id, metadata_json=dump_json(metadata), status="ready")

            if publish:
                generated = GeneratedVideo(
                    run_id=run_id,
                    video_path=str(video_path),
                    metadata=metadata,
                    script=script,
                    idea=idea,
                )
                result.publish = self.publish(generated, schedule=schedule)
                self.storage.update_run(run_id, status="published")
            else:
                logger.info(
                    "Video ready for review at %s — not uploaded. Pass --publish to upload.",
                    video_path,
                )

            return result

        except Exception as exc:
            self.storage.update_run(run_id, status="failed", error=str(exc))
            raise

    def refresh_stats(self) -> int:
        """Pull view counts for past uploads so ideation can learn from them."""
        from shorts_agent.youtube import fetch_video_stats, get_youtube_service

        video_ids = self.storage.uploads_needing_stats()
        if not video_ids:
            return 0

        settings = get_settings()
        service = get_youtube_service(
            settings.youtube_client_secrets_file, settings.youtube_token_file
        )
        stats = fetch_video_stats(service, video_ids)

        for video_id, values in stats.items():
            self.storage.record_stats(
                video_id, values["views"], values["likes"], values["comments"]
            )
        return len(stats)

    # --- helpers ------------------------------------------------------

    def _guard(self) -> PolicyGuard:
        return PolicyGuard(
            self.config,
            llm=self.llm if self.config.policy.use_llm_moderation else None,
            recent_topics=self.storage.recent_topics(
                self.config.policy.disallow_duplicate_topics_days
            ),
        )

    def _check_upload_quota(self) -> None:
        cap = self.config.publishing.max_uploads_per_day
        if cap <= 0:
            raise RateLimitExceeded("publishing.max_uploads_per_day is 0, so uploading is disabled")

        since = datetime.utcnow() - timedelta(days=1)
        used = self.storage.count_uploads_since(since)
        if used >= cap:
            raise RateLimitExceeded(
                f"Daily upload limit reached ({used}/{cap} in the last 24h). "
                "Raise publishing.max_uploads_per_day if this is intentional."
            )

    def _next_slot(self) -> datetime:
        """Pick the next publish time, spacing runs out rather than stacking them."""
        delay = timedelta(hours=self.config.publishing.default_publish_delay_hours)
        earliest = datetime.now(timezone.utc) + delay

        latest = self.storage.latest_publish_at()
        if latest is not None:
            if latest.tzinfo is None:
                latest = latest.replace(tzinfo=timezone.utc)
            # Keep at least the configured gap between scheduled videos so a
            # burst of runs doesn't publish several Shorts at once.
            spaced = latest + delay
            if spaced > earliest:
                return spaced
        return earliest
