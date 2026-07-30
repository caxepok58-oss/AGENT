"""Assemble scenes into a finished vertical video.

Two stages, deliberately separated:

1. moviepy composes the visuals and audio into a silent-of-captions master.
2. ffmpeg burns the ASS captions in one pass.

Doing captions in ffmpeg rather than moviepy matters: libass renders text far
more crisply than compositing a text clip per word, and it costs one extra
encode instead of hundreds of per-frame composites.

Every clip is normalised to exactly ``width x height``. Source footage is scaled
to cover the frame and centre-cropped — letterboxing a Short is worse than
losing the edges of a shot, because black bars read as low-effort in the feed.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
from pathlib import Path

from shorts_agent.config import AppConfig
from shorts_agent.exceptions import ShortsAgentError
from shorts_agent.models import SceneAudio, VisualAsset

logger = logging.getLogger(__name__)

# Ken Burns: images start slightly overscanned and drift inward, so a still
# never looks frozen. Small values on purpose — aggressive zoom looks cheap.
ZOOM_START = 1.04
ZOOM_PER_SECOND = 0.018

FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
CRF = "20"


class VideoRenderError(ShortsAgentError):
    """Video composition or encoding failed."""


class VideoAssembler:
    def __init__(self, config: AppConfig):
        self.config = config
        self.width = config.visuals.width
        self.height = config.visuals.height

    # --- public API ---------------------------------------------------

    def assemble(
        self,
        visuals: list[VisualAsset],
        audios: list[SceneAudio],
        output_path: Path,
        *,
        captions_path: Path | None = None,
        work_dir: Path | None = None,
    ) -> Path:
        """Render the final video and return its path."""
        if not visuals or not audios:
            raise VideoRenderError("Cannot assemble a video without visuals and audio")

        work_dir = work_dir or output_path.parent
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        master = work_dir / "master.mp4" if captions_path else output_path
        self._render_master(visuals, audios, master)

        if captions_path and captions_path.exists():
            self._burn_captions(master, captions_path, output_path)
            master.unlink(missing_ok=True)
        elif captions_path:
            logger.warning(
                "Captions file %s missing; shipping video without captions", captions_path
            )

        logger.info("Rendered video: %s", output_path)
        return output_path

    # --- stage 1: composition ----------------------------------------

    def _render_master(
        self, visuals: list[VisualAsset], audios: list[SceneAudio], output_path: Path
    ) -> None:
        try:
            from moviepy import AudioFileClip, CompositeAudioClip, concatenate_videoclips
        except ImportError as exc:
            raise VideoRenderError(
                "moviepy is required to render video. "
                "Install with: pip install 'shorts-agent[video]'"
            ) from exc

        by_index = {audio.scene_index: audio for audio in audios}
        clips = []
        opened: list = []

        try:
            for visual in sorted(visuals, key=lambda v: v.scene_index):
                audio = by_index.get(visual.scene_index)
                if audio is None:
                    logger.warning("Scene %d has no audio; skipping", visual.scene_index)
                    continue

                narration = AudioFileClip(audio.audio_path)
                opened.append(narration)
                # Trust the decoded audio's own duration over the reported one:
                # a mismatch here desynchronises every later scene.
                duration = float(narration.duration or audio.duration_seconds)

                video = self._scene_clip(visual, duration)
                opened.append(video)
                clips.append(video.with_audio(narration))

            if not clips:
                raise VideoRenderError("No scenes could be composed")

            timeline = concatenate_videoclips(clips, method="compose")
            opened.append(timeline)

            music = self._music_clip(timeline.duration)
            if music is not None:
                opened.append(music)
                timeline = timeline.with_audio(CompositeAudioClip([timeline.audio, music]))

            timeline.write_videofile(
                str(output_path),
                fps=FPS,
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                preset="medium",
                ffmpeg_params=["-crf", CRF, "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                logger=None,
            )
        except VideoRenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - moviepy/ffmpeg raise many types
            raise VideoRenderError(f"Video composition failed: {exc}") from exc
        finally:
            for clip in opened:
                try:
                    clip.close()
                except Exception:  # noqa: BLE001 - closing is best-effort cleanup
                    pass

    def _scene_clip(self, visual: VisualAsset, duration: float):
        from moviepy import CompositeVideoClip, ImageClip, VideoFileClip, vfx

        path = Path(visual.path)
        if not path.exists():
            raise VideoRenderError(f"Visual for scene {visual.scene_index} not found: {path}")

        if visual.kind == "video":
            clip = VideoFileClip(str(path)).without_audio()
            if clip.duration < duration:
                # Loop rather than freeze on the last frame: a still image at the
                # end of a scene reads as a glitch.
                clip = clip.with_effects([vfx.Loop(duration=duration)])
            else:
                clip = clip.subclipped(0, duration)
            covered = self._cover(clip)
        else:
            still = ImageClip(str(path)).with_duration(duration)
            covered = self._cover(still).resized(lambda t: ZOOM_START + ZOOM_PER_SECOND * t)

        return CompositeVideoClip(
            [covered.with_position("center")], size=(self.width, self.height)
        ).with_duration(duration)

    def _cover(self, clip):
        """Scale a clip so it fully covers the target frame, preserving aspect."""
        source_w, source_h = clip.size
        scale = max(self.width / source_w, self.height / source_h)
        return clip.resized((round(source_w * scale), round(source_h * scale)))

    def _music_clip(self, duration: float):
        """Pick a background track and mix it well under the narration.

        This is a fixed attenuation, not sidechain ducking — narration runs
        continuously through a Short, so a constant bed is both simpler and
        indistinguishable in practice.
        """
        from moviepy import AudioFileClip, afx

        music_dir = self.config.resolve_path(self.config.visuals.music_dir)
        if not music_dir.exists():
            return None

        tracks = sorted(
            p
            for p in music_dir.iterdir()
            if p.suffix.lower() in {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
        )
        if not tracks:
            return None

        # Chosen at random rather than always the first: a channel that supplies
        # several tracks should not have every video share one bed, which makes
        # a feed of them sound mass-produced.
        track = random.choice(tracks)
        gain = 10 ** (self.config.visuals.music_volume_db / 20)

        try:
            music = AudioFileClip(str(track))
            if music.duration < duration:
                music = music.with_effects([afx.AudioLoop(duration=duration)])
            else:
                music = music.subclipped(0, duration)
            logger.info(
                "Mixing background music %s at %.1f dB",
                track.name,
                self.config.visuals.music_volume_db,
            )
            return music.with_volume_scaled(gain)
        except Exception as exc:  # noqa: BLE001 - music is optional
            logger.warning("Could not load background music %s: %s", track, exc)
            return None

    # --- stage 2: caption burn ---------------------------------------

    def _burn_captions(self, source: Path, captions: Path, output_path: Path) -> None:
        ffmpeg = self._ffmpeg_binary()
        # The ass filter takes a path in a colon-delimited option list, so
        # Windows drive letters and any colons must be escaped.
        filter_path = str(captions.resolve()).replace("\\", "/").replace(":", r"\:")

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"ass={filter_path}",
            "-c:v",
            VIDEO_CODEC,
            "-crf",
            CRF,
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "copy",
            str(output_path),
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise VideoRenderError(
                f"Burning captions failed (ffmpeg exit {result.returncode}): "
                f"{result.stderr.strip()[:800]}"
            )

    @staticmethod
    def _ffmpeg_binary() -> str:
        """Locate ffmpeg, preferring the imageio-ffmpeg build moviepy already uses."""
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - fall back to PATH
            pass

        found = shutil.which("ffmpeg")
        if not found:
            raise VideoRenderError(
                "ffmpeg not found. Install the video extra "
                "(pip install 'shorts-agent[video]') or put ffmpeg on PATH."
            )
        return found
