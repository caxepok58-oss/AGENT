"""Frame extraction: contact sheets for review, and thumbnail candidates.

This exists because the project asks you to watch every video before publishing,
and opening a file for each run is the friction that makes people skip the step.
A contact sheet shows the whole video at a glance — whether the visuals changed
per scene, whether captions are positioned sensibly, whether anything is
obviously broken — in the time it takes to look at one image.

Frames are pulled with ffmpeg rather than moviepy: a single process reading a
finished file is much faster than re-opening it through a clip abstraction.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FRAME_COUNT = 6
SHEET_COLUMNS = 3
SHEET_TILE_WIDTH = 320
# JPEG quality for thumbnails: high enough to avoid visible artefacts, small
# enough to stay well under YouTube's 2 MB thumbnail limit.
THUMBNAIL_QUALITY = 88


class FrameExtractionError(RuntimeError):
    """A frame could not be read from the video."""


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to PATH
        found = shutil.which("ffmpeg")
        if not found:
            raise FrameExtractionError("ffmpeg not found") from None
        return found


def probe_duration(video_path: Path) -> float:
    """Return the video's duration in seconds, using ffmpeg's own report."""
    result = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    # ffmpeg prints stream info to stderr and exits non-zero with no output file,
    # which is expected here — we only want the Duration line.
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            stamp = line.split("Duration:")[1].split(",")[0].strip()
            try:
                hours, minutes, seconds = stamp.split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            except ValueError:
                break
    raise FrameExtractionError(f"Could not read duration from {video_path}")


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> Path:
    """Write a single frame at ``timestamp`` to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Input seek: far faster than decoding up to the timestamp.
            "-ss",
            f"{max(timestamp, 0):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_path.exists():
        raise FrameExtractionError(
            f"Could not extract a frame at {timestamp:.2f}s: {result.stderr.strip()[:300]}"
        )
    return output_path


def frame_timestamps(duration: float, count: int = DEFAULT_FRAME_COUNT) -> list[float]:
    """Evenly spaced sample points, avoiding the very first and last frames.

    The extremes are skipped because the opening frame is often mid-transition
    and the final one can land after the audio ends, neither of which represents
    the video.
    """
    if duration <= 0 or count < 1:
        return []
    if count == 1:
        return [duration * 0.25]

    margin = duration * 0.04
    span = duration - 2 * margin
    return [margin + span * i / (count - 1) for i in range(count)]


def contact_sheet(
    video_path: Path,
    output_path: Path,
    *,
    count: int = DEFAULT_FRAME_COUNT,
    columns: int = SHEET_COLUMNS,
) -> Path:
    """Render evenly spaced frames into a single reviewable image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise FrameExtractionError(
            "Pillow is required for contact sheets. Install with: pip install -e '.[video]'"
        ) from exc

    duration = probe_duration(video_path)
    stamps = frame_timestamps(duration, count)
    if not stamps:
        raise FrameExtractionError(f"{video_path} has no usable duration")

    work_dir = output_path.parent / ".frames"
    work_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    try:
        for index, stamp in enumerate(stamps):
            frame_path = extract_frame(video_path, stamp, work_dir / f"f{index:02d}.png")
            with Image.open(frame_path) as frame:
                scale = SHEET_TILE_WIDTH / frame.width
                tiles.append(
                    frame.convert("RGB").resize(
                        (SHEET_TILE_WIDTH, round(frame.height * scale)),
                        Image.Resampling.LANCZOS,
                    )
                )

        tile_w, tile_h = tiles[0].size
        rows = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), (18, 18, 18))
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
    finally:
        for tile in tiles:
            tile.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Wrote contact sheet with %d frames: %s", len(stamps), output_path)
    return output_path


def thumbnail(video_path: Path, output_path: Path, *, at_fraction: float = 0.2) -> Path:
    """Pick a representative frame and save it as a JPEG thumbnail.

    Sampled a fifth of the way in: by then the hook is on screen with its
    caption, whereas the opening frames are often still establishing the shot.

    Note that YouTube shows a frame from the video itself in the Shorts feed, so
    a custom thumbnail mainly affects search, browse and the channel page — and
    setting one requires a verified channel.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise FrameExtractionError(
            "Pillow is required for thumbnails. Install with: pip install -e '.[video]'"
        ) from exc

    duration = probe_duration(video_path)
    raw = extract_frame(video_path, duration * at_fraction, output_path.with_suffix(".raw.png"))
    try:
        with Image.open(raw) as frame:
            frame.convert("RGB").save(output_path, "JPEG", quality=THUMBNAIL_QUALITY)
    finally:
        raw.unlink(missing_ok=True)

    logger.info("Wrote thumbnail: %s", output_path)
    return output_path
