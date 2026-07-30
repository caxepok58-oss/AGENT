from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from shorts_agent.video.frames import (
    FrameExtractionError,
    contact_sheet,
    extract_frame,
    frame_timestamps,
    probe_duration,
    thumbnail,
)


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    """A real 4-second H.264 file, so ffmpeg behaviour is exercised, not mocked."""
    import imageio_ffmpeg

    path = tmp_path_factory.mktemp("frames") / "clip.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=1080x1920:duration=4:rate=30",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_timestamps_are_ordered_and_inside_the_video():
    stamps = frame_timestamps(10.0, count=5)

    assert stamps == sorted(stamps)
    assert stamps[0] > 0
    assert stamps[-1] < 10.0


def test_timestamps_avoid_the_very_first_and_last_frames():
    """The opening frame is often mid-transition and the last can trail the audio."""
    stamps = frame_timestamps(10.0, count=4)

    assert stamps[0] >= 0.3
    assert stamps[-1] <= 9.7


def test_single_frame_samples_early_not_at_zero():
    assert frame_timestamps(10.0, count=1) == [2.5]


def test_no_timestamps_for_empty_or_invalid_input():
    assert frame_timestamps(0.0) == []
    assert frame_timestamps(-5.0) == []
    assert frame_timestamps(10.0, count=0) == []


def test_probe_duration_reads_the_real_length(video):
    assert probe_duration(video) == pytest.approx(4.0, abs=0.2)


def test_probe_duration_rejects_a_non_video(tmp_path):
    fake = tmp_path / "not-a-video.mp4"
    fake.write_bytes(b"definitely not mp4")

    with pytest.raises(FrameExtractionError, match="Could not read duration"):
        probe_duration(fake)


def test_extract_frame_writes_an_image(video, tmp_path):
    out = extract_frame(video, 1.0, tmp_path / "f.png")

    from PIL import Image

    with Image.open(out) as image:
        assert image.size == (1080, 1920)


def test_extract_frame_beyond_the_end_fails_clearly(video, tmp_path):
    with pytest.raises(FrameExtractionError, match="Could not extract a frame"):
        extract_frame(video, 999.0, tmp_path / "f.png")


def test_contact_sheet_tiles_the_requested_frames(video, tmp_path):
    sheet = contact_sheet(video, tmp_path / "sheet.png", count=6, columns=3)

    from PIL import Image

    with Image.open(sheet) as image:
        width, height = image.size
    # 6 frames in 3 columns is 2 rows of 320-wide tiles.
    assert width == 960
    assert height > width  # vertical source keeps tall tiles


def test_contact_sheet_cleans_up_its_temporary_frames(video, tmp_path):
    contact_sheet(video, tmp_path / "sheet.png", count=3)

    assert not (tmp_path / ".frames").exists()


def test_contact_sheet_handles_a_frame_count_below_the_column_count(video, tmp_path):
    sheet = contact_sheet(video, tmp_path / "sheet.png", count=2, columns=3)

    from PIL import Image

    with Image.open(sheet) as image:
        assert image.size[0] == 960  # still a full row of three tile slots


def test_thumbnail_is_a_jpeg_under_the_youtube_size_limit(video, tmp_path):
    thumb = thumbnail(video, tmp_path / "t.jpg")

    from PIL import Image

    with Image.open(thumb) as image:
        assert image.format == "JPEG"
    assert thumb.stat().st_size < 2_000_000


def test_thumbnail_removes_its_intermediate_png(video, tmp_path):
    thumb = thumbnail(video, tmp_path / "t.jpg")

    assert not thumb.with_suffix(".raw.png").exists()
