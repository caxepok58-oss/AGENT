from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from shorts_agent.video.assembler import VideoAssembler, VideoRenderError


@pytest.fixture(scope="module")
def silent_track(tmp_path_factory) -> Path:
    """A real 2-second silent WAV, so moviepy's AudioFileClip is actually exercised."""
    import imageio_ffmpeg

    path = tmp_path_factory.mktemp("music") / "track.wav"
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
            "anullsrc=r=44100:cl=mono:d=2",
            str(path),
        ],
        check=True,
    )
    return path


class FakeClip:
    """Stands in for a moviepy clip for testing _cover()'s pure geometry."""

    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.crops: list[dict] = []
        self.resized_to: tuple[int, int] | None = None

    def cropped(self, **kwargs) -> FakeClip:
        self.crops.append(kwargs)
        width = kwargs.get("width", self.size[0])
        height = kwargs.get("height", self.size[1])
        cropped = FakeClip((width, height))
        cropped.crops = self.crops
        return cropped

    def resized(self, new_size: tuple[int, int]) -> FakeClip:
        self.resized_to = new_size
        self.size = new_size
        return self


class FakeResult:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


# --- _cover: trim to the frame's aspect ratio first, then scale ----------------


def test_cover_trims_a_landscape_source_to_the_frame_ratio_around_its_centre(config):
    """1080x1920 target, 1920x1080 source: keep the middle 608x1080 (9:16) and
    scale that, instead of enlarging all of it first and throwing most away."""
    assembler = VideoAssembler(config)
    clip = FakeClip((1920, 1080))

    covered = assembler._cover(clip)

    assert clip.crops == [{"x_center": 960, "width": 608}]
    assert covered.resized_to == (1080, 1920)


def test_cover_trims_a_narrow_source_top_and_bottom(config):
    assembler = VideoAssembler(config)
    clip = FakeClip((600, 1900))

    covered = assembler._cover(clip)

    assert clip.crops == [{"y_center": 950, "height": 1067}]
    assert covered.resized_to == (1080, 1920)


def test_cover_leaves_an_exact_match_alone(config):
    assembler = VideoAssembler(config)
    clip = FakeClip((1080, 1920))

    covered = assembler._cover(clip)

    assert clip.crops == []
    assert covered.resized_to is None  # nothing to scale either
    assert covered.size == (1080, 1920)


def test_cover_only_scales_when_the_ratio_already_matches(config):
    assembler = VideoAssembler(config)
    clip = FakeClip((540, 960))  # 9:16, just smaller

    covered = assembler._cover(clip)

    assert clip.crops == []
    assert covered.resized_to == (1080, 1920)


def test_cover_produces_the_exact_frame_and_keeps_the_middle_of_a_wide_shot(config):
    """The real moviepy path: a landscape picture with blue edges and a red middle."""
    import numpy as np
    from moviepy import ImageClip

    config.visuals.width, config.visuals.height = 108, 192
    assembler = VideoAssembler(config)

    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)  # blue everywhere ...
    frame[:, 64:128] = (255, 0, 0)  # ... but the middle third, which is red
    covered = assembler._cover(ImageClip(frame))

    assert covered.size == (108, 192)
    picture = covered.get_frame(0)
    assert picture.shape == (192, 108, 3)
    # The 9:16 window cut from a 192x108 picture is 61 columns wide and centred,
    # so it lies wholly inside the red band: not one blue pixel may survive.
    assert picture[..., 0].min() > 200
    assert picture[..., 2].max() < 60


def test_cover_produces_the_exact_frame_and_keeps_the_middle_of_a_tall_shot(config):
    import numpy as np
    from moviepy import ImageClip

    config.visuals.width, config.visuals.height = 108, 192
    assembler = VideoAssembler(config)

    frame = np.zeros((300, 108, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 255)
    frame[40:260, :] = (255, 0, 0)  # the middle rows; the 192-row window is 54..246
    covered = assembler._cover(ImageClip(frame))

    assert covered.size == (108, 192)
    picture = covered.get_frame(0)
    assert picture[..., 0].min() > 200
    assert picture[..., 2].max() < 60


# --- _burn_captions: the ffmpeg colon-escaping gotcha -------------------------


def test_burn_captions_escapes_colons_in_the_filter_path(config, tmp_path, monkeypatch):
    """ffmpeg's -vf ass=<path> takes a colon-delimited option list, so an
    unescaped colon (e.g. a Windows drive letter) breaks filter parsing."""
    assembler = VideoAssembler(config)
    weird_dir = tmp_path / "weird:dir"
    weird_dir.mkdir()
    captions = weird_dir / "captions.ass"
    captions.write_text("[Script Info]\n")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not a real video, subprocess.run is mocked below")

    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return FakeResult(returncode=0)

    monkeypatch.setattr("shorts_agent.video.assembler.subprocess.run", fake_run)

    assembler._burn_captions(source, captions, tmp_path / "out.mp4")

    vf_arg = captured["command"][captured["command"].index("-vf") + 1]
    assert vf_arg.startswith("ass=")
    assert r"weird\:dir" in vf_arg


def test_burn_captions_raises_with_stderr_on_a_nonzero_exit(config, tmp_path, monkeypatch):
    assembler = VideoAssembler(config)
    captions = tmp_path / "captions.ass"
    captions.write_text("[Script Info]\n")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    monkeypatch.setattr(
        "shorts_agent.video.assembler.subprocess.run",
        lambda *a, **k: FakeResult(returncode=1, stderr="Unrecognized option 'ass'"),
    )

    with pytest.raises(VideoRenderError, match="ffmpeg exit 1"):
        assembler._burn_captions(source, captions, tmp_path / "out.mp4")


# --- _music_clip ---------------------------------------------------------------


def test_music_clip_is_none_when_the_directory_is_absent(config):
    config.visuals.music_dir = "does/not/exist"
    assembler = VideoAssembler(config)

    assert assembler._music_clip(30.0) is None


def test_music_clip_is_none_when_the_directory_has_no_audio_files(config, tmp_path):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    (music_dir / "notes.txt").write_text("not audio")
    config.visuals.music_dir = str(music_dir)
    assembler = VideoAssembler(config)

    assert assembler._music_clip(30.0) is None


def test_music_clip_loops_a_short_track_to_cover_the_requested_duration(
    config, tmp_path, silent_track
):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    shutil.copy(silent_track, music_dir / "track.wav")
    config.visuals.music_dir = str(music_dir)
    assembler = VideoAssembler(config)

    clip = assembler._music_clip(5.0)  # longer than the 2s source track, so it must loop

    assert clip is not None
    try:
        assert clip.duration == pytest.approx(5.0, abs=0.1)
    finally:
        clip.close()


def test_music_clip_subclips_a_long_track_down_to_the_requested_duration(
    config, tmp_path, silent_track
):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    shutil.copy(silent_track, music_dir / "track.wav")
    config.visuals.music_dir = str(music_dir)
    assembler = VideoAssembler(config)

    clip = assembler._music_clip(0.5)  # shorter than the 2s source track

    assert clip is not None
    try:
        assert clip.duration == pytest.approx(0.5, abs=0.05)
    finally:
        clip.close()


# --- scene clips built from real files ------------------------------------------------


def make_video(path: Path, *, size: str, seconds: float) -> Path:
    import imageio_ffmpeg

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
            f"testsrc2=size={size}:rate=10:duration={seconds}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def small_frame(config) -> None:
    config.visuals.width, config.visuals.height = 108, 192


def video_visual(path: Path):
    from shorts_agent.models import VisualAsset

    return VisualAsset(scene_index=0, path=str(path), kind="video", source="test")


def image_visual(path: Path):
    from shorts_agent.models import VisualAsset

    return VisualAsset(scene_index=0, path=str(path), kind="image", source="test")


def test_a_landscape_video_comes_out_exactly_frame_sized(config, tmp_path):
    small_frame(config)
    source = make_video(tmp_path / "wide.mp4", size="384x216", seconds=2)

    clip = VideoAssembler(config)._scene_clip(video_visual(source), 1.0)
    try:
        assert clip.size == (108, 192)
        assert clip.duration == pytest.approx(1.0)
        assert clip.get_frame(0.5).shape == (192, 108, 3)
    finally:
        clip.close()


def test_a_clip_shorter_than_its_scene_is_looped_not_frozen(config, tmp_path):
    small_frame(config)
    source = make_video(tmp_path / "short.mp4", size="108x192", seconds=1)

    clip = VideoAssembler(config)._scene_clip(video_visual(source), 2.5)
    try:
        assert clip.duration == pytest.approx(2.5)
        # A frame from the second pass through the clip must be readable.
        assert clip.get_frame(2.0).shape == (192, 108, 3)
    finally:
        clip.close()


def test_a_long_clip_is_used_from_a_random_point_not_always_its_start(
    config, tmp_path, monkeypatch
):
    small_frame(config)
    source = make_video(tmp_path / "long.mp4", size="108x192", seconds=6)
    windows = []

    def fake_uniform(low, high):
        windows.append((low, high))
        return high  # the latest possible start: the very end of the clip

    monkeypatch.setattr("shorts_agent.video.assembler.random.uniform", fake_uniform)

    clip = VideoAssembler(config)._scene_clip(video_visual(source), 2.0)
    try:
        assert windows == [(0, pytest.approx(4.0, abs=0.2))]
        assert clip.duration == pytest.approx(2.0)
        assert clip.get_frame(1.9).shape == (192, 108, 3)  # the last frame of the window exists
    finally:
        clip.close()


def test_a_sideways_phone_photo_is_stood_upright(config, tmp_path):
    """EXIF orientation 6 means "rotate 90° clockwise to display": a 40x20 file
    on disk is a 20x40 picture."""
    from PIL import Image

    photo = tmp_path / "phone.jpg"
    picture = Image.new("RGB", (40, 20), (200, 30, 30))
    exif = picture.getexif()
    exif[0x0112] = 6
    picture.save(photo, exif=exif)

    clip = VideoAssembler(config)._load_still(photo)

    assert clip.size == (20, 40)


def test_a_palette_png_becomes_ordinary_colour(config, tmp_path):
    from PIL import Image

    path = tmp_path / "indexed.png"
    Image.new("P", (16, 16), 1).save(path)

    frame = VideoAssembler(config)._load_still(path).get_frame(0)

    assert frame.shape == (16, 16, 3)


def test_a_huge_photo_is_shrunk_before_the_zoom_resamples_it_every_frame(config, tmp_path):
    from PIL import Image

    path = tmp_path / "huge.jpg"
    Image.new("RGB", (6000, 4000), (10, 120, 200)).save(path, quality=50)

    clip = VideoAssembler(config)._load_still(path)

    assert max(clip.size) == max(config.visuals.width, config.visuals.height) * 2
    assert clip.size[0] / clip.size[1] == pytest.approx(1.5, abs=0.01)  # same shape


def test_a_picture_smaller_than_the_frame_is_not_enlarged_on_load(config, tmp_path):
    from PIL import Image

    path = tmp_path / "tiny.png"
    Image.new("RGB", (30, 50), (1, 2, 3)).save(path)

    assert VideoAssembler(config)._load_still(path).size == (30, 50)


def test_a_whole_video_renders_from_footage_of_mixed_shapes_and_pictures(
    config, tmp_path, silent_track
):
    """End to end through moviepy and ffmpeg with real files: a landscape clip, a
    still, and the caption burn, at a small size so it takes a moment, not minutes."""
    from PIL import Image

    from shorts_agent.models import SceneAudio, VisualAsset

    small_frame(config)
    clip_path = make_video(tmp_path / "wide.mp4", size="384x216", seconds=3)
    photo_path = tmp_path / "photo.png"
    Image.new("RGB", (90, 160), (30, 90, 200)).save(photo_path)

    visuals = [
        VisualAsset(scene_index=0, path=str(clip_path), kind="video", source="test"),
        VisualAsset(scene_index=1, path=str(photo_path), kind="image", source="test"),
    ]
    audios = [
        SceneAudio(
            scene_index=i, audio_path=str(silent_track), word_timings=[], duration_seconds=2.0
        )
        for i in range(2)
    ]
    output = tmp_path / "out" / "video.mp4"

    VideoAssembler(config).assemble(visuals, audios, output, work_dir=tmp_path / "work")

    from moviepy import VideoFileClip

    with VideoFileClip(str(output)) as rendered:
        assert tuple(rendered.size) == (108, 192)
        assert rendered.duration == pytest.approx(4.0, abs=0.3)
        assert rendered.get_frame(1.0).shape == (192, 108, 3)  # inside the footage scene
        assert rendered.get_frame(3.0).shape == (192, 108, 3)  # inside the picture scene
