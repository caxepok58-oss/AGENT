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
    """Stands in for a moviepy clip for testing _cover()'s pure scaling math."""

    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.resized_to: tuple[int, int] | None = None

    def resized(self, new_size: tuple[int, int]) -> FakeClip:
        self.resized_to = new_size
        return self


class FakeResult:
    def __init__(self, returncode: int = 0, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


# --- _cover: pure scaling math ------------------------------------------------


def test_cover_scales_a_landscape_source_to_fill_the_portrait_frame(config):
    """1080x1920 target, 1920x1080 (landscape) source: height must hit the
    target exactly, width must overshoot it so centre-cropping has something
    to trim rather than leaving black bars."""
    assembler = VideoAssembler(config)
    clip = FakeClip((1920, 1080))

    covered = assembler._cover(clip)

    width, height = covered.resized_to
    assert height == 1920
    assert width > 1080


def test_cover_scales_a_narrow_source_to_fill_the_frame_width(config):
    assembler = VideoAssembler(config)
    clip = FakeClip((600, 1900))

    covered = assembler._cover(clip)

    width, height = covered.resized_to
    assert width == 1080
    assert height >= 1920


def test_cover_leaves_an_exact_match_unchanged(config):
    assembler = VideoAssembler(config)
    clip = FakeClip((1080, 1920))

    covered = assembler._cover(clip)

    assert covered.resized_to == (1080, 1920)


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
