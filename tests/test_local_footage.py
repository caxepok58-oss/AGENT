from __future__ import annotations

import random
from pathlib import Path

import pytest

from shorts_agent.visuals.local import LocalFootageProvider, list_footage, words_match


def touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not really a video; nothing here decodes it")


@pytest.fixture
def footage(tmp_path) -> Path:
    folder = tmp_path / "footage"
    folder.mkdir()
    return folder


def provider(folder: Path, seed: int = 0) -> LocalFootageProvider:
    return LocalFootageProvider(folder, width=32, height=48, rng=random.Random(seed))


def pick(folder: Path, keyword: str, tmp_path: Path, seed: int = 0) -> str:
    return Path(provider(folder, seed).fetch(keyword, tmp_path / "out", 0).path).name


# --- what counts as footage ---------------------------------------------------


def test_videos_and_pictures_are_found_in_subfolders_in_a_stable_order(footage):
    touch(footage, "b.mp4", "a.mov", "sub/c.jpg", "sub/deeper/d.webm", "e.PNG")

    found = [p.relative_to(footage).as_posix() for p in list_footage(footage)]

    assert found == sorted(["b.mp4", "a.mov", "sub/c.jpg", "sub/deeper/d.webm", "e.PNG"])


def test_hidden_empty_and_unsupported_files_are_ignored(footage):
    touch(
        footage, ".DS_Store", "._clip.mp4", ".trash/clip.mp4", "notes.txt", "photo.heic", "ok.mp4"
    )
    (footage / "empty.mp4").write_bytes(b"")

    assert [p.name for p in list_footage(footage)] == ["ok.mp4"]


def test_a_missing_folder_simply_has_no_footage(tmp_path):
    assert list_footage(tmp_path / "nope") == []


def test_a_file_where_the_folder_should_be_has_no_footage(tmp_path):
    (tmp_path / "footage").write_text("a file, not a folder")

    assert list_footage(tmp_path / "footage") == []


# --- matching words -------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [("cash", "cash"), ("coin", "coins"), ("coins", "coin"), ("count", "counting")],
)
def test_words_that_are_equal_or_share_a_long_enough_start_match(a, b):
    assert words_match(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [("car", "cardio"), ("cash", "money"), ("bank", "banner"), ("sun", "sunset")],
)
def test_unrelated_or_too_short_to_judge_words_do_not(a, b):
    # "bank"/"banner" share only "ban", not the whole of the shorter word.
    assert not words_match(a, b)


# --- choosing a clip ---------------------------------------------------------------


def test_the_clip_whose_name_matches_the_keyword_is_chosen(footage, tmp_path):
    touch(footage, "cash_counting.mp4", "sunset_beach.mp4", "city_traffic.mp4")

    for seed in range(5):
        assert pick(footage, "person counting cash at table", tmp_path, seed) == "cash_counting.mp4"


def test_a_folder_name_counts_as_part_of_a_clips_description(footage, tmp_path):
    touch(footage, "money/clip01.mp4", "nature/clip02.mp4")

    for seed in range(5):
        assert pick(footage, "saving money", tmp_path, seed) == "clip01.mp4"


def test_more_matching_words_beat_fewer(footage, tmp_path):
    touch(footage, "cash.mp4", "cash_counting_hands.mp4")

    for seed in range(5):
        assert pick(footage, "counting cash", tmp_path, seed) == "cash_counting_hands.mp4"


def test_a_plural_in_the_file_name_still_matches(footage, tmp_path):
    touch(footage, "gold_coins.mp4", "sunset_beach.mp4")

    for seed in range(5):
        assert pick(footage, "coin", tmp_path, seed) == "gold_coins.mp4"


def test_equally_relevant_clips_are_each_used_once_before_any_is_repeated(footage, tmp_path):
    touch(footage, "money_a.mp4", "money_b.mp4", "money_c.mp4")
    visuals = provider(footage)

    first_three = [Path(visuals.fetch("money", tmp_path, i).path).name for i in range(3)]
    fourth = Path(visuals.fetch("money", tmp_path, 3).path).name

    assert sorted(first_three) == ["money_a.mp4", "money_b.mp4", "money_c.mp4"]
    assert fourth in first_three  # a small library repeats rather than failing


def test_a_clip_that_is_more_relevant_is_reused_rather_than_a_worse_one_taking_its_place(
    footage, tmp_path
):
    touch(footage, "cash_counting.mp4", "beach.mp4")
    visuals = provider(footage)

    names = [Path(visuals.fetch("cash counting", tmp_path, i).path).name for i in range(2)]

    assert names == ["cash_counting.mp4", "cash_counting.mp4"]


def test_the_choice_among_equals_follows_the_random_source(footage, tmp_path):
    touch(footage, *(f"clip_{i}.mp4" for i in range(6)))

    same_seed = {pick(footage, "anything", tmp_path, seed=7) for _ in range(5)}
    different_seeds = {pick(footage, "anything", tmp_path, seed=s) for s in range(30)}

    assert len(same_seed) == 1
    assert len(different_seeds) > 1


def test_a_keyword_nothing_matches_still_gets_a_real_clip(footage, tmp_path, caplog):
    touch(footage, "beach.mp4")

    with caplog.at_level("INFO"):
        asset = provider(footage).fetch("stock market", tmp_path, 2)

    assert asset.source == "local"
    assert Path(asset.path).name == "beach.mp4"
    assert "nothing in footage matches 'stock market'" in caplog.text


# --- what comes back -------------------------------------------------------------


def test_the_asset_points_at_the_original_file_and_carries_no_credit(footage, tmp_path):
    touch(footage, "mine.mp4")

    asset = provider(footage).fetch("mine", tmp_path / "out", 4)

    assert asset.path == str(footage / "mine.mp4")
    assert asset.scene_index == 4
    assert asset.credit is None
    assert not (tmp_path / "out").exists()  # nothing was copied or created


def test_videos_and_pictures_are_told_apart(footage, tmp_path):
    touch(footage, "reel_video.mov", "snap_photo.jpg")
    visuals = provider(footage)

    assert visuals.fetch("reel video", tmp_path, 0).kind == "video"
    assert visuals.fetch("snap photo", tmp_path, 1).kind == "image"


# --- an empty library -----------------------------------------------------------


def test_an_empty_folder_falls_back_to_a_generated_card_and_says_so_once(footage, tmp_path, caplog):
    visuals = provider(footage)

    with caplog.at_level("WARNING"):
        first = visuals.fetch("anything", tmp_path / "out", 0)
        second = visuals.fetch("anything else", tmp_path / "out", 1)

    assert first.source == second.source == "generated"
    assert caplog.text.count("No video or picture files") == 1


def test_a_missing_folder_falls_back_to_a_generated_card(tmp_path):
    visuals = provider(tmp_path / "does-not-exist")

    assert visuals.fetch("anything", tmp_path / "out", 0).source == "generated"
