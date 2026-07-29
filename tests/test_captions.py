from __future__ import annotations

from shorts_agent.captions.ass_builder import (
    _escape,
    _timestamp,
    build_ass,
    chunk_timings,
    write_ass,
)
from shorts_agent.config import CaptionsConfig
from shorts_agent.models import WordTiming


def words(*specs):
    return [WordTiming(word=w, start=s, end=e) for w, s, e in specs]


def test_timestamp_formats_ass_centiseconds():
    assert _timestamp(0) == "0:00:00.00"
    assert _timestamp(3.456) == "0:00:03.46"
    assert _timestamp(61.5) == "0:01:01.50"
    assert _timestamp(3661.0) == "1:01:01.00"


def test_timestamp_handles_rounding_carry():
    """0.999s must not render as the invalid .100 centiseconds."""
    assert _timestamp(3.999) == "0:00:04.00"


def test_timestamp_clamps_negative_values():
    assert _timestamp(-5) == "0:00:00.00"


def test_escape_neutralizes_ass_markup():
    escaped = _escape("say {\\b1}bold{\\b0} now")

    assert "{" not in escaped
    assert "}" not in escaped
    assert "\\" not in escaped


def test_escape_flattens_newlines():
    assert "\n" not in _escape("line one\nline two")


def test_chunking_respects_word_limit():
    timings = words(*[(f"w{i}", i * 0.2, i * 0.2 + 0.15) for i in range(9)])

    chunks = chunk_timings(timings, words_per_chunk=4)

    assert [len(c) for c in chunks] == [4, 4, 1]


def test_chunking_breaks_on_a_long_pause():
    timings = words(("one", 0.0, 0.3), ("two", 0.4, 0.7), ("three", 2.0, 2.3))

    chunks = chunk_timings(timings, words_per_chunk=8)

    assert [[w.word for w in c] for c in chunks] == [["one", "two"], ["three"]]


def test_build_ass_emits_one_event_per_word():
    timings = words(("hello", 0.0, 0.4), ("world", 0.4, 0.9))

    document = build_ass(timings, CaptionsConfig())
    events = [line for line in document.splitlines() if line.startswith("Dialogue:")]

    assert len(events) == 2


def test_build_ass_highlights_the_active_word_only():
    timings = words(("alpha", 0.0, 0.4), ("beta", 0.4, 0.9))
    config = CaptionsConfig(highlight_color="&H0000FFFF", base_color="&H00FFFFFF")

    events = [
        line for line in build_ass(timings, config).splitlines() if line.startswith("Dialogue:")
    ]

    # The first event highlights "alpha"; the second highlights "beta".
    assert "{\\c&H0000FFFF&}alpha" in events[0]
    assert "{\\c&H0000FFFF&}beta" in events[1]
    assert "{\\c&H0000FFFF&}beta" not in events[0]


def test_inline_colour_tags_are_terminated():
    """ASS inline colour overrides require a trailing '&'."""
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig(highlight_color="&H0000D7FF"))

    assert "{\\c&H0000D7FF&}" in document


def test_header_carries_frame_size_and_style():
    document = build_ass(
        words(("x", 0.0, 0.3)), CaptionsConfig(font="Roboto", font_size=72), width=720, height=1280
    )

    assert "PlayResX: 720" in document
    assert "PlayResY: 1280" in document
    assert "Roboto,72" in document


def test_position_maps_to_ass_alignment():
    timings = words(("x", 0.0, 0.3))

    top = build_ass(timings, CaptionsConfig(position="top"))
    middle = build_ass(timings, CaptionsConfig(position="middle"))
    bottom = build_ass(timings, CaptionsConfig(position="bottom"))

    assert ",8," in top.split("Style: Default,")[1]
    assert ",5," in middle.split("Style: Default,")[1]
    assert ",2," in bottom.split("Style: Default,")[1]


def test_highlight_is_held_until_the_next_word_starts():
    """Captions must not blink off during the silence between words."""
    timings = words(("one", 0.0, 0.3), ("two", 0.5, 0.9))

    events = [
        line
        for line in build_ass(timings, CaptionsConfig()).splitlines()
        if line.startswith("Dialogue:")
    ]

    # First event ends when the second word begins (0.50), not at 0.30.
    assert events[0].split(",")[2] == "0:00:00.50"


def test_finished_phrase_holds_into_the_pause_but_is_capped():
    """A 0.7s pause splits phrases; the first should linger, not vanish or stick."""
    timings = words(("one", 0.0, 0.3), ("two", 1.0, 1.4))

    events = [
        line
        for line in build_ass(timings, CaptionsConfig()).splitlines()
        if line.startswith("Dialogue:")
    ]

    # Held past its own end (0.30) by MAX_PHRASE_HOLD, stopping before 1.00.
    assert events[0].split(",")[2] == "0:00:00.90"


def test_final_phrase_is_not_extended_past_the_audio():
    timings = words(("only", 0.0, 0.4))

    events = [
        line
        for line in build_ass(timings, CaptionsConfig()).splitlines()
        if line.startswith("Dialogue:")
    ]

    assert events[0].split(",")[2] == "0:00:00.40"


def test_zero_length_word_gets_a_minimum_visible_duration():
    events = [
        line
        for line in build_ass(words(("x", 1.0, 1.0)), CaptionsConfig()).splitlines()
        if line.startswith("Dialogue:")
    ]

    start, end = events[0].split(",")[1], events[0].split(",")[2]
    assert start != end


def test_empty_timings_produce_a_valid_but_empty_document():
    document = build_ass([], CaptionsConfig())

    assert "[Events]" in document
    assert "Dialogue:" not in document


def test_write_ass_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "deeper" / "captions.ass"

    write_ass(words(("x", 0.0, 0.3)), target, CaptionsConfig())

    assert target.exists()
    assert "[Script Info]" in target.read_text(encoding="utf-8")


def test_long_chunks_wrap_across_lines():
    long_words = words(*[(f"word{i}word", i * 0.3, i * 0.3 + 0.25) for i in range(4)])

    document = build_ass(long_words, CaptionsConfig())

    assert "\\N" in document
