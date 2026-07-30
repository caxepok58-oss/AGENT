from __future__ import annotations

from shorts_agent.captions.ass_builder import (
    Overlay,
    _escape,
    _timestamp,
    build_ass,
    chars_per_line,
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


def test_overlay_declares_its_own_style():
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig())
    styles = [line.split(",")[0] for line in document.splitlines() if line.startswith("Style:")]

    assert styles == ["Style: Default", "Style: Overlay"]


def test_overlay_event_uses_the_overlay_style_and_span():
    document = build_ass(
        words(("x", 0.0, 3.0)),
        CaptionsConfig(),
        overlays=[Overlay(text="Sort by amount", start=0.5, end=2.5)],
    )

    event = next(line for line in document.splitlines() if ",Overlay," in line)
    assert event.startswith("Dialogue: 1,0:00:00.50,0:00:02.50,Overlay")
    assert event.endswith("Sort by amount")


def test_overlay_sits_opposite_middle_captions():
    """Overlays must not land on top of the captions."""
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig(position="middle"))
    overlay_style = next(
        line for line in document.splitlines() if line.startswith("Style: Overlay")
    )

    # Field 19 is Alignment; 8 is top-centre.
    assert overlay_style.split(",")[18] == "8"


def test_overlay_moves_to_the_bottom_when_captions_are_at_the_top():
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig(position="top"))
    overlay_style = next(
        line for line in document.splitlines() if line.startswith("Style: Overlay")
    )

    assert overlay_style.split(",")[18] == "2"  # bottom-centre


def test_overlay_renders_smaller_than_the_captions():
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig(font_size=100))
    sizes = {
        line.split(",")[0]: int(line.split(",")[2])
        for line in document.splitlines()
        if line.startswith("Style:")
    }

    assert sizes["Style: Overlay"] < sizes["Style: Default"]


def test_overlay_text_is_escaped():
    document = build_ass(
        words(("x", 0.0, 0.3)),
        CaptionsConfig(),
        overlays=[Overlay(text="{\\b1}injected", start=0.0, end=1.0)],
    )

    event = next(line for line in document.splitlines() if ",Overlay," in line)
    assert "{" not in event.split(",,")[-1]


def test_empty_or_inverted_overlays_are_dropped():
    document = build_ass(
        words(("x", 0.0, 0.3)),
        CaptionsConfig(),
        overlays=[
            Overlay(text="   ", start=0.0, end=1.0),
            Overlay(text="backwards", start=2.0, end=1.0),
        ],
    )

    assert ",Overlay," not in document


def test_no_overlays_produces_no_overlay_events():
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig())

    assert ",Overlay," not in document


def test_long_chunks_wrap_across_lines():
    long_words = words(*[(f"word{i}word", i * 0.3, i * 0.3 + 0.25) for i in range(4)])

    document = build_ass(long_words, CaptionsConfig())

    assert "\\N" in document


def test_line_budget_shrinks_as_the_font_grows():
    """A fixed character budget overflows the frame at large font sizes."""
    assert chars_per_line(1080, 60, 65) > chars_per_line(1080, 90, 65)


def test_line_budget_scales_with_frame_width():
    assert chars_per_line(1920, 90, 65) > chars_per_line(1080, 90, 65)


def test_line_budget_never_collapses_to_nothing():
    """An absurd font size must still leave a usable line, not zero characters."""
    assert chars_per_line(1080, 500, 65) >= 8


def test_libass_is_allowed_to_wrap_within_the_margins():
    """WrapStyle 2 disables wrapping, which let long captions run off the frame."""
    document = build_ass(words(("x", 0.0, 0.3)), CaptionsConfig())

    assert "WrapStyle: 0" in document


def test_a_long_caption_is_split_rather_than_left_on_one_line():
    long_words = words(
        ("forgot.", 0.0, 0.3),
        ("Open", 0.3, 0.6),
        ("your", 0.6, 0.9),
        ("bank", 0.9, 1.2),
    )

    events = [
        line
        for line in build_ass(long_words, CaptionsConfig(font_size=90)).splitlines()
        if line.startswith("Dialogue:")
    ]

    # "forgot. Open your bank" is 22 characters, over the 18-char budget at
    # font size 90, so it must be broken across lines.
    assert "\\N" in events[0]
