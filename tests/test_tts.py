from __future__ import annotations

from shorts_agent.tts.base import approximate_word_timings


def test_timings_cover_the_whole_duration():
    timings = approximate_word_timings("one two three", 3.0)

    assert timings[0].start == 0.0
    assert timings[-1].end == 3.0


def test_timings_are_contiguous_and_ordered():
    timings = approximate_word_timings("alpha beta gamma delta", 4.0)

    for earlier, later in zip(timings, timings[1:], strict=False):
        assert earlier.end <= later.start + 0.001
        assert earlier.start < earlier.end


def test_longer_words_get_more_time():
    timings = approximate_word_timings("a extraordinarily", 2.0)

    short, long = timings
    assert (long.end - long.start) > (short.end - short.start)


def test_word_order_is_preserved():
    timings = approximate_word_timings("first second third", 1.5)

    assert [t.word for t in timings] == ["first", "second", "third"]


def test_empty_text_yields_no_timings():
    assert approximate_word_timings("", 5.0) == []
    assert approximate_word_timings("   ", 5.0) == []


def test_non_positive_duration_yields_no_timings():
    assert approximate_word_timings("some words", 0.0) == []
    assert approximate_word_timings("some words", -1.0) == []


def test_single_word_spans_the_full_duration():
    timings = approximate_word_timings("solo", 2.5)

    assert len(timings) == 1
    assert timings[0].start == 0.0
    assert timings[0].end == 2.5
