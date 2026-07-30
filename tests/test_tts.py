from __future__ import annotations

from shorts_agent.models import WordTiming
from shorts_agent.tts.base import approximate_word_timings
from shorts_agent.tts.edge_tts_provider import EdgeTTSProvider
from shorts_agent.tts.factory import build_tts_provider


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


def test_offset_shifts_the_whole_span():
    timings = approximate_word_timings("one two", 2.0, offset=10.0)

    assert timings[0].start == 10.0
    assert timings[-1].end == 12.0


class _Provider(EdgeTTSProvider):
    """Exercises timing resolution without touching the network."""

    def __init__(self, probed: float | None = None):
        super().__init__()
        self._probed = probed

    def _probe_duration(self, path):  # type: ignore[override]
        return self._probed


def timing(word, start, end):
    return WordTiming(word=word, start=start, end=end)


def test_word_boundaries_are_used_verbatim_when_available(tmp_path):
    words = [timing("hello", 0.0, 0.4), timing("world", 0.4, 0.9)]

    resolved, source = _Provider()._resolve_timings("hello world", tmp_path / "a.mp3", words, [])

    assert resolved == words
    assert source == "WordBoundary"


def test_sentence_boundaries_anchor_words_to_real_spans(tmp_path):
    """Without per-word events, each sentence's real start/duration still anchors captions."""
    sentences = [timing("one two.", 1.0, 3.0), timing("three four.", 3.0, 5.0)]

    resolved, source = _Provider()._resolve_timings("ignored", tmp_path / "a.mp3", [], sentences)

    assert source == "SentenceBoundary"
    assert [t.word for t in resolved] == ["one", "two.", "three", "four."]
    # Words stay inside their own sentence's measured span.
    assert resolved[0].start == 1.0
    assert resolved[1].end == 3.0
    assert resolved[2].start == 3.0
    assert resolved[-1].end == 5.0


def test_measured_duration_is_the_last_resort(tmp_path):
    resolved, source = _Provider(probed=6.0)._resolve_timings(
        "one two three", tmp_path / "a.mp3", [], []
    )

    assert source == "measured duration"
    assert resolved[-1].end == 6.0


def test_word_boundaries_win_over_sentence_boundaries(tmp_path):
    words = [timing("exact", 0.0, 0.5)]
    sentences = [timing("exact", 0.0, 9.0)]

    _, source = _Provider()._resolve_timings("exact", tmp_path / "a.mp3", words, sentences)

    assert source == "WordBoundary"


def test_language_voice_mismatch_is_warned_about(config, caplog):
    """An English voice reading Russian text produces confident nonsense."""
    config.channel.language = "ru"
    config.providers.edge_tts_voice = "en-US-AndrewNeural"

    with caplog.at_level("WARNING"):
        build_tts_provider(config)

    assert "channel.language" in caplog.text


def test_matching_language_and_voice_is_silent(config, caplog):
    config.channel.language = "ru"
    config.providers.edge_tts_voice = "ru-RU-DmitryNeural"

    with caplog.at_level("WARNING"):
        build_tts_provider(config)

    assert caplog.text == ""


def test_regional_variants_do_not_warn(config, caplog):
    """en-GB content with an en-US voice is a style choice, not an error."""
    config.channel.language = "en-GB"
    config.providers.edge_tts_voice = "en-US-AndrewNeural"

    with caplog.at_level("WARNING"):
        build_tts_provider(config)

    assert caplog.text == ""
