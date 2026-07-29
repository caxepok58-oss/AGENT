"""Text-to-speech interface, including the word timings captions depend on.

Word-level timing is the reason this interface returns more than an audio path.
Burned-in captions that highlight the currently spoken word are what short-form
video looks like now, and getting them right requires knowing when each word is
actually spoken — not guessing from character counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from shorts_agent.models import SceneAudio, WordTiming


class TTSProvider(ABC):
    name: str = "unknown"

    @abstractmethod
    def synthesize(self, text: str, output_path: Path, scene_index: int = 0) -> SceneAudio:
        """Render ``text`` to an audio file and report per-word timings."""


def approximate_word_timings(
    text: str, duration: float, *, offset: float = 0.0
) -> list[WordTiming]:
    """Distribute ``duration`` across the words of ``text``, weighted by length.

    Used where per-word timing is unavailable. Weighting by character count is
    crude but noticeably better than equal spacing, because long words genuinely
    take longer to say. A small constant is added per word so one-letter words
    still get visible screen time.

    ``offset`` shifts the result, which lets a caller anchor a span to a real
    timestamp — e.g. distributing words inside a sentence whose true start and
    duration came from the synthesizer.
    """
    words = text.split()
    if not words or duration <= 0:
        return []

    weights = [len(word) + 2 for word in words]
    total_weight = sum(weights)

    timings: list[WordTiming] = []
    cursor = offset
    for word, weight in zip(words, weights, strict=True):
        span = duration * (weight / total_weight)
        timings.append(WordTiming(word=word, start=round(cursor, 3), end=round(cursor + span, 3)))
        cursor += span
    return timings
