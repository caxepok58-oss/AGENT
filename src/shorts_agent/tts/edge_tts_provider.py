"""Microsoft Edge TTS — the default voice provider.

Chosen as the default because it needs no API key and reports boundary events,
which anchor captions to the real audio instead of a guess.

Which events arrive is not under our control: the service emits ``WordBoundary``
for some voices and only ``SentenceBoundary`` for others, and this has changed
over time for the same voice. All three cases are handled, in descending order
of precision:

1. ``WordBoundary`` — exact per-word start and duration.
2. ``SentenceBoundary`` — real sentence start and duration, with words
   distributed inside that span by length. Captions stay anchored per sentence.
3. Neither — the rendered audio's measured duration, distributed across all
   words.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from shorts_agent.exceptions import ProviderError
from shorts_agent.models import SceneAudio, WordTiming
from shorts_agent.tts.base import TTSProvider, approximate_word_timings

logger = logging.getLogger(__name__)

# edge-tts reports offsets and durations in 100-nanosecond ticks.
TICKS_PER_SECOND = 10_000_000


class EdgeTTSProvider(TTSProvider):
    name = "edge"

    def __init__(self, voice: str = "en-US-AndrewNeural", *, rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, output_path: Path, scene_index: int = 0) -> SceneAudio:
        try:
            import edge_tts
        except ImportError as exc:
            raise ProviderError(
                "The edge-tts package is required for the edge TTS provider. "
                "Install with: pip install 'shorts-agent[tts]'"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio, word_events, sentence_events = asyncio.run(self._stream(edge_tts, text, output_path))

        if not audio:
            raise ProviderError(f"edge-tts returned no audio for scene {scene_index}")

        timings, source = self._resolve_timings(text, output_path, word_events, sentence_events)
        if not timings:
            raise ProviderError(f"Could not determine caption timings for scene {scene_index}")

        logger.debug("Scene %d timing source: %s", scene_index, source)
        duration = max(t.end for t in timings)
        return SceneAudio(
            scene_index=scene_index,
            audio_path=str(output_path),
            word_timings=timings,
            duration_seconds=round(duration, 3),
        )

    def _resolve_timings(
        self,
        text: str,
        output_path: Path,
        word_events: list[WordTiming],
        sentence_events: list[WordTiming],
    ) -> tuple[list[WordTiming], str]:
        if word_events:
            return word_events, "WordBoundary"

        if sentence_events:
            timings: list[WordTiming] = []
            for sentence in sentence_events:
                timings.extend(
                    approximate_word_timings(
                        sentence.word,
                        sentence.end - sentence.start,
                        offset=sentence.start,
                    )
                )
            if timings:
                return timings, "SentenceBoundary"

        measured = self._probe_duration(output_path) or _estimate_duration(text)
        return approximate_word_timings(text, measured), "measured duration"

    async def _stream(
        self, edge_tts, text: str, output_path: Path
    ) -> tuple[bytes, list[WordTiming], list[WordTiming]]:
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        chunks: list[bytes] = []
        words: list[WordTiming] = []
        sentences: list[WordTiming] = []

        try:
            async for chunk in communicate.stream():
                kind = chunk["type"]
                if kind == "audio":
                    chunks.append(chunk["data"])
                elif kind in ("WordBoundary", "SentenceBoundary"):
                    start = chunk["offset"] / TICKS_PER_SECOND
                    boundary = WordTiming(
                        word=chunk["text"],
                        start=round(start, 3),
                        end=round(start + chunk["duration"] / TICKS_PER_SECOND, 3),
                    )
                    (words if kind == "WordBoundary" else sentences).append(boundary)
        except Exception as exc:  # noqa: BLE001 - edge-tts raises varied network errors
            raise ProviderError(f"edge-tts synthesis failed: {exc}") from exc

        audio = b"".join(chunks)
        output_path.write_bytes(audio)
        return audio, words, sentences

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        try:
            from moviepy import AudioFileClip
        except ImportError:
            return None
        try:
            with AudioFileClip(str(path)) as clip:
                return float(clip.duration)
        except Exception:  # noqa: BLE001 - probing is best-effort
            return None


def _estimate_duration(text: str, words_per_second: float = 2.5) -> float:
    return max(len(text.split()) / words_per_second, 0.5)
