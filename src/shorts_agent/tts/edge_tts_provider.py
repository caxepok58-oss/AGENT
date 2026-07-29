"""Microsoft Edge TTS — the default voice provider.

Chosen as the default because it needs no API key and, critically, reports
``WordBoundary`` events. Those give real per-word timings, which is what makes
word-by-word highlighted captions land on the right syllable instead of drifting
out of sync partway through a scene.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from shorts_agent.exceptions import ProviderError
from shorts_agent.models import SceneAudio, WordTiming
from shorts_agent.tts.base import TTSProvider, approximate_word_timings

logger = logging.getLogger(__name__)

# edge-tts reports offsets in 100-nanosecond ticks.
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
        audio, timings = asyncio.run(self._stream(edge_tts, text, output_path))

        if not audio:
            raise ProviderError(f"edge-tts returned no audio for scene {scene_index}")

        duration = timings[-1].end if timings else 0.0
        if not timings:
            # No WordBoundary events (happens with some voices/locales): fall
            # back to estimated timing so captions still work.
            duration = self._probe_duration(output_path) or _estimate_duration(text)
            timings = approximate_word_timings(text, duration)
            logger.info("No WordBoundary events for scene %d; using estimated timings", scene_index)

        return SceneAudio(
            scene_index=scene_index,
            audio_path=str(output_path),
            word_timings=timings,
            duration_seconds=round(duration, 3),
        )

    async def _stream(self, edge_tts, text: str, output_path: Path) -> tuple[bytes, list[WordTiming]]:
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        chunks: list[bytes] = []
        timings: list[WordTiming] = []

        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / TICKS_PER_SECOND
                    timings.append(
                        WordTiming(
                            word=chunk["text"],
                            start=round(start, 3),
                            end=round(start + chunk["duration"] / TICKS_PER_SECOND, 3),
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - edge-tts raises varied network errors
            raise ProviderError(f"edge-tts synthesis failed: {exc}") from exc

        audio = b"".join(chunks)
        output_path.write_bytes(audio)
        return audio, timings

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
