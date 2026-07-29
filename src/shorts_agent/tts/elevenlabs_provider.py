"""ElevenLabs TTS — higher voice quality, at the cost of an API key and credits.

Word timing here is estimated rather than reported. ElevenLabs does expose
character-level alignment on some endpoints, but the shape of that response has
moved between SDK versions, so this provider measures the rendered audio's real
duration and distributes it across words. Captions stay in sync at the scene
level; individual word highlights can drift slightly mid-scene.

Use the edge provider when caption precision matters more than voice quality.
"""

from __future__ import annotations

import logging
from pathlib import Path

from shorts_agent.exceptions import ProviderError
from shorts_agent.models import SceneAudio
from shorts_agent.tts.base import TTSProvider, approximate_word_timings

logger = logging.getLogger(__name__)

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # "Rachel", available on all accounts
DEFAULT_MODEL = "eleven_multilingual_v2"


class ElevenLabsProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        if not api_key:
            raise ProviderError("ELEVENLABS_API_KEY is required for the elevenlabs TTS provider")
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise ProviderError(
                "The elevenlabs package is required for this provider. "
                "Install with: pip install 'shorts-agent[tts-elevenlabs]'"
            ) from exc

        self._client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.model = model

    def synthesize(self, text: str, output_path: Path, scene_index: int = 0) -> SceneAudio:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            stream = self._client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id=self.model,
                output_format="mp3_44100_128",
            )
            audio = b"".join(stream)
        except Exception as exc:  # noqa: BLE001 - SDK raises varied error types
            raise ProviderError(f"ElevenLabs synthesis failed: {exc}") from exc

        if not audio:
            raise ProviderError(f"ElevenLabs returned no audio for scene {scene_index}")
        output_path.write_bytes(audio)

        duration = self._probe_duration(output_path) or _estimate_duration(text)
        return SceneAudio(
            scene_index=scene_index,
            audio_path=str(output_path),
            word_timings=approximate_word_timings(text, duration),
            duration_seconds=round(duration, 3),
        )

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
