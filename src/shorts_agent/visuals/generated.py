"""Zero-dependency visual fallback: rendered gradient cards.

This exists so the pipeline produces a real, watchable video with no visual API
key configured at all — useful for first-run verification and as the safety net
when a stock search returns nothing. Cards are deliberately plain: a deep
gradient darkened toward the edges, since the captions and voiceover carry the
video and busy backgrounds fight them.

Colours are derived from a hash of the scene keyword, so the same topic looks
consistent across runs while successive scenes still differ visibly.
"""

from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

from shorts_agent.exceptions import ProviderError
from shorts_agent.models import VisualAsset
from shorts_agent.visuals.base import VisualProvider

# How much the frame edges are darkened relative to the centre.
VIGNETTE_STRENGTH = 0.45


class GeneratedVisualProvider(VisualProvider):
    name = "generated"

    def __init__(self, width: int = 1080, height: int = 1920):
        self.width = width
        self.height = height

    def fetch(
        self,
        keyword: str,
        output_dir: Path,
        scene_index: int,
        *,
        text: str | None = None,
    ) -> VisualAsset:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise ProviderError(
                "Pillow and numpy are required to generate visuals. "
                "Install with: pip install 'shorts-agent[video]'"
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"scene_{scene_index:02d}_generated.png"

        top, bottom = self._palette(keyword, scene_index)

        # Vertical gradient: interpolate top -> bottom down the frame.
        ramp = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        gradient = (
            np.asarray(top, dtype=np.float32) * (1.0 - ramp)
            + np.asarray(bottom, dtype=np.float32) * ramp
        )
        frame = np.repeat(gradient[:, None, :], self.width, axis=1)

        # Radial brightness falloff: 1.0 at the centre, lowest at the corners,
        # normalised so the corners sit at radius 1.0.
        y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)[None, :]
        radius = np.sqrt(x * x + y * y) / np.sqrt(2.0)
        vignette = np.clip(1.0 - VIGNETTE_STRENGTH * radius * radius, 0.0, 1.0)
        frame *= vignette[..., None]

        Image.fromarray(frame.clip(0, 255).astype(np.uint8), mode="RGB").save(path, "PNG")
        return VisualAsset(scene_index=scene_index, path=str(path), kind="image", source=self.name)

    def _palette(self, keyword: str, scene_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        digest = hashlib.sha256(keyword.encode()).digest()
        # Rotate the hue per scene so consecutive cards are visibly different
        # without leaving the topic's colour family.
        hue = ((digest[0] / 255.0) + scene_index * 0.11) % 1.0

        top = colorsys.hsv_to_rgb(hue, 0.55, 0.46)
        bottom = colorsys.hsv_to_rgb((hue + 0.06) % 1.0, 0.72, 0.18)
        return (
            tuple(round(c * 255) for c in top),
            tuple(round(c * 255) for c in bottom),
        )
