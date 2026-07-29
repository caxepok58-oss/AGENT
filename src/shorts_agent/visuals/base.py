from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from shorts_agent.models import VisualAsset


class VisualProvider(ABC):
    """A source of per-scene footage or imagery.

    Providers must always return an asset. A scene without a visual has no
    fallback later in the pipeline — the video simply cannot be assembled — so
    implementations that depend on a remote API are expected to delegate to the
    generated-card provider when a search comes back empty.
    """

    name: str = "unknown"

    @abstractmethod
    def fetch(
        self,
        keyword: str,
        output_dir: Path,
        scene_index: int,
        *,
        text: str | None = None,
    ) -> VisualAsset:
        """Return a visual for one scene, downloading or generating as needed."""
