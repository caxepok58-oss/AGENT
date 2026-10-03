"""Visuals from the channel owner's own footage folder.

Put video clips and pictures in ``config/footage`` and ``providers.visuals: local``
uses them as scene backgrounds. Nothing is downloaded and no key is involved, so
it is the most dependable source: it works offline, cannot be rate-limited or
blocked, and the footage is whatever the owner chose and holds the rights to.

A clip is chosen for each scene like this:

1. Relevance first. The scene's keyword is compared with the words in each file's
   path, which is its name and the folders above it, so ``footage/money/piggy.mp4``
   answers "saving money". Two words match when they are equal or one starts with
   the other ("coin" and "coins"). The clips sharing the most words win.
2. Among equally relevant clips, those this video has not used yet.
3. Then a random pick, so that one video does not always open on the same shot.

A keyword that nothing matches still gets a clip rather than a blank card: real
footage that is only loosely on topic reads better than a gradient, and a library
of generic b-roll should not have to be named after every possible topic. An empty
or missing folder falls back to generated cards, with a warning.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shorts_agent.models import VisualAsset
from shorts_agent.text import word_set
from shorts_agent.visuals.base import FILLER_WORDS, VisualProvider
from shorts_agent.visuals.generated import GeneratedVisualProvider

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Below this length a shared prefix proves nothing ("car" would match "cardio").
_MIN_PREFIX_MATCH = 4


@dataclass(frozen=True)
class Clip:
    path: Path
    kind: Literal["video", "image"]
    words: frozenset[str]


def list_footage(directory: Path) -> list[Path]:
    """The usable video and picture files under ``directory``, in a stable order.

    Hidden files (``.DS_Store``, ``._clip.mp4`` resource forks from macOS) and
    empty files are ignored, as is anything in a hidden folder.
    """
    if not directory.is_dir():
        return []

    found: list[Path] = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            continue
        try:
            if path.is_file() and path.stat().st_size > 0:
                found.append(path)
        except OSError:  # a broken symlink, or a file that vanished mid-scan
            continue
    return sorted(found)


def words_match(a: str, b: str) -> bool:
    """Equal words, or one the start of the other (and long enough to mean it)."""
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= _MIN_PREFIX_MATCH and longer.startswith(shorter)


def keyword_words(keyword: str) -> set[str]:
    return word_set(keyword, min_length=3, stopwords=FILLER_WORDS)


class LocalFootageProvider(VisualProvider):
    name = "local"

    def __init__(
        self,
        footage_dir: Path,
        *,
        width: int = 1080,
        height: int = 1920,
        rng: random.Random | None = None,
    ):
        self.footage_dir = footage_dir
        self._rng = rng or random.Random()
        self._fallback = GeneratedVisualProvider(width=width, height=height)
        self._library: list[Clip] | None = None
        self._used: set[Path] = set()
        self._warned_empty = False

    @property
    def clips(self) -> list[Clip]:
        """The library, scanned on first use and then kept for the provider's life."""
        if self._library is None:
            self._library = [
                Clip(
                    path=path,
                    kind="video" if path.suffix.lower() in VIDEO_SUFFIXES else "image",
                    # The folders count as words too: footage/money/clip1.mp4 is about money.
                    words=frozenset(
                        word_set(
                            " ".join(path.relative_to(self.footage_dir).with_suffix("").parts),
                            min_length=3,
                        )
                    ),
                )
                for path in list_footage(self.footage_dir)
            ]
        return self._library

    def fetch(
        self,
        keyword: str,
        output_dir: Path,
        scene_index: int,
        *,
        text: str | None = None,
    ) -> VisualAsset:
        clips = self.clips
        if not clips:
            if not self._warned_empty:
                logger.warning(
                    "No video or picture files in %s; scenes get generated cards instead. "
                    "Put .mp4/.mov/.jpg/.png files there (see docs/SETUP.md).",
                    self.footage_dir,
                )
                self._warned_empty = True
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        wanted = keyword_words(keyword)

        def relevance(clip: Clip) -> int:
            return sum(1 for w in wanted if any(words_match(w, c) for c in clip.words))

        # sorted() evaluates the key once per clip, so each gets one random tiebreaker.
        ranked = sorted(
            clips,
            key=lambda clip: (-relevance(clip), clip.path in self._used, self._rng.random()),
        )
        chosen = ranked[0]
        self._used.add(chosen.path)

        if relevance(chosen):
            logger.info("Scene %d: footage %s matches %r", scene_index, chosen.path.name, keyword)
        else:
            logger.info(
                "Scene %d: nothing in %s matches %r; using %s",
                scene_index,
                self.footage_dir.name,
                keyword,
                chosen.path.name,
            )

        return VisualAsset(
            scene_index=scene_index, path=str(chosen.path), kind=chosen.kind, source=self.name
        )
