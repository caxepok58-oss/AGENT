"""Stock footage from Pexels and Pixabay.

Both APIs are free with an API key. Vertical clips are strongly preferred: a
landscape clip cropped to 9:16 loses most of its subject, so portrait results
are ranked first and the assembler crops whatever it gets.

Every failure path — no key, no results, a network error, a bad download — falls
through to :class:`GeneratedVisualProvider`, because a missing scene visual
cannot be recovered from later in the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import requests

from shorts_agent.models import VisualAsset
from shorts_agent.visuals.base import VisualProvider
from shorts_agent.visuals.generated import GeneratedVisualProvider

logger = logging.getLogger(__name__)

TIMEOUT = 30
DOWNLOAD_TIMEOUT = 90

# (download url, asset kind, file suffix)
SearchResult = tuple[str, Literal["video", "image"], str]


class _StockProvider(VisualProvider):
    """Shared download/fallback plumbing for the keyed stock providers."""

    def __init__(
        self,
        api_key: str | None,
        *,
        width: int = 1080,
        height: int = 1920,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.width = width
        self.height = height
        self._session = session or requests.Session()
        self._fallback = GeneratedVisualProvider(width=width, height=height)

    def fetch(
        self,
        keyword: str,
        output_dir: Path,
        scene_index: int,
        *,
        text: str | None = None,
    ) -> VisualAsset:
        if not self.api_key:
            logger.info("%s provider has no API key; generating a card instead", self.name)
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        try:
            candidate = self._search(keyword)
        except requests.RequestException as exc:
            logger.warning("%s search failed for %r: %s", self.name, keyword, exc)
            candidate = None

        if not candidate:
            logger.info(
                "%s returned no usable result for %r; generating a card", self.name, keyword
            )
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        url, kind, suffix = candidate
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"scene_{scene_index:02d}_{self.name}{suffix}"

        try:
            self._download(url, path)
        except requests.RequestException as exc:
            logger.warning("%s download failed for %r: %s", self.name, keyword, exc)
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        return VisualAsset(scene_index=scene_index, path=str(path), kind=kind, source=self.name)

    def _download(self, url: str, path: Path) -> None:
        with self._session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)

    def _search(self, keyword: str) -> SearchResult | None:
        """Return ``(url, kind, file_suffix)`` for the best match, or None."""
        raise NotImplementedError


class PexelsVisualProvider(_StockProvider):
    name = "pexels"

    def _search(self, keyword: str) -> SearchResult | None:
        response = self._session.get(
            "https://api.pexels.com/videos/search",
            params={
                "query": keyword,
                "orientation": "portrait",
                "per_page": "10",
                "size": "medium",
            },
            headers={"Authorization": self.api_key or ""},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        videos = response.json().get("videos", [])

        best: tuple[int, str] | None = None
        for video in videos:
            for file in video.get("video_files", []):
                link = file.get("link")
                if not link or file.get("file_type") != "video/mp4":
                    continue
                w = file.get("width") or 0
                h = file.get("height") or 0
                if not w or not h:
                    continue
                # Prefer portrait, then resolution close to the target height —
                # oversized files cost download time for no visible gain.
                portrait_bonus = 10_000 if h > w else 0
                score = portrait_bonus - abs(h - self.height)
                if best is None or score > best[0]:
                    best = (score, link)

        if not best:
            return None
        return best[1], "video", ".mp4"


class PixabayVisualProvider(_StockProvider):
    name = "pixabay"

    def _search(self, keyword: str) -> SearchResult | None:
        response = self._session.get(
            "https://pixabay.com/api/videos/",
            params={
                "key": self.api_key or "",
                "q": keyword,
                "per_page": "10",
                "safesearch": "true",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        hits = response.json().get("hits", [])

        best: tuple[int, str] | None = None
        for hit in hits:
            for variant in (hit.get("videos") or {}).values():
                url = variant.get("url")
                w = variant.get("width") or 0
                h = variant.get("height") or 0
                if not url or not w or not h:
                    continue
                portrait_bonus = 10_000 if h > w else 0
                score = portrait_bonus - abs(h - self.height)
                if best is None or score > best[0]:
                    best = (score, url)

        if not best:
            return None
        return best[1], "video", ".mp4"
