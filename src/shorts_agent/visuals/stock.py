"""Stock footage from Pexels and Pixabay.

Both APIs are free with an API key. A vertical clip is taken over a landscape one,
because a landscape clip cropped to 9:16 loses most of its subject; within either
kind the search engine's own order, which is its relevance order, decides.

Which file to download: the smallest rendition that is sharp enough, not the
biggest on offer. A 4K file is many times the download of a 1080p one and slower to
decode, and the extra detail only shows if the picture is displayed larger than the
frame. A landscape clip is cropped to its middle 9:16, so its height is what matters,
and it needs to be as tall as the frame is wide (1080); a portrait clip needs to be
as tall as the frame (1920). When nothing is that big, the largest available is used.

A search that finds nothing is retried with progressively shorter queries, since
keywords written by an LLM are often more specific than a stock library can match.

Every failure path (no key, no results, a network error, a bad download) falls
through to a fallback provider, the owner's own footage if there is any and a
generated card if not, because a missing scene visual cannot be recovered from later
in the pipeline.

Pexels' API terms ask for the creator to be credited and for a link to Pexels, so
each asset carries a credit line that ends up in the video description.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, NamedTuple

import requests

from shorts_agent.models import VisualAsset
from shorts_agent.redact import redact
from shorts_agent.visuals.base import FILLER_WORDS, VisualProvider
from shorts_agent.visuals.generated import GeneratedVisualProvider

logger = logging.getLogger(__name__)

TIMEOUT = 30
DOWNLOAD_TIMEOUT = 90

# (download url, asset kind, file suffix, credit line)
SearchResult = tuple[str, Literal["video", "image"], str, str]

MAX_QUERIES = 3


def search_queries(keyword: str) -> list[str]:
    """The keyword itself, then up to two progressively shorter versions of it.

    Stopping at two words keeps the footage relevant: a one-word query like
    "person" would match anything.
    """
    words = [w for w in keyword.split() if w.casefold() not in FILLER_WORDS]
    queries = [keyword.strip()]
    for size in (3, 2):
        if len(words) > size:
            queries.append(" ".join(words[:size]))
    return list(dict.fromkeys(q for q in queries if q))[:MAX_QUERIES]


def credit_line(site: str, author: str | None, page: str | None, home: str) -> str:
    """ "Video by <author> on <site>: <page>", always linking to the site itself."""
    who = f" by {author}" if author else ""
    return f"Video{who} on {site}: {page or home}"


class Rendition(NamedTuple):
    url: str
    width: int
    height: int

    @property
    def portrait(self) -> bool:
        return self.height > self.width


def best_rendition(
    renditions: Iterable[Rendition], *, frame_width: int, frame_height: int
) -> Rendition | None:
    """The smallest rendition that is sharp enough for the frame, else the largest."""

    def preference(rendition: Rendition) -> tuple[bool, int]:
        needed = frame_height if rendition.portrait else frame_width
        enough = rendition.height >= needed
        return enough, -rendition.height if enough else rendition.height

    return max(renditions, key=preference, default=None)


def first_vertical_else_first(
    candidates: list[tuple[Rendition, str]],
) -> tuple[Rendition, str] | None:
    """Prefer a portrait clip; otherwise the engine's top result."""
    for candidate in candidates:
        if candidate[0].portrait:
            return candidate
    return candidates[0] if candidates else None


class _StockProvider(VisualProvider):
    """Shared download/fallback plumbing for the keyed stock providers."""

    def __init__(
        self,
        api_key: str | None,
        *,
        width: int = 1080,
        height: int = 1920,
        session: requests.Session | None = None,
        fallback: VisualProvider | None = None,
    ):
        self.api_key = api_key
        self.width = width
        self.height = height
        self._session = session or requests.Session()
        self._fallback = fallback or GeneratedVisualProvider(width=width, height=height)

    def fetch(
        self,
        keyword: str,
        output_dir: Path,
        scene_index: int,
        *,
        text: str | None = None,
    ) -> VisualAsset:
        if not self.api_key:
            logger.info(
                "%s provider has no API key; using %s instead", self.name, self._fallback.name
            )
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        candidate = None
        for query in search_queries(keyword):
            try:
                candidate = self._search(query)
            except requests.RequestException as exc:
                logger.warning(
                    "%s search failed for %r: %s", self.name, query, redact(str(exc), self.api_key)
                )
                break  # a shorter query will not fix a network failure
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                # The service answered, but not in the shape its documentation promises.
                logger.warning(
                    "%s sent an unexpected response for %r: %s: %s",
                    self.name,
                    query,
                    type(exc).__name__,
                    redact(str(exc), self.api_key),
                )
                break
            if candidate:
                if query != keyword:
                    logger.info(
                        "%s: nothing for %r, using results for %r", self.name, keyword, query
                    )
                break

        if not candidate:
            logger.info(
                "%s returned no usable result for %r; using %s instead",
                self.name,
                keyword,
                self._fallback.name,
            )
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        url, kind, suffix, credit = candidate
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"scene_{scene_index:02d}_{self.name}{suffix}"

        try:
            self._download(url, path)
        except requests.RequestException as exc:
            logger.warning(
                "%s download failed for %r: %s", self.name, keyword, redact(str(exc), self.api_key)
            )
            return self._fallback.fetch(keyword, output_dir, scene_index, text=text)

        return VisualAsset(
            scene_index=scene_index, path=str(path), kind=kind, source=self.name, credit=credit
        )

    def _download(self, url: str, path: Path) -> None:
        with self._session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)

    def _search(self, keyword: str) -> SearchResult | None:
        """Return ``(url, kind, file_suffix, credit)`` for the best match, or None."""
        raise NotImplementedError

    def _pick(self, candidates: list[tuple[Rendition, str]]) -> SearchResult | None:
        chosen = first_vertical_else_first(candidates)
        if chosen is None:
            return None
        rendition, credit = chosen
        return rendition.url, "video", ".mp4", credit

    def _best(self, renditions: Iterable[Rendition]) -> Rendition | None:
        return best_rendition(renditions, frame_width=self.width, frame_height=self.height)


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

        candidates: list[tuple[Rendition, str]] = []
        for video in videos:
            best = self._best(
                Rendition(file["link"], file["width"], file["height"])
                for file in video.get("video_files", [])
                if file.get("link")
                and file.get("file_type") == "video/mp4"
                and file.get("width")
                and file.get("height")
            )
            if best:
                credit = credit_line(
                    "Pexels",
                    (video.get("user") or {}).get("name"),
                    video.get("url"),
                    "https://www.pexels.com",
                )
                candidates.append((best, credit))

        return self._pick(candidates)


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
                # Real footage only: the default also returns animations, which
                # are abstract loops rather than b-roll.
                "video_type": "film",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        hits = response.json().get("hits", [])

        candidates: list[tuple[Rendition, str]] = []
        for hit in hits:
            best = self._best(
                Rendition(variant["url"], variant["width"], variant["height"])
                for variant in (hit.get("videos") or {}).values()
                if variant.get("url") and variant.get("width") and variant.get("height")
            )
            if best:
                credit = credit_line(
                    "Pixabay", hit.get("user"), hit.get("pageURL"), "https://pixabay.com"
                )
                candidates.append((best, credit))

        return self._pick(candidates)
