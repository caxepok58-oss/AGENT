"""Trend signals from the YouTube Data API.

Quota shapes this module's design. ``videos.list`` costs 1 unit per call, while
``search.list`` costs 100 — so the default path reads the mostPopular chart
(one cheap call) and scores those videos for niche relevance locally. Keyword
search is available but opt-in, because a handful of searches can consume an
entire day's default 10,000-unit allowance.

Read-only access needs only an API key, not OAuth — see SETUP.md.
"""

from __future__ import annotations

import logging

import requests

from shorts_agent.models import TrendTopic
from shorts_agent.text import word_set
from shorts_agent.trends.base import TrendProvider

logger = logging.getLogger(__name__)

API_ROOT = "https://www.googleapis.com/youtube/v3"
TIMEOUT = 20

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "at",
    "by",
    "from",
    "how",
    "why",
    "what",
    "you",
    "your",
    "my",
    "i",
    "we",
    "he",
    "she",
    "they",
    "new",
    "best",
    "top",
    "vs",
    "shorts",
}


def _keywords(text: str) -> set[str]:
    """Extract comparable words from a title, in any script.

    The stopword list is English-only, which is harmless: for other languages it
    simply removes nothing, and relevance is still driven by the overlap between
    the niche's words and the video's.
    """
    return word_set(text, min_length=3, stopwords=_STOPWORDS)


class YouTubeTrendsProvider(TrendProvider):
    name = "youtube"

    def __init__(
        self,
        api_key: str | None,
        *,
        region_code: str = "US",
        use_search: bool = False,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.region_code = region_code
        self.use_search = use_search
        self._session = session or requests.Session()

    def available(self) -> bool:
        return bool(self.api_key)

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "key": self.api_key}
        response = self._session.get(f"{API_ROOT}/{endpoint}", params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()

    def fetch(self, niche: str, limit: int = 10) -> list[TrendTopic]:
        if not self.available():
            logger.info("YouTube trends provider skipped: no YOUTUBE_API_KEY set")
            return []

        try:
            topics = self._fetch_most_popular(niche, limit)
            if self.use_search:
                topics.extend(self._fetch_search(niche, limit))
        except requests.RequestException as exc:
            logger.warning("YouTube trends fetch failed, continuing without it: %s", exc)
            return []

        return topics[:limit]

    def _fetch_most_popular(self, niche: str, limit: int) -> list[TrendTopic]:
        data = self._get(
            "videos",
            {
                "part": "snippet,statistics",
                "chart": "mostPopular",
                "regionCode": self.region_code,
                "maxResults": 50,
            },
        )
        niche_words = _keywords(niche)
        topics: list[TrendTopic] = []

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            tags = snippet.get("tags") or []
            haystack = _keywords(f"{title} {' '.join(tags)}")
            overlap = len(niche_words & haystack)
            views = int(item.get("statistics", {}).get("viewCount", 0) or 0)

            # Relevance dominates; view count only breaks ties between equally
            # relevant videos, so a viral-but-unrelated video can't crowd out
            # an on-niche one.
            score = overlap * 10 + min(views / 1_000_000, 5.0)
            topics.append(
                TrendTopic(
                    keyword=title,
                    score=score,
                    source=f"{self.name}:most_popular",
                    region=self.region_code,
                    sample_titles=[title],
                    metadata={"views": views, "niche_overlap": overlap, "tags": tags[:10]},
                )
            )

        topics.sort(key=lambda t: t.score, reverse=True)
        return topics[: limit * 2]

    def _fetch_search(self, niche: str, limit: int) -> list[TrendTopic]:
        """Keyword search — 100 quota units per call, so used only when enabled."""
        data = self._get(
            "search",
            {
                "part": "snippet",
                "q": niche,
                "type": "video",
                "videoDuration": "short",
                "order": "viewCount",
                "regionCode": self.region_code,
                "maxResults": min(limit, 25),
            },
        )
        return [
            TrendTopic(
                keyword=item["snippet"]["title"],
                score=5.0,
                source=f"{self.name}:search",
                region=self.region_code,
                sample_titles=[item["snippet"]["title"]],
                metadata={"channel": item["snippet"].get("channelTitle", "")},
            )
            for item in data.get("items", [])
            if item.get("snippet")
        ]
