"""Combine signals from several trend providers into one ranked list."""

from __future__ import annotations

import logging
from collections import defaultdict

from shorts_agent.config import AppConfig, get_settings
from shorts_agent.models import TrendTopic
from shorts_agent.text import normalize_key
from shorts_agent.trends.base import TrendProvider
from shorts_agent.trends.google_trends import GoogleTrendsProvider
from shorts_agent.trends.manual import ManualTrendsProvider
from shorts_agent.trends.youtube_trends import YouTubeTrendsProvider

logger = logging.getLogger(__name__)


def _normalize(keyword: str) -> str:
    """Collapse a keyword to a dedup key ignoring case, punctuation and spacing."""
    return normalize_key(keyword)


class TrendAggregator:
    def __init__(self, providers: list[TrendProvider]):
        self.providers = providers

    def collect(self, niche: str, limit: int = 15) -> list[TrendTopic]:
        per_provider = max(limit, 10)
        buckets: dict[str, list[TrendTopic]] = defaultdict(list)

        for provider in self.providers:
            try:
                found = provider.fetch(niche, limit=per_provider)
            except Exception as exc:  # noqa: BLE001 - one bad provider must not fail the run
                logger.warning("Trend provider %s failed: %s", provider.name, exc)
                continue
            logger.info("Trend provider %s returned %d topics", provider.name, len(found))
            for topic in found:
                key = _normalize(topic.keyword)
                if key:
                    buckets[key].append(topic)

        merged = [self._merge(group) for group in buckets.values()]
        merged.sort(key=lambda t: t.score, reverse=True)
        return merged[:limit]

    @staticmethod
    def _merge(group: list[TrendTopic]) -> TrendTopic:
        """Fold duplicate topics into one, rewarding cross-source agreement.

        A topic surfaced by two independent providers is a stronger signal than
        one provider's high score, so corroboration multiplies rather than
        simply summing.
        """
        best = max(group, key=lambda t: t.score)
        sources = sorted({t.source.split(":")[0] for t in group})
        total = sum(t.score for t in group)
        corroboration = 1.0 + 0.5 * (len(sources) - 1)

        titles: list[str] = []
        metadata: dict = {}
        for topic in group:
            titles.extend(topic.sample_titles)
            metadata.update(topic.metadata)

        return TrendTopic(
            keyword=best.keyword,
            score=round(total * corroboration, 3),
            source="+".join(sources),
            region=best.region,
            sample_titles=list(dict.fromkeys(titles))[:5],
            metadata={**metadata, "merged_from": len(group), "sources": sources},
        )


def build_trend_providers(config: AppConfig) -> list[TrendProvider]:
    settings = get_settings()
    providers: list[TrendProvider] = []

    for name in config.trends.providers:
        if name == "youtube":
            providers.append(
                YouTubeTrendsProvider(
                    api_key=settings.youtube_api_key,
                    region_code=config.trends.youtube_region_code,
                )
            )
        elif name == "google_trends":
            providers.append(
                GoogleTrendsProvider(
                    enabled=settings.google_trends_enabled,
                    geo=config.trends.youtube_region_code,
                )
            )
        elif name == "manual":
            path = (
                config.resolve_path(config.trends.manual_keywords_file)
                if config.trends.manual_keywords_file
                else None
            )
            providers.append(ManualTrendsProvider(path))

    return providers
