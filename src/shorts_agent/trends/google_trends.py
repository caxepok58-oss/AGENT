"""Google Trends signals via pytrends.

pytrends scrapes an undocumented endpoint, so it breaks and rate-limits without
warning. Every failure here is swallowed and logged: a run should still produce
a video from the remaining providers.
"""

from __future__ import annotations

import logging

from shorts_agent.models import TrendTopic
from shorts_agent.trends.base import TrendProvider

logger = logging.getLogger(__name__)


class GoogleTrendsProvider(TrendProvider):
    name = "google_trends"

    def __init__(self, *, enabled: bool = True, geo: str = "US", timeframe: str = "now 7-d"):
        self.enabled = enabled
        self.geo = geo
        self.timeframe = timeframe

    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            import pytrends  # noqa: F401
        except ImportError:
            logger.info(
                "Google Trends provider skipped: pytrends not installed "
                "(pip install 'shorts-agent[trends]')"
            )
            return False
        return True

    def fetch(self, niche: str, limit: int = 10) -> list[TrendTopic]:
        if not self.available():
            return []

        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=0)
            seed = niche[:100]
            pytrends.build_payload([seed], timeframe=self.timeframe, geo=self.geo)
            related = pytrends.related_queries()
        except Exception as exc:  # noqa: BLE001 - pytrends raises many unrelated types
            logger.warning("Google Trends fetch failed, continuing without it: %s", exc)
            return []

        topics: list[TrendTopic] = []
        for payload in related.values():
            rising = payload.get("rising") if isinstance(payload, dict) else None
            if rising is None or getattr(rising, "empty", True):
                continue
            for row in rising.head(limit).itertuples():
                query = str(getattr(row, "query", "")).strip()
                if not query:
                    continue
                # pytrends reports breakout spikes as the sentinel 5000; clamp so
                # one breakout term can't dominate the aggregate ranking.
                raw_value = float(getattr(row, "value", 0) or 0)
                topics.append(
                    TrendTopic(
                        keyword=query,
                        score=min(raw_value / 100.0, 20.0),
                        source=self.name,
                        region=self.geo,
                        metadata={"rising_value": raw_value},
                    )
                )

        topics.sort(key=lambda t: t.score, reverse=True)
        return topics[:limit]
