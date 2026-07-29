"""Curated keywords from a YAML file.

No network, no key, never fails — this is the provider that keeps a run
productive when the live signals are down or rate-limited.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from shorts_agent.models import TrendTopic
from shorts_agent.trends.base import TrendProvider

logger = logging.getLogger(__name__)


class ManualTrendsProvider(TrendProvider):
    name = "manual"

    def __init__(self, keywords_file: str | Path | None):
        self.keywords_file = Path(keywords_file) if keywords_file else None

    def available(self) -> bool:
        return bool(self.keywords_file and self.keywords_file.exists())

    def fetch(self, niche: str, limit: int = 10) -> list[TrendTopic]:
        if not self.available():
            logger.info("Manual trends provider skipped: no keywords file configured")
            return []

        assert self.keywords_file is not None
        try:
            raw = yaml.safe_load(self.keywords_file.read_text()) or {}
        except yaml.YAMLError as exc:
            logger.warning("Could not parse %s: %s", self.keywords_file, exc)
            return []

        topics: list[TrendTopic] = []
        for entry in raw.get("topics", []):
            if isinstance(entry, str):
                keyword, weight, notes = entry, 1.0, ""
            elif isinstance(entry, dict) and entry.get("keyword"):
                keyword = str(entry["keyword"])
                weight = float(entry.get("weight", 1.0))
                notes = str(entry.get("notes", ""))
            else:
                continue

            topics.append(
                TrendTopic(
                    keyword=keyword,
                    score=5.0 * weight,
                    source=self.name,
                    metadata={"notes": notes} if notes else {},
                )
            )

        topics.sort(key=lambda t: t.score, reverse=True)
        return topics[:limit]
