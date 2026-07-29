from __future__ import annotations

from abc import ABC, abstractmethod

from shorts_agent.models import TrendTopic


class TrendProvider(ABC):
    """A source of trend signals.

    Providers must degrade rather than explode: a missing API key or a
    rate-limited upstream should yield an empty list (and log), not abort the
    run, because the aggregator combines whatever providers happen to work.
    """

    name: str = "unknown"

    @abstractmethod
    def fetch(self, niche: str, limit: int = 10) -> list[TrendTopic]:
        """Return trend topics relevant to ``niche``, best-effort."""

    def available(self) -> bool:
        return True
