from __future__ import annotations

import requests

from shorts_agent.models import TrendTopic
from shorts_agent.trends.aggregator import TrendAggregator, _normalize
from shorts_agent.trends.base import TrendProvider
from shorts_agent.trends.manual import ManualTrendsProvider
from shorts_agent.trends.youtube_trends import YouTubeTrendsProvider


class StubProvider(TrendProvider):
    def __init__(self, name, topics=None, error=None):
        self.name = name
        self._topics = topics or []
        self._error = error

    def fetch(self, niche, limit=10):
        if self._error:
            raise self._error
        return self._topics


def topic(keyword, score, source):
    return TrendTopic(keyword=keyword, score=score, source=source)


def test_normalize_collapses_case_and_punctuation():
    assert _normalize("Save MONEY, fast!") == _normalize("save money fast")


def test_aggregator_merges_duplicates_and_rewards_corroboration():
    a = StubProvider("a", [topic("save money fast", 5.0, "youtube")])
    b = StubProvider("b", [topic("Save Money Fast!", 5.0, "manual")])

    merged = TrendAggregator([a, b]).collect("finance", limit=5)

    assert len(merged) == 1
    # 10.0 summed, then x1.5 for appearing in two independent sources.
    assert merged[0].score == 15.0
    assert merged[0].metadata["sources"] == ["manual", "youtube"]


def test_single_source_topic_is_not_boosted():
    provider = StubProvider("a", [topic("budgeting", 4.0, "manual")])

    merged = TrendAggregator([provider]).collect("finance", limit=5)

    assert merged[0].score == 4.0


def test_aggregator_survives_a_failing_provider():
    good = StubProvider("good", [topic("budgeting", 3.0, "manual")])
    bad = StubProvider("bad", error=requests.RequestException("upstream down"))

    merged = TrendAggregator([bad, good]).collect("finance", limit=5)

    assert [t.keyword for t in merged] == ["budgeting"]


def test_aggregator_sorts_by_score_and_applies_limit():
    provider = StubProvider(
        "a",
        [topic("low", 1.0, "manual"), topic("high", 9.0, "manual"), topic("mid", 5.0, "manual")],
    )

    merged = TrendAggregator([provider]).collect("finance", limit=2)

    assert [t.keyword for t in merged] == ["high", "mid"]


def test_manual_provider_reads_weights(tmp_path):
    path = tmp_path / "manual.yaml"
    path.write_text(
        "topics:\n  - keyword: side hustle ideas\n    weight: 2.0\n  - budgeting tips\n"
    )

    topics = ManualTrendsProvider(path).fetch("finance")

    assert topics[0].keyword == "side hustle ideas"
    assert topics[0].score == 10.0
    assert topics[1].score == 5.0


def test_manual_provider_is_unavailable_without_a_file(tmp_path):
    provider = ManualTrendsProvider(tmp_path / "missing.yaml")

    assert provider.available() is False
    assert provider.fetch("finance") == []


def test_manual_provider_tolerates_malformed_yaml(tmp_path):
    path = tmp_path / "manual.yaml"
    path.write_text("topics: [unclosed")

    assert ManualTrendsProvider(path).fetch("finance") == []


def test_youtube_provider_skipped_without_api_key():
    provider = YouTubeTrendsProvider(api_key=None)

    assert provider.available() is False
    assert provider.fetch("finance") == []


def test_youtube_provider_scores_niche_relevance_above_view_count(monkeypatch):
    provider = YouTubeTrendsProvider(api_key="key")
    payload = {
        "items": [
            {
                "snippet": {"title": "Cat compilation", "tags": ["cats"]},
                "statistics": {"viewCount": "50000000"},
            },
            {
                "snippet": {"title": "Budgeting tips for students", "tags": ["budgeting"]},
                "statistics": {"viewCount": "1000"},
            },
        ]
    }
    monkeypatch.setattr(provider, "_get", lambda endpoint, params: payload)

    topics = provider.fetch("budgeting tips", limit=5)

    # The on-niche video wins despite 50,000x fewer views.
    assert topics[0].keyword == "Budgeting tips for students"


def test_youtube_provider_returns_empty_on_network_error(monkeypatch):
    provider = YouTubeTrendsProvider(api_key="key")

    def boom(endpoint, params):
        raise requests.RequestException("429 rate limited")

    monkeypatch.setattr(provider, "_get", boom)

    assert provider.fetch("finance") == []
