from __future__ import annotations

from datetime import datetime, timedelta

from shorts_agent.storage import Storage


def test_run_lifecycle_is_recorded(tmp_path):
    storage = Storage(tmp_path / "db.sqlite3")
    storage.start_run("run1", "finance")
    storage.update_run("run1", status="rendered", idea_title="A title", topic="A title")

    record = storage.get_run("run1")
    assert record is not None
    assert record["status"] == "rendered"
    assert record["idea_title"] == "A title"


def test_update_run_ignores_unknown_columns(tmp_path):
    """Unknown keys must not be interpolated into SQL."""
    storage = Storage(tmp_path / "db.sqlite3")
    storage.start_run("run1", "finance")

    storage.update_run("run1", status="ready", bogus="DROP TABLE runs")

    assert storage.get_run("run1")["status"] == "ready"
    assert storage.recent_runs() != []


def test_count_uploads_since_only_counts_recent(tmp_path):
    storage = Storage(tmp_path / "db.sqlite3")
    storage.start_run("run1", "finance")
    storage.record_upload("vid1", "run1", "private", "T", "topic")

    assert storage.count_uploads_since(datetime.utcnow() - timedelta(days=1)) == 1
    assert storage.count_uploads_since(datetime.utcnow() + timedelta(minutes=1)) == 0


def test_recent_topics_excludes_empty_and_deduplicates(tmp_path):
    storage = Storage(tmp_path / "db.sqlite3")
    for i, topic in enumerate(["budgeting", "budgeting", "", None]):
        storage.start_run(f"run{i}", "finance")
        storage.update_run(f"run{i}", topic=topic)

    assert storage.recent_topics(days=30) == ["budgeting"]


def test_top_performers_ranks_by_views(tmp_path):
    storage = Storage(tmp_path / "db.sqlite3")
    for i, (views, title) in enumerate([(500, "low"), (9000, "high")]):
        storage.start_run(f"run{i}", "finance")
        storage.record_upload(f"vid{i}", f"run{i}", "public", title, "topic")
        storage.record_stats(f"vid{i}", views, views // 10, 3)

    performers = storage.top_performers(limit=2)
    assert [p["title"] for p in performers] == ["high", "low"]


def test_latest_publish_at_returns_max(tmp_path):
    storage = Storage(tmp_path / "db.sqlite3")
    storage.start_run("run1", "finance")
    earlier = datetime(2026, 1, 1, 10, 0)
    later = datetime(2026, 1, 2, 10, 0)
    storage.record_upload("v1", "run1", "private", "A", "t", publish_at=earlier)
    storage.record_upload("v2", "run1", "private", "B", "t", publish_at=later)

    assert storage.latest_publish_at() == later


def test_latest_publish_at_is_none_without_scheduled_uploads(tmp_path):
    storage = Storage(tmp_path / "db.sqlite3")
    storage.start_run("run1", "finance")
    storage.record_upload("v1", "run1", "public", "A", "t")

    assert storage.latest_publish_at() is None
