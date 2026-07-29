"""SQLite persistence.

Three jobs:

1. Enforce the daily upload rate limit (``count_uploads_since``).
2. Avoid re-publishing the same topic (``recent_topics``), which is both a
   quality issue and a YouTube spam-policy risk.
3. Store performance stats fetched back from the Data API so the ideation
   prompt can learn which topics actually worked (``top_performers``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    niche        TEXT,
    idea_title   TEXT,
    topic        TEXT,
    script_json  TEXT,
    metadata_json TEXT,
    video_path   TEXT,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS uploads (
    youtube_video_id TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    uploaded_at      TEXT NOT NULL,
    privacy_status   TEXT,
    publish_at       TEXT,
    title            TEXT,
    topic            TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

CREATE TABLE IF NOT EXISTS video_stats (
    youtube_video_id TEXT NOT NULL,
    fetched_at       TEXT NOT NULL,
    views            INTEGER,
    likes            INTEGER,
    comments         INTEGER,
    PRIMARY KEY (youtube_video_id, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at);
CREATE INDEX IF NOT EXISTS idx_uploads_uploaded_at ON uploads (uploaded_at);
"""


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- runs ---------------------------------------------------------

    def start_run(self, run_id: str, niche: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, created_at, status, niche) VALUES (?, ?, ?, ?)",
                (run_id, _now(), "started", niche),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "idea_title",
            "topic",
            "script_json",
            "metadata_json",
            "video_path",
            "error",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE runs SET {assignments} WHERE run_id = ?",
                (*updates.values(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- uploads ------------------------------------------------------

    def record_upload(
        self,
        youtube_video_id: str,
        run_id: str,
        privacy_status: str,
        title: str,
        topic: str,
        publish_at: datetime | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO uploads
                   (youtube_video_id, run_id, uploaded_at, privacy_status, publish_at, title, topic)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    youtube_video_id,
                    run_id,
                    _now(),
                    privacy_status,
                    publish_at.isoformat() if publish_at else None,
                    title,
                    topic,
                ),
            )

    def count_uploads_since(self, since: datetime) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM uploads WHERE uploaded_at >= ?",
                (since.isoformat(timespec="seconds"),),
            ).fetchone()
        return int(row["n"])

    def recent_topics(self, days: int) -> list[str]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT topic FROM runs
                   WHERE topic IS NOT NULL AND topic != '' AND created_at >= ?""",
                (cutoff,),
            ).fetchall()
        return [r["topic"] for r in rows]

    def latest_publish_at(self) -> datetime | None:
        """Latest scheduled publish time, so successive runs don't collide on one slot."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(publish_at) AS latest FROM uploads WHERE publish_at IS NOT NULL"
            ).fetchone()
        if not row or not row["latest"]:
            return None
        return datetime.fromisoformat(row["latest"])

    # --- stats --------------------------------------------------------

    def record_stats(self, youtube_video_id: str, views: int, likes: int, comments: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO video_stats
                   (youtube_video_id, fetched_at, views, likes, comments) VALUES (?, ?, ?, ?, ?)""",
                (youtube_video_id, _now(), views, likes, comments),
            )

    def top_performers(self, limit: int = 5) -> list[dict[str, Any]]:
        """Best-performing published topics, used to bias future ideation."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT u.title, u.topic, MAX(s.views) AS views, MAX(s.likes) AS likes
                   FROM uploads u
                   JOIN video_stats s ON s.youtube_video_id = u.youtube_video_id
                   GROUP BY u.youtube_video_id
                   ORDER BY views DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def uploads_needing_stats(self, limit: int = 50) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT youtube_video_id FROM uploads ORDER BY uploaded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r["youtube_video_id"] for r in rows]


def dump_json(model: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return str(model.model_dump_json())
    return json.dumps(model, default=str)
