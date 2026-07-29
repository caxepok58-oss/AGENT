"""Fetch published video stats to feed the ideation loop.

``videos.list`` costs 1 quota unit per call regardless of how many ids are
batched into it, so stats are always requested in batches of 50 — the API's
per-call maximum — rather than one call per video.

This uses the Data API's public statistics (views, likes, comments), not the
YouTube Analytics API. Retention and impressions live in Analytics, which needs
a separate scope and reports on a delay; view counts are enough to tell the
ideation prompt which topics landed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def fetch_video_stats(service: Any, video_ids: list[str]) -> dict[str, dict[str, int]]:
    """Return ``{video_id: {views, likes, comments}}`` for the given ids."""
    from googleapiclient.errors import HttpError

    stats: dict[str, dict[str, int]] = {}

    for start in range(0, len(video_ids), BATCH_SIZE):
        batch = video_ids[start : start + BATCH_SIZE]
        try:
            response = service.videos().list(part="statistics", id=",".join(batch)).execute()
        except HttpError as exc:
            logger.warning("Could not fetch stats for %d videos: %s", len(batch), exc)
            continue

        for item in response.get("items", []):
            raw = item.get("statistics", {})
            stats[item["id"]] = {
                "views": int(raw.get("viewCount", 0) or 0),
                "likes": int(raw.get("likeCount", 0) or 0),
                "comments": int(raw.get("commentCount", 0) or 0),
            }

    logger.info("Fetched stats for %d of %d videos", len(stats), len(video_ids))
    return stats
