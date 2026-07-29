"""Resumable upload to YouTube, with scheduling and AI disclosure.

Three things here are easy to get wrong and expensive to discover in production:

* **Scheduling.** ``publishAt`` is only honoured when ``privacyStatus`` is
  ``private``. Setting a publish time on a public video silently does nothing.
* **AI disclosure.** ``status.containsSyntheticMedia`` is how the Data API
  declares altered or synthetic content. A synthetic-voice video must set it —
  see docs/POLICY.md.
* **Quota.** An upload costs roughly 100 units against a dedicated daily
  allowance. The pipeline's own ``max_uploads_per_day`` limit exists to keep a
  runaway loop from burning it, and from tripping YouTube's spam policies.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shorts_agent.exceptions import UploadError
from shorts_agent.models import PublishResult, VideoMetadata

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4 * 1024 * 1024
MAX_ATTEMPTS = 5
RETRIABLE_STATUS = {500, 502, 503, 504}


class YouTubeUploader:
    def __init__(self, service: Any):
        self.service = service

    def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        *,
        thumbnail_path: Path | None = None,
    ) -> PublishResult:
        if not video_path.exists():
            raise UploadError(f"Video file not found: {video_path}")

        body = self._build_body(metadata)
        media = self._media_upload(video_path)

        logger.info(
            "Uploading %s (%.1f MB) as %s",
            video_path.name,
            video_path.stat().st_size / 1e6,
            body["status"]["privacyStatus"],
        )

        request = self.service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = self._execute_resumable(request)

        video_id = response.get("id")
        if not video_id:
            raise UploadError(f"Upload succeeded but returned no video id: {response}")

        if thumbnail_path and thumbnail_path.exists():
            self._set_thumbnail(video_id, thumbnail_path)

        if metadata.playlist_id:
            self._add_to_playlist(video_id, metadata.playlist_id)

        return PublishResult(
            youtube_video_id=video_id,
            url=f"https://www.youtube.com/shorts/{video_id}",
            privacy_status=response.get("status", {}).get("privacyStatus", "unknown"),
            publish_at=metadata.publish_at,
        )

    def _build_body(self, metadata: VideoMetadata) -> dict:
        status: dict[str, Any] = {
            "privacyStatus": metadata.privacy_status,
            "selfDeclaredMadeForKids": metadata.made_for_kids,
            "containsSyntheticMedia": metadata.contains_synthetic_media,
        }

        if metadata.publish_at:
            if metadata.privacy_status != "private":
                # Fail loudly: silently ignoring the schedule would publish the
                # video immediately, which is the opposite of what was asked.
                raise UploadError(
                    "publish_at requires privacy_status='private'; YouTube ignores a "
                    f"scheduled time on a {metadata.privacy_status} video."
                )
            status["publishAt"] = _rfc3339(metadata.publish_at)

        return {
            "snippet": {
                "title": metadata.title,
                "description": metadata.description,
                "tags": metadata.tags,
                "categoryId": metadata.category_id,
            },
            "status": status,
        }

    def _media_upload(self, video_path: Path) -> Any:
        from googleapiclient.http import MediaFileUpload

        return MediaFileUpload(
            str(video_path), chunksize=CHUNK_SIZE, resumable=True, mimetype="video/mp4"
        )

    def _execute_resumable(self, request: Any) -> dict:
        """Drive a resumable upload, retrying transient failures with backoff."""
        from googleapiclient.errors import HttpError

        attempt = 0
        response = None

        while response is None:
            try:
                _, response = request.next_chunk()
            except HttpError as exc:
                if exc.resp.status not in RETRIABLE_STATUS:
                    raise UploadError(f"Upload failed: {exc}") from exc
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    raise UploadError(
                        f"Upload failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                delay = min(2**attempt + random.random(), 60)
                logger.warning(
                    "Transient upload error (HTTP %s); retrying in %.1fs",
                    exc.resp.status,
                    delay,
                )
                time.sleep(delay)
            except OSError as exc:
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    raise UploadError(
                        f"Upload failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                delay = min(2**attempt + random.random(), 60)
                logger.warning("Network error during upload; retrying in %.1fs", delay)
                time.sleep(delay)

        return response

    def _set_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        from googleapiclient.errors import HttpError

        try:
            self.service.thumbnails().set(
                videoId=video_id, media_body=str(thumbnail_path)
            ).execute()
            logger.info("Set custom thumbnail for %s", video_id)
        except HttpError as exc:
            # Custom thumbnails need a verified channel, and Shorts largely
            # ignore them anyway — never fail an upload over this.
            logger.warning("Could not set thumbnail (upload itself succeeded): %s", exc)

    def _add_to_playlist(self, video_id: str, playlist_id: str) -> None:
        from googleapiclient.errors import HttpError

        try:
            self.service.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            ).execute()
            logger.info("Added %s to playlist %s", video_id, playlist_id)
        except HttpError as exc:
            logger.warning("Could not add to playlist (upload itself succeeded): %s", exc)


def _rfc3339(value: datetime) -> str:
    """Format a datetime as RFC 3339 UTC, which is what publishAt expects."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
