from __future__ import annotations

from datetime import datetime, timezone

import pytest

from shorts_agent.exceptions import UploadError
from shorts_agent.models import VideoMetadata
from shorts_agent.youtube.uploader import YouTubeUploader, _rfc3339


def metadata(**overrides) -> VideoMetadata:
    base = {
        "title": "A title",
        "description": "A description",
        "tags": ["budgeting"],
        "category_id": "27",
    }
    return VideoMetadata(**{**base, **overrides})


@pytest.fixture
def uploader() -> YouTubeUploader:
    return YouTubeUploader(service=object())


def test_rfc3339_marks_naive_datetimes_as_utc():
    assert _rfc3339(datetime(2026, 5, 1, 12, 30)) == "2026-05-01T12:30:00Z"


def test_rfc3339_converts_aware_datetimes_to_utc():
    aware = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    assert _rfc3339(aware).endswith("Z")


def test_body_declares_synthetic_media(uploader):
    body = uploader._build_body(metadata(contains_synthetic_media=True))

    assert body["status"]["containsSyntheticMedia"] is True


def test_body_can_declare_non_synthetic_media(uploader):
    body = uploader._build_body(metadata(contains_synthetic_media=False))

    assert body["status"]["containsSyntheticMedia"] is False


def test_body_carries_snippet_fields(uploader):
    body = uploader._build_body(metadata())

    assert body["snippet"]["title"] == "A title"
    assert body["snippet"]["tags"] == ["budgeting"]
    assert body["snippet"]["categoryId"] == "27"


def test_body_declares_made_for_kids(uploader):
    body = uploader._build_body(metadata(made_for_kids=True))

    assert body["status"]["selfDeclaredMadeForKids"] is True


def test_scheduling_on_a_private_video_sets_publish_at(uploader):
    when = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)

    body = uploader._build_body(metadata(privacy_status="private", publish_at=when))

    assert body["status"]["publishAt"] == "2026-06-01T09:00:00Z"


def test_scheduling_a_public_video_is_rejected(uploader):
    """YouTube ignores publishAt unless the video is private — fail loudly instead."""
    when = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)

    with pytest.raises(UploadError, match="requires privacy_status='private'"):
        uploader._build_body(metadata(privacy_status="public", publish_at=when))


def test_scheduling_an_unlisted_video_is_rejected(uploader):
    when = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)

    with pytest.raises(UploadError, match="requires privacy_status='private'"):
        uploader._build_body(metadata(privacy_status="unlisted", publish_at=when))


def test_unscheduled_body_has_no_publish_at(uploader):
    body = uploader._build_body(metadata())

    assert "publishAt" not in body["status"]


def test_upload_rejects_a_missing_file(uploader, tmp_path):
    with pytest.raises(UploadError, match="Video file not found"):
        uploader.upload(tmp_path / "absent.mp4", metadata())
