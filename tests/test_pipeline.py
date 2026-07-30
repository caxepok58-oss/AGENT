from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from shorts_agent.exceptions import PolicyViolation, RateLimitExceeded, ShortsAgentError
from shorts_agent.models import Idea, Scene, Script, TrendTopic
from shorts_agent.pipeline import Pipeline
from shorts_agent.storage import Storage
from tests.conftest import FakeLLM

IDEAS = {
    "ideas": [
        {
            "title": "The forgotten subscription",
            "hook": "You are paying for this right now.",
            "premise": "Find forgotten subscriptions.",
            "target_emotion": "curiosity",
            "trend_keywords": ["budgeting tips"],
            "virality_reasoning": "Opens a loop about the viewer's own money.",
        }
    ]
}

SCRIPT = {
    "scenes": [
        {
            "text": "You are paying for this right now.",
            "visual_keyword": "bank app",
            "on_screen_text": "",
        },
        {
            "text": "Open your statement and scan for small charges.",
            "visual_keyword": "phone",
            "on_screen_text": "",
        },
    ],
    "cta": "Follow for more.",
}

MODERATION_OK = {"approved": True, "concerns": []}

METADATA = {
    "title": "The subscription draining your account",
    "description": "How to find forgotten subscriptions.",
    "tags": ["budgeting", "subscriptions"],
    "hashtags": ["#Shorts", "#money"],
}


def make_pipeline(config, llm) -> Pipeline:
    storage = Storage(config.resolve_path("data/test.db"))
    return Pipeline(config, llm=llm, storage=storage)


def test_select_idea_returns_the_first_clean_idea(config):
    pipeline = make_pipeline(config, FakeLLM())
    good = Idea(title="A clean tip", hook="H", premise="P")

    selected, skipped = pipeline.select_idea([good])

    assert selected is good
    assert skipped == []


def test_select_idea_skips_blocked_ideas_and_reports_them(config):
    config.policy.use_llm_moderation = False
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("old", config.channel.niche)
    pipeline.storage.update_run("old", topic="How to save money fast")

    blocked = Idea(title="How to save money fast", hook="H", premise="P")
    clean = Idea(title="Three index funds explained", hook="H", premise="P")

    selected, skipped = pipeline.select_idea([blocked, clean])

    assert selected is clean
    assert skipped[0]["title"] == "How to save money fast"


def test_select_idea_raises_when_everything_is_blocked(config):
    config.policy.use_llm_moderation = False
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("old", config.channel.niche)
    pipeline.storage.update_run("old", topic="How to save money fast")

    with pytest.raises(PolicyViolation, match="all 1 generated ideas were rejected"):
        pipeline.select_idea([Idea(title="How to save money fast", hook="H", premise="P")])


def test_write_script_blocks_on_a_failed_moderation(config):
    llm = FakeLLM(
        json_responses=[
            SCRIPT,
            {
                "approved": False,
                "concerns": [
                    {"category": "financial", "detail": "Guarantees returns.", "severity": "high"}
                ],
            },
        ]
    )
    pipeline = make_pipeline(config, llm)

    with pytest.raises(PolicyViolation, match="financial"):
        pipeline.write_script(Idea(title="T", hook="H", premise="P"))


def test_write_script_returns_an_approved_script(config):
    llm = FakeLLM(json_responses=[SCRIPT, MODERATION_OK])
    pipeline = make_pipeline(config, llm)

    script = pipeline.write_script(Idea(title="T", hook="H", premise="P"))

    assert len(script.scenes) == 2


def test_upload_quota_blocks_once_the_daily_cap_is_reached(config):
    config.publishing.max_uploads_per_day = 1
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("r", config.channel.niche)
    pipeline.storage.record_upload("vid1", "r", "private", "T", "topic")

    with pytest.raises(RateLimitExceeded, match="Daily upload limit reached"):
        pipeline._check_upload_quota()


def test_upload_quota_allows_uploads_below_the_cap(config):
    config.publishing.max_uploads_per_day = 2
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("r", config.channel.niche)
    pipeline.storage.record_upload("vid1", "r", "private", "T", "topic")

    pipeline._check_upload_quota()  # must not raise


def test_zero_cap_disables_publishing_entirely(config):
    config.publishing.max_uploads_per_day = 0
    pipeline = make_pipeline(config, FakeLLM())

    with pytest.raises(RateLimitExceeded, match="uploading is disabled"):
        pipeline._check_upload_quota()


def test_next_slot_respects_the_configured_delay(config):
    config.publishing.default_publish_delay_hours = 6
    pipeline = make_pipeline(config, FakeLLM())

    slot = pipeline._next_slot()

    expected = datetime.now(timezone.utc) + timedelta(hours=6)
    assert abs((slot - expected).total_seconds()) < 60


def test_next_slot_spaces_out_from_the_last_scheduled_upload(config):
    config.publishing.default_publish_delay_hours = 4
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("r", config.channel.niche)

    far_future = datetime.now(timezone.utc) + timedelta(days=1)
    pipeline.storage.record_upload("vid1", "r", "private", "T", "topic", publish_at=far_future)

    slot = pipeline._next_slot()

    # Queued behind the existing slot rather than colliding with it.
    assert slot >= far_future + timedelta(hours=4) - timedelta(seconds=1)


def test_auto_publish_config_enables_uploading(config, monkeypatch):
    """A user who sets auto_publish expects uploads without passing --publish."""
    config.publishing.auto_publish = True
    config.publishing.max_uploads_per_day = 1
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("r", config.channel.niche)
    pipeline.storage.record_upload("vid1", "r", "private", "T", "topic")

    # The cap check only runs when publishing is on, so this proves the config
    # flag reached the publish path.
    with pytest.raises(RateLimitExceeded):
        pipeline.run()


def test_auto_publish_defaults_off(config, monkeypatch):
    """Uploading must stay opt-in: a default config never publishes."""
    config.publishing.max_uploads_per_day = 1
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("r", config.channel.niche)
    pipeline.storage.record_upload("vid1", "r", "private", "T", "topic")

    published = []
    monkeypatch.setattr(pipeline, "publish", lambda *a, **k: published.append(1))
    monkeypatch.setattr(pipeline, "research", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "ideate", lambda *a, **k: [])

    # The run fails on having no ideas — but crucially it got past the upload cap
    # check, which would have raised first had publishing been enabled.
    with pytest.raises(ShortsAgentError, match="No usable ideas"):
        pipeline.run()

    assert published == []


def test_run_records_failure_status(config, monkeypatch):
    pipeline = make_pipeline(config, FakeLLM())
    monkeypatch.setattr(
        pipeline, "research", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run()

    failed = [r for r in pipeline.storage.recent_runs() if r["status"] == "failed"]
    assert failed and failed[0]["error"] == "boom"


def test_run_checks_the_upload_cap_before_doing_any_work(config, monkeypatch):
    """The cap must be enforced before spending LLM calls and a render."""
    config.publishing.max_uploads_per_day = 1
    pipeline = make_pipeline(config, FakeLLM())
    pipeline.storage.start_run("r", config.channel.niche)
    pipeline.storage.record_upload("vid1", "r", "private", "T", "topic")

    called = []
    monkeypatch.setattr(pipeline, "research", lambda *a, **k: called.append("research") or [])

    with pytest.raises(RateLimitExceeded):
        pipeline.run(publish=True)

    assert called == []


def test_run_without_publish_produces_a_reviewable_video(config, monkeypatch):
    llm = FakeLLM(json_responses=[IDEAS, SCRIPT, MODERATION_OK, METADATA])
    pipeline = make_pipeline(config, llm)

    monkeypatch.setattr(
        pipeline,
        "research",
        lambda *a, **k: [TrendTopic(keyword="budgeting tips", source="manual")],
    )

    fake_video = config.resolve_path("output/fake.mp4")
    fake_video.parent.mkdir(parents=True, exist_ok=True)
    fake_video.write_bytes(b"not really a video")
    monkeypatch.setattr(pipeline, "produce", lambda script, run_dir: (fake_video, 30.0))

    result = pipeline.run(publish=False)

    assert result.idea is not None
    assert result.metadata is not None
    assert result.publish is None  # nothing was uploaded
    assert result.video_path == str(fake_video)

    record = pipeline.storage.get_run(result.run_id)
    assert record["status"] == "ready"


def test_produce_offsets_scene_timings_across_the_whole_video(config, monkeypatch):
    """Each scene's timings are audio-relative; the caption track is not."""
    from shorts_agent.models import SceneAudio, VisualAsset, WordTiming

    captured: dict = {}

    class StubTTS:
        def synthesize(self, text, output_path, scene_index=0):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"audio")
            return SceneAudio(
                scene_index=scene_index,
                audio_path=str(output_path),
                word_timings=[WordTiming(word="w", start=0.0, end=1.0)],
                duration_seconds=2.0,
            )

    class StubVisuals:
        def fetch(self, keyword, output_dir, scene_index, *, text=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{scene_index}.png"
            path.write_bytes(b"img")
            return VisualAsset(scene_index=scene_index, path=str(path), kind="image", source="stub")

    monkeypatch.setattr(
        "shorts_agent.pipeline.orchestrator.build_tts_provider", lambda c: StubTTS()
    )
    monkeypatch.setattr(
        "shorts_agent.pipeline.orchestrator.build_visual_provider", lambda c: StubVisuals()
    )

    def fake_write_ass(timings, path, cfg, **kwargs):
        captured["timings"] = timings
        captured["overlays"] = kwargs.get("overlays")
        return path

    monkeypatch.setattr("shorts_agent.pipeline.orchestrator.write_ass", fake_write_ass)

    class StubAssembler:
        def __init__(self, config):
            pass

        def assemble(self, visuals, audios, output_path, *, captions_path=None, work_dir=None):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"video")
            return output_path

    monkeypatch.setattr("shorts_agent.pipeline.orchestrator.VideoAssembler", StubAssembler)

    pipeline = make_pipeline(config, FakeLLM())
    script = Script(
        idea_id="x",
        title="T",
        scenes=[
            Scene(index=0, text="one", visual_keyword="a"),
            Scene(index=1, text="two", visual_keyword="b", on_screen_text="Emphasis"),
        ],
    )

    _, total = pipeline.produce(script, config.resolve_path("output/run1"))

    # A scene's on_screen_text must reach the caption track, spanning that
    # scene's own slice of the timeline.
    overlays = captured["overlays"]
    assert [o.text for o in overlays] == ["Emphasis"]
    assert (overlays[0].start, overlays[0].end) == (2.0, 4.0)

    # Scene 1's word starts at 2.0s, not 0.0s.
    assert [t.start for t in captured["timings"]] == [0.0, 2.0]
    assert total == 4.0
