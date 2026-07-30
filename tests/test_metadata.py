from __future__ import annotations

import pytest

from shorts_agent.metadata.seo import (
    DESCRIPTION_LIMIT,
    TAGS_TOTAL_LIMIT,
    TITLE_LIMIT,
    MetadataGenerator,
    _build_description,
    _clean_hashtags,
    _clean_tags,
    validate_metadata,
)
from shorts_agent.models import Idea, Scene, Script, VideoMetadata
from tests.conftest import FakeLLM

PAYLOAD = {
    "title": "The subscription quietly draining your account",
    "description": "Most people forget at least one subscription.\nHere is how to find it.",
    "tags": ["budgeting", "save money", "Budgeting", "#subscriptions"],
    "hashtags": ["#money", "shorts"],
}


def script():
    return Script(
        idea_id="x",
        title="T",
        scenes=[Scene(index=0, text="Check your statement.", visual_keyword="v")],
    )


def metadata(**overrides):
    base = {"title": "T", "description": "D"}
    return VideoMetadata(**{**base, **overrides})


def test_generate_builds_complete_metadata(config):
    llm = FakeLLM(json_responses=[PAYLOAD])

    result = MetadataGenerator(llm, config).generate(
        Idea(title="T", hook="H", premise="P"), script()
    )

    assert result.title == PAYLOAD["title"]
    assert result.category_id == config.channel.category_id
    assert result.privacy_status == "private"
    assert result.contains_synthetic_media is True


def test_ai_disclosure_flag_follows_config(config):
    config.content.ai_disclosure = False
    llm = FakeLLM(json_responses=[PAYLOAD])

    result = MetadataGenerator(llm, config).generate(
        Idea(title="T", hook="H", premise="P"), script()
    )

    assert result.contains_synthetic_media is False


def test_tags_are_lowercased_deduplicated_and_stripped_of_hashes():
    assert _clean_tags(["Budgeting", "budgeting", "#saving", "  "]) == ["budgeting", "saving"]


def test_tags_are_trimmed_to_the_api_total_limit():
    tags = _clean_tags([f"tag-number-{i:03d}" for i in range(200)])

    total = sum(len(t) for t in tags) + max(len(tags) - 1, 0)
    assert total <= TAGS_TOTAL_LIMIT
    assert len(tags) < 200


def test_shorts_hashtag_is_always_present():
    assert "#Shorts" in _clean_hashtags(["#money"], [])


def test_hashtags_are_normalized_and_capped():
    result = _clean_hashtags(["money", "#money", "#saving", "#budget", "#extra", "#more"], [])

    assert result[0].startswith("#")
    assert len(result) <= 4
    # "money" and "#money" are the same tag.
    assert sum(1 for h in result if h.lower() == "#money") == 1


def test_channel_default_hashtags_are_merged_in():
    result = _clean_hashtags(["#money"], ["#Shorts", "#finance"])

    assert "#finance" in result


def test_non_latin_hashtags_keep_their_combining_marks():
    """A \\w-based strip would mangle "#कैसे" into "#कस" by dropping vowel marks."""
    result = _clean_hashtags(["कैसे", "#सीखें"], [])

    assert "#कैसे" in result
    assert "#सीखें" in result


def test_description_appends_hashtags_once():
    body = "A useful tip.\n\n#old #tags"

    description = _build_description(body, ["#Shorts", "#money"])

    assert description.count("#Shorts") == 1
    assert "#old" not in description
    assert description.endswith("#Shorts #money")


def test_description_without_hashtags_is_left_alone():
    assert _build_description("Just a body.", []) == "Just a body."


def test_overlong_title_is_truncated():
    result = validate_metadata(metadata(title="x" * 200))

    assert len(result.title) <= TITLE_LIMIT
    assert result.title.endswith("…")


def test_title_within_limit_is_untouched():
    title = "A perfectly reasonable title"

    assert validate_metadata(metadata(title=title)).title == title


def test_empty_title_is_rejected():
    with pytest.raises(ValueError, match="empty title"):
        validate_metadata(metadata(title="   "))


def test_overlong_description_is_truncated():
    result = validate_metadata(metadata(description="y" * (DESCRIPTION_LIMIT + 500)))

    assert len(result.description) == DESCRIPTION_LIMIT


def test_angle_brackets_are_stripped_from_title(config):
    llm = FakeLLM(json_responses=[{**PAYLOAD, "title": "Watch <script>alert(1)</script> this"}])

    result = MetadataGenerator(llm, config).generate(
        Idea(title="T", hook="H", premise="P"), script()
    )

    assert "<" not in result.title and ">" not in result.title


def test_metadata_prompt_includes_the_narration(config):
    llm = FakeLLM(json_responses=[PAYLOAD])

    MetadataGenerator(llm, config).generate(Idea(title="T", hook="H", premise="P"), script())

    assert "Check your statement." in llm.prompts[0]
