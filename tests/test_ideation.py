from __future__ import annotations

import pytest

from shorts_agent.exceptions import ProviderError
from shorts_agent.ideation.ideas import IdeaGenerator
from shorts_agent.ideation.script import ScriptWriter, scene_count_for
from shorts_agent.models import Idea, TrendTopic
from tests.conftest import FakeLLM

IDEA_PAYLOAD = {
    "ideas": [
        {
            "title": "The subscription draining your account",
            "hook": "You are paying for this and forgot.",
            "premise": "How to find forgotten subscriptions.",
            "target_emotion": "curiosity",
            "trend_keywords": ["budgeting tips"],
            "virality_reasoning": "Opens a loop the viewer must resolve about their own money.",
        },
        {"title": "", "hook": "no title", "premise": "should be dropped"},
    ]
}


def script_payload(scene_texts, cta="Follow for more."):
    return {
        "scenes": [
            {"text": t, "visual_keyword": f"visual {i}", "on_screen_text": ""}
            for i, t in enumerate(scene_texts)
        ],
        "cta": cta,
    }


def test_idea_generator_drops_ideas_without_a_title(config):
    llm = FakeLLM(json_responses=[IDEA_PAYLOAD])
    ideas = IdeaGenerator(llm, config).generate(
        [TrendTopic(keyword="budgeting tips", source="manual")]
    )

    assert len(ideas) == 1
    assert ideas[0].title == "The subscription draining your account"
    assert ideas[0].niche == config.channel.niche


def test_idea_prompt_includes_trends_and_avoids_recent_topics(config):
    llm = FakeLLM(json_responses=[IDEA_PAYLOAD])
    IdeaGenerator(llm, config).generate(
        [TrendTopic(keyword="side hustle ideas", score=7.5, source="youtube")],
        recent_topics=["how to save money fast"],
        top_performers=[{"title": "A past hit", "views": 90000, "topic": "budgeting"}],
    )

    prompt = llm.prompts[0]
    assert "side hustle ideas" in prompt
    assert "how to save money fast" in prompt  # do-not-repeat list
    assert "A past hit" in prompt  # performance feedback


def test_idea_generator_handles_no_trend_signals(config):
    llm = FakeLLM(json_responses=[IDEA_PAYLOAD])
    IdeaGenerator(llm, config).generate([])

    assert "no live trend signals" in llm.prompts[0]


def test_scene_count_scales_with_duration_and_is_clamped():
    assert scene_count_for(45) == 8
    assert scene_count_for(5) == 3  # floor
    assert scene_count_for(600) == 12  # ceiling


def test_script_writer_builds_scenes_and_cta(config):
    llm = FakeLLM(json_responses=[script_payload(["Line one.", "Line two."])])
    idea = Idea(title="T", hook="Hook line.", premise="P")

    script = ScriptWriter(llm, config).write(idea)

    assert [s.text for s in script.scenes] == ["Line one.", "Line two."]
    assert script.scenes[0].index == 0
    assert script.cta == "Follow for more."


def test_script_writer_skips_blank_scenes(config):
    llm = FakeLLM(json_responses=[script_payload(["Real line.", "   ", ""])])

    script = ScriptWriter(llm, config).write(Idea(title="T", hook="H", premise="P"))

    assert len(script.scenes) == 1


def test_script_writer_falls_back_to_title_for_missing_visual_keyword(config):
    llm = FakeLLM(json_responses=[{"scenes": [{"text": "Line.", "visual_keyword": ""}], "cta": ""}])

    script = ScriptWriter(llm, config).write(Idea(title="My Title", hook="H", premise="P"))

    assert script.scenes[0].visual_keyword == "My Title"
    assert script.cta is None


def test_script_writer_raises_when_no_usable_scenes(config):
    llm = FakeLLM(json_responses=[{"scenes": [], "cta": ""}])

    with pytest.raises(ProviderError, match="no usable scenes"):
        ScriptWriter(llm, config).write(Idea(title="T", hook="H", premise="P"))


def test_overlong_script_triggers_one_tightening_pass(config):
    config.content.max_duration_seconds = 30
    config.content.words_per_second = 2.0

    long_script = script_payload(["word " * 100])  # ~50s of narration
    short_script = script_payload(["Tight line."])
    llm = FakeLLM(json_responses=[long_script, short_script])

    writer = ScriptWriter(llm, config)
    script = writer.write(Idea(title="T", hook="H", premise="P"))

    assert len(llm.prompts) == 2
    assert "too long" in llm.prompts[1]
    assert script.word_count == 2  # the tightened version was kept


def test_tightening_pass_is_discarded_if_it_is_not_shorter(config):
    config.content.max_duration_seconds = 10
    config.content.words_per_second = 2.0

    first = script_payload(["word " * 40])
    worse = script_payload(["word " * 80])
    llm = FakeLLM(json_responses=[first, worse])

    script = ScriptWriter(llm, config).write(Idea(title="T", hook="H", premise="P"))

    assert script.word_count == 40


def test_script_within_budget_makes_only_one_call(config):
    llm = FakeLLM(json_responses=[script_payload(["Short and sweet."])])

    ScriptWriter(llm, config).write(Idea(title="T", hook="H", premise="P"))

    assert len(llm.prompts) == 1
