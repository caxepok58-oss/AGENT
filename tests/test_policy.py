from __future__ import annotations

import yaml

from shorts_agent.exceptions import ProviderError
from shorts_agent.models import Idea, Scene, Script
from shorts_agent.policy.guardrails import PolicyGuard
from tests.conftest import FakeLLM

BLOCKLIST = {
    "categories": {
        "medical_financial_claims": ["guaranteed returns", "risk-free investment"],
        "dangerous_acts": ["how to make explosives"],
    }
}

APPROVED = {"approved": True, "concerns": []}


def blocklist_file(tmp_path):
    path = tmp_path / "blocklist.yaml"
    path.write_text(yaml.safe_dump(BLOCKLIST))
    return path


def idea(title="A safe money tip", hook="Here is a tip.", premise="A premise."):
    return Idea(title=title, hook=hook, premise=premise)


def script_of(*texts, title="A safe money tip"):
    return Script(
        idea_id="x",
        title=title,
        scenes=[Scene(index=i, text=t, visual_keyword="v") for i, t in enumerate(texts)],
    )


def test_clean_idea_passes(config, tmp_path):
    config.policy.blocklist_file = str(blocklist_file(tmp_path))
    guard = PolicyGuard(config)

    assert guard.check_idea(idea()).passed is True


def test_blocklisted_phrase_blocks_an_idea(config, tmp_path):
    config.policy.blocklist_file = str(blocklist_file(tmp_path))
    guard = PolicyGuard(config)

    result = guard.check_idea(idea(premise="This offers guaranteed returns every month."))

    assert result.passed is False
    assert "medical_financial_claims" in result.reasons[0]


def test_blocklist_match_is_case_insensitive(config, tmp_path):
    config.policy.blocklist_file = str(blocklist_file(tmp_path))

    result = PolicyGuard(config).check_idea(idea(title="GUARANTEED RETURNS in 2026"))

    assert result.passed is False


def test_missing_blocklist_file_disables_keyword_check(config, tmp_path):
    config.policy.blocklist_file = str(tmp_path / "absent.yaml")

    result = PolicyGuard(config).check_idea(idea(premise="guaranteed returns"))

    assert result.passed is True


def test_near_duplicate_topic_is_blocked(config):
    guard = PolicyGuard(config, recent_topics=["How to save money fast in 2026"])

    result = guard.check_idea(idea(title="How to save money fast in 2026!"))

    assert result.passed is False
    assert "duplicate" in result.reasons[0]


def test_distinct_topic_is_not_treated_as_duplicate(config):
    guard = PolicyGuard(config, recent_topics=["How to save money fast"])

    assert guard.check_idea(idea(title="Three index funds explained")).passed is True


def test_duplicate_check_can_be_disabled(config):
    config.policy.disallow_duplicate_topics_days = 0
    guard = PolicyGuard(config, recent_topics=["How to save money fast"])

    assert guard.check_idea(idea(title="How to save money fast")).passed is True


def test_llm_moderation_blocks_only_on_high_severity(config):
    config.policy.use_llm_moderation = True
    llm = FakeLLM(
        json_responses=[
            {
                "approved": False,
                "concerns": [
                    {
                        "category": "financial",
                        "detail": "Promises a guaranteed outcome.",
                        "severity": "high",
                    },
                    {"category": "tone", "detail": "Slightly hyped.", "severity": "low"},
                ],
            }
        ]
    )

    result = PolicyGuard(config, llm=llm).check_script(script_of("Invest now."), idea())

    assert result.passed is False
    assert len(result.reasons) == 1
    assert "financial" in result.reasons[0]


def test_llm_moderation_allows_low_severity_concerns(config):
    config.policy.use_llm_moderation = True
    llm = FakeLLM(
        json_responses=[
            {
                "approved": True,
                "concerns": [{"category": "tone", "detail": "A bit punchy.", "severity": "medium"}],
            }
        ]
    )

    result = PolicyGuard(config, llm=llm).check_script(script_of("A tip."), idea())

    assert result.passed is True


def test_moderation_outage_fails_closed(config):
    """An unavailable moderator must block, not silently approve."""
    config.policy.use_llm_moderation = True
    llm = FakeLLM(error=ProviderError("upstream unavailable"))

    result = PolicyGuard(config, llm=llm).check_script(script_of("A tip."), idea())

    assert result.passed is False
    assert "moderation unavailable" in result.reasons[0]


def test_moderation_is_skipped_when_disabled(config):
    config.policy.use_llm_moderation = False
    llm = FakeLLM(json_responses=[])  # would raise if consulted

    assert PolicyGuard(config, llm=None).check_script(script_of("A tip."), idea()).passed is True
    assert llm.prompts == []


def test_script_body_is_checked_against_the_blocklist(config, tmp_path):
    config.policy.blocklist_file = str(blocklist_file(tmp_path))
    config.policy.use_llm_moderation = False

    result = PolicyGuard(config).check_script(
        script_of("Here is how to make explosives at home."), idea()
    )

    assert result.passed is False
    assert "dangerous_acts" in result.reasons[0]
