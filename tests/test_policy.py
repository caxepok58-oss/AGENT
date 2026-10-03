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


def test_blocklist_match_folds_eszett_like_casefold(config, tmp_path):
    """.lower() leaves German ß alone, so a blocklisted "straße" would miss an
    all-caps "STRASSE" in the script; casefold() maps both to "strasse"."""
    path = tmp_path / "blocklist.yaml"
    path.write_text(yaml.safe_dump({"categories": {"test": ["straße"]}}))
    config.policy.blocklist_file = str(path)

    result = PolicyGuard(config).check_idea(idea(premise="Welcome to STRASSE avenue."))

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


# --- a self-contradicting reviewer is asked again, then fails closed -----------


def concern(severity, category="financial", detail="A problem."):
    return {"category": category, "detail": detail, "severity": severity}


def moderated(config, *responses):
    config.policy.use_llm_moderation = True
    llm = FakeLLM(json_responses=list(responses))
    result = PolicyGuard(config, llm=llm).check_script(script_of("A tip."), idea())
    return result, llm


def test_a_consistent_answer_costs_exactly_one_call(config):
    result, llm = moderated(config, APPROVED)

    assert result.passed is True
    assert len(llm.prompts) == 1


def test_rejecting_without_any_concern_is_asked_again_and_can_then_pass(config):
    """Seen from a real 7B model: approved=false with an empty list of concerns."""
    contradiction = {"approved": False, "concerns": []}

    result, llm = moderated(config, contradiction, APPROVED)

    assert result.passed is True
    assert len(llm.prompts) == 2
    assert "inconsistent" in llm.prompts[1]


def test_rejecting_over_low_severity_concerns_is_asked_again(config):
    low_only = {"approved": False, "concerns": [concern("low"), concern("medium")]}

    result, llm = moderated(config, low_only, APPROVED)

    assert result.passed is True
    assert len(llm.prompts) == 2


def test_approving_while_listing_a_high_severity_concern_is_not_taken_at_face_value(config):
    """The mirror image: a high-severity concern must never be waved through just
    because the reviewer also ticked "approved"."""
    result, llm = moderated(config, {"approved": True, "concerns": [concern("high")]}, APPROVED)

    assert result.passed is True  # it reconsidered and answered consistently
    assert len(llm.prompts) == 2


def test_a_second_contradiction_fails_closed_and_says_why(config):
    low_only = {"approved": False, "concerns": [concern("low")]}

    result, llm = moderated(config, low_only, low_only)

    assert result.passed is False
    assert "without a stated high-severity concern" in result.reasons[0]
    assert len(llm.prompts) == 2  # asked again once, not in a loop


def test_approving_with_a_high_concern_twice_blocks_and_names_the_concern(config):
    contradiction = {
        "approved": True,
        "concerns": [concern("high", "medical", "Promises a cure.")],
    }

    result, _ = moderated(config, contradiction, contradiction)

    assert result.passed is False
    assert "Promises a cure." in result.reasons[0]


def test_a_reviewer_that_changes_its_mind_to_block_still_blocks(config):
    contradiction = {"approved": False, "concerns": []}
    block = {"approved": False, "concerns": [concern("high", "medical", "Promises a cure.")]}

    result, _ = moderated(config, contradiction, block)

    assert result.passed is False
    assert "Promises a cure." in result.reasons[0]


def test_a_provider_error_is_not_retried(config):
    config.policy.use_llm_moderation = True
    llm = FakeLLM(error=ProviderError("down"))

    result = PolicyGuard(config, llm=llm).check_script(script_of("A tip."), idea())

    assert result.passed is False
    assert len(llm.prompts) == 1
