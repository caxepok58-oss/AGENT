from __future__ import annotations

import pytest

from shorts_agent.exceptions import ProviderError
from shorts_agent.llm.base import LLMClient, LLMResponse, extract_json, object_schema


class ScriptedTextLLM(LLMClient):
    """Returns canned text, so the base-class JSON fallback can be exercised."""

    def __init__(self, replies: list[str]):
        super().__init__("scripted")
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, max_tokens=None):
        self.prompts.append(prompt)
        return LLMResponse(text=self.replies.pop(0))


def test_extract_json_parses_bare_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_ignores_surrounding_prose():
    assert extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_extract_json_handles_arrays():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError, match="no JSON object found"):
        extract_json("there is no json here")


def test_complete_json_retries_with_the_parse_error():
    llm = ScriptedTextLLM(["not json at all", '{"ok": true}'])

    result = llm.complete_json("Do the thing", schema=object_schema({"ok": {"type": "boolean"}}))

    assert result == {"ok": True}
    assert len(llm.prompts) == 2
    # The retry tells the model what went wrong.
    assert "could not be parsed" in llm.prompts[1]


def test_complete_json_gives_up_after_the_attempt_budget():
    llm = ScriptedTextLLM(["nope", "still nope", "nope again"])

    with pytest.raises(ProviderError, match="no valid JSON after 3 attempts"):
        llm.complete_json("Do it", schema=object_schema({"ok": {"type": "boolean"}}), attempts=3)


def test_object_schema_is_structured_output_compatible():
    schema = object_schema({"a": {"type": "string"}, "b": {"type": "integer"}})

    # Structured outputs reject objects that allow extra keys or omit `required`.
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["a", "b"]


def test_object_schema_accepts_an_explicit_required_subset():
    schema = object_schema({"a": {"type": "string"}, "b": {"type": "integer"}}, required=["a"])

    assert schema["required"] == ["a"]
