from __future__ import annotations

from shorts_agent.redact import redact


def test_redact_replaces_the_secret_with_a_placeholder():
    text = "Max retries exceeded with url: /videos?key=abc123&chart=mostPopular"

    assert "abc123" not in redact(text, "abc123")


def test_redact_handles_multiple_secrets():
    text = "key=one and also token=two"

    result = redact(text, "one", "two")

    assert "one" not in result
    assert "two" not in result


def test_redact_is_a_no_op_without_a_secret():
    assert redact("nothing sensitive here", None) == "nothing sensitive here"


def test_redact_ignores_an_empty_secret():
    """An empty string would otherwise match everywhere and mangle the text."""
    assert redact("hello world", "") == "hello world"


def test_redact_leaves_unrelated_text_untouched():
    text = "connection refused"

    assert redact(text, "some-other-key") == text
