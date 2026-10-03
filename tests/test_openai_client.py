"""OpenAIClient against a real HTTP server on localhost.

The point of the local-model feature is talking to servers that are only *mostly*
OpenAI-compatible, so these tests run the real ``openai`` SDK over a real socket
instead of mocking the SDK: base_url handling, the placeholder key, and every way
a server can mishandle ``response_format`` are exercised end to end.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from shorts_agent.exceptions import ProviderError
from shorts_agent.llm.openai_client import OpenAIClient
from tests.conftest import FakeServer, completion

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@pytest.fixture
def make_client(server: FakeServer) -> Callable[..., OpenAIClient]:
    def build(**kwargs: Any) -> OpenAIClient:
        kwargs.setdefault("api_key", "not-needed")
        client = OpenAIClient("test-model", base_url=server.url, **kwargs)
        # No SDK retries: a failure should surface immediately, not after backoff.
        client._client = client._client.with_options(max_retries=0)
        return client

    return build


def test_complete_talks_to_the_configured_endpoint(server, make_client):
    server.reply_with("hello there")

    response = make_client().complete("Say hi", system="Be brief")

    assert response.text == "hello there"
    assert response.usage == {"input_tokens": 7, "output_tokens": 3}
    request = server.requests[0]
    assert request.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer not-needed"
    assert request.body["model"] == "test-model"
    assert [m["role"] for m in request.body["messages"]] == ["system", "user"]


def test_complete_json_requests_a_strict_schema_and_parses_the_reply(server, make_client):
    server.reply_with('{"ok": true}')

    result = make_client().complete_json("Do it", schema=SCHEMA)

    assert result == {"ok": True}
    assert len(server.requests) == 1
    response_format = server.requests[0].body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_fenced_json_from_a_server_that_ignores_response_format_still_parses(server, make_client):
    """Small local models often wrap JSON in a markdown fence despite the schema."""
    server.reply_with('```json\n{"ok": true}\n```')

    result = make_client().complete_json("Do it", schema=SCHEMA)

    assert result == {"ok": True}
    assert len(server.requests) == 1  # parsed leniently, no wasted retry


def test_json_buried_in_chatty_prose_still_parses(server, make_client):
    server.reply_with('Sure! Here is the result:\n{"ok": true}\nLet me know if that helps.')

    assert make_client().complete_json("Do it", schema=SCHEMA) == {"ok": True}
    assert len(server.requests) == 1


def test_unparseable_reply_falls_back_to_reprompting(server, make_client):
    server.reply_with("I cannot comply", '{"ok": true}')

    result = make_client().complete_json("Do it", schema=SCHEMA)

    assert result == {"ok": True}
    assert len(server.requests) == 2
    retry = server.requests[1].body
    assert "response_format" not in retry
    assert "ONLY valid JSON" in retry["messages"][-1]["content"]


def test_server_rejecting_response_format_falls_back_to_reprompting(server, make_client):
    rejection = {"error": {"message": "response_format is not supported", "type": "invalid"}}
    server.replies = [(400, rejection), (200, completion('{"ok": true}'))]

    result = make_client().complete_json("Do it", schema=SCHEMA)

    assert result == {"ok": True}
    assert len(server.requests) == 2


def test_gives_up_with_a_clear_error_when_the_model_never_returns_json(server, make_client):
    server.reply_with("nope")

    with pytest.raises(ProviderError, match="no valid JSON"):
        make_client().complete_json("Do it", schema=SCHEMA)


def test_an_unreachable_server_error_names_the_endpoint():
    """The usual cause with a local server is that nothing is listening there."""
    client = OpenAIClient("test-model", api_key="not-needed", base_url="http://127.0.0.1:9/v1")
    client._client = client._client.with_options(max_retries=0)

    with pytest.raises(ProviderError, match=r"http://127\.0\.0\.1:9/v1"):
        client.complete("hello")


def test_local_servers_are_reached_even_when_a_system_proxy_is_configured(
    server, make_client, monkeypatch
):
    """VPN and proxy tools often export HTTP_PROXY without excluding localhost, which
    would send Ollama traffic into the proxy and fail with a baffling error."""
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    server.reply_with("reached")

    # Built after the environment is set up, the way a real run builds it.
    client = OpenAIClient("test-model", api_key="not-needed", base_url=server.url)

    assert client.complete("hello").text == "reached"


def test_a_local_client_caps_output_waits_long_and_does_not_retry(server):
    client = OpenAIClient("m", api_key="x", base_url=server.url, local=True)

    assert client.max_tokens == 4096
    assert client._client.max_retries == 0
    assert client._client.timeout == 1800.0


def test_a_cloud_client_keeps_the_sdk_defaults(server):
    client = OpenAIClient("m", api_key="x", base_url=server.url)

    assert client.max_tokens == 8192
    assert client._client.max_retries == 2
    assert client._client.timeout != 1800.0


def test_an_explicit_token_limit_beats_the_local_default(server):
    assert (
        OpenAIClient("m", api_key="x", base_url=server.url, local=True, max_tokens=999).max_tokens
        == 999
    )
