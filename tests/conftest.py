from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from shorts_agent.config import AppConfig, ChannelConfig
from shorts_agent.llm.base import LLMClient, LLMResponse


class FakeLLM(LLMClient):
    """Scripted LLM stand-in.

    ``json_responses`` is consumed in order by ``complete_json``, so a test can
    script a multi-call sequence (e.g. ideas, then script, then metadata).
    """

    def __init__(
        self,
        *,
        text: str = "fake text",
        json_responses: list[Any] | None = None,
        error: Exception | None = None,
    ):
        super().__init__("fake-model")
        self.text = text
        self.json_responses = list(json_responses or [])
        self.error = error
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        return LLMResponse(text=self.text)

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        system: str | None = None,
        max_tokens: int | None = None,
        attempts: int = 3,
    ) -> Any:
        self.prompts.append(prompt)
        self.systems.append(system)
        if self.error is not None:
            raise self.error
        if not self.json_responses:
            raise AssertionError("FakeLLM ran out of scripted JSON responses")
        return self.json_responses.pop(0)


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return AppConfig(
        channel=ChannelConfig(
            name="Test Channel",
            niche="personal finance for beginners",
            persona="A concise, upbeat finance coach.",
            default_hashtags=["#Shorts", "#money"],
        ),
        base_dir=tmp_path,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    """A minimal OpenAI chat-completions response body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


@dataclass
class Recorded:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class FakeServer:
    """A real OpenAI-compatible HTTP server on localhost, scripted per test."""

    url: str
    requests: list[Recorded] = field(default_factory=list)
    # Each entry is (status, json body), consumed in order; the last one repeats.
    replies: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    # Model ids reported by GET /v1/models, the way Ollama lists what is pulled.
    models: list[str] = field(default_factory=list)

    def reply_with(self, *contents: str) -> None:
        self.replies = [(200, completion(c)) for c in contents]


@pytest.fixture
def server() -> Iterator[FakeServer]:
    state = FakeServer(url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # keep test output quiet
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path.endswith("/models"):
                data = [{"id": m, "object": "model"} for m in state.models]
                self._send(200, {"object": "list", "data": data})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append(
                Recorded(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
            )
            index = min(len(state.requests), len(state.replies)) - 1
            status, payload = state.replies[index]
            self._send(status, payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    # A short poll interval keeps shutdown (and so every test) from waiting 0.5s.
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
