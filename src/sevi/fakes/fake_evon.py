"""Fake Evon (llama-server, OpenAI-compatible) endpoint (contracts §11.8, B spec §12.1).

An `httpx.MockTransport` handler serving `POST {base}/chat/completions` and `GET /health`.
Each completion pops the next scripted output (the last one repeats; default `valid()`):

`valid(text)`, `invalid_json`, `numeral`, `number_word`, `medical`, `empty`, `slow(ms, text)`,
`length` (`finish_reason=length`), `http(status)`, `think_prefix(text)`, `extra_key`,
`refused` (raises `httpx.ConnectError`, i.e. connection refused).

`health_status` / `health_body` drive `/health`. Every completion request body is recorded in
`requests` (U-EVN-06 asserts on it).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_TEXT = "Your result is ready. Please read the page for what it means for you."


@dataclass(frozen=True)
class Out:
    kind: str
    text: str = DEFAULT_TEXT
    ms: int = 0
    status: int = 200


def valid(text: str = DEFAULT_TEXT) -> Out:
    return Out("valid", text)


def slow(ms: int, text: str = DEFAULT_TEXT) -> Out:
    return Out("slow", text, ms=ms)


def http(status: int) -> Out:
    return Out("http", status=status)


def think_prefix(text: str = DEFAULT_TEXT) -> Out:
    return Out("think_prefix", text)


INVALID_JSON = Out("invalid_json")
NUMERAL = Out("numeral", "Your hearing score was 7 which is fine for your age group today.")
NUMBER_WORD = Out("number_word", "You heard most of the seven digits clearly in the noise today.")
MEDICAL = Out("medical", "You should buy a hearing aid soon because your ears are damaged.")
EMPTY = Out("empty", "")
LENGTH = Out("length")
EXTRA_KEY = Out("extra_key")
REFUSED = Out("refused")


class FakeEvon:
    def __init__(
        self,
        script: list[Out] | None = None,
        *,
        health_status: int = 200,
        health_body: dict[str, Any] | None = None,
    ) -> None:
        self.script = list(script or [valid()])
        self.health_status = health_status
        self.health_body = health_body if health_body is not None else {"status": "ok"}
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.health_calls = 0

    def _next(self) -> Out:
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    @staticmethod
    def _completion(content: str, finish: str = "stop") -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "evon",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 212, "completion_tokens": 61, "total_tokens": 273},
            "timings": {"predicted_per_second": 18.4},
        }

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path.endswith("/health"):
            self.health_calls += 1
            return httpx.Response(self.health_status, json=self.health_body)
        if request.method != "POST" or not path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": "not found"})
        self.requests.append(json.loads(request.content or b"{}"))
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        out = self._next()
        if out.kind == "refused":
            raise httpx.ConnectError("connection refused", request=request)
        if out.kind == "http":
            return httpx.Response(out.status, json={"error": {"message": "loading model"}})
        if out.kind == "slow":
            await asyncio.sleep(out.ms / 1000.0)
        if out.kind == "invalid_json":
            return httpx.Response(200, json=self._completion("Sure! Here is the text: {text"))
        if out.kind == "length":
            return httpx.Response(200, json=self._completion('{"text":"Your res', "length"))
        if out.kind == "extra_key":
            content = json.dumps({"text": DEFAULT_TEXT, "band": "PASS"})
            return httpx.Response(200, json=self._completion(content))
        content = json.dumps({"text": out.text}, ensure_ascii=False)
        if out.kind == "think_prefix":
            content = "<think>\nplanning the answer\n</think>\n" + content
        return httpx.Response(200, json=self._completion(content))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
