"""Evon client (B spec §5.6.2, §13.5 U-EVN-01..07). fake_evon only."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from sevi.adapters.evon import NO_DIGITS_PATTERN, EvonClient, EvonOptions, schema_for
from sevi.contracts.adapters import LlmRequest
from sevi.contracts.enums import Lang
from sevi.contracts.errors import BreakerOpenError, ErrorCode, LlmError
from sevi.fakes import fake_evon as fe
from sevi.fakes.clock import FakeClock
from sevi.fakes.null_telemetry import NullTelemetry
from sevi.fakes.recording_deps import RecordingBreaker

USER = json.dumps({"band": "REFER", "wording": "W_REF_SRT spoken text"}, ensure_ascii=False)


def _req(purpose: Any = "EXPLAIN_SPOKEN", max_tokens: int = 256) -> LlmRequest:
    return LlmRequest(
        purpose=purpose,
        lang=Lang.TA,
        system="SYSTEM",
        user=USER,
        json_schema={},
        max_tokens=max_tokens,
        temperature=0.6,
    )


def _client(fake: fe.FakeEvon, **opts: Any) -> tuple[EvonClient, RecordingBreaker]:
    b = RecordingBreaker()
    o = EvonOptions(base_url="http://100.101.102.103:8080/v1", api_key="EK", **opts)
    http = httpx.AsyncClient(transport=fake.transport())
    return EvonClient(o, http, b, FakeClock(), NullTelemetry()), b


@pytest.mark.asyncio
async def test_U_EVN_01_body_llamacpp_exact() -> None:
    fake = fe.FakeEvon()
    c, _ = _client(fake)
    r = await c.complete_json(_req())
    assert r.parsed == {"text": fe.DEFAULT_TEXT}
    assert fake.headers[0]["authorization"] == "Bearer EK"
    assert fake.requests[0] == {
        "model": "evon",
        "messages": [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": USER},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 300,
                        "pattern": NO_DIGITS_PATTERN,
                    }
                },
            },
        },
    }
    assert NO_DIGITS_PATTERN == "^[^0-9०-९௦-௲೦-೯]*$"  # noqa: RUF001
    await c.complete_json(_req("REPORT_PROSE", 640))
    rp = fake.requests[1]
    assert rp["max_tokens"] == 640
    assert rp["response_format"]["schema"]["properties"]["text"]["maxLength"] == 600
    assert "pattern" not in schema_for("TRANSLATION_DRAFT", True)["properties"]["text"]
    c2, _ = _client(fake, schema_pattern=False)
    await c2.complete_json(_req())
    assert "pattern" not in fake.requests[2]["response_format"]["schema"]["properties"]["text"]


@pytest.mark.asyncio
async def test_U_EVN_02_openai_style() -> None:
    fake = fe.FakeEvon()
    c, _ = _client(fake, response_format_style="openai")
    await c.complete_json(_req())
    rf = fake.requests[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "kj_explain_spoken"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["required"] == ["text"]


@pytest.mark.asyncio
async def test_U_EVN_03_think_prefix_stripped() -> None:
    c, _ = _client(fe.FakeEvon([fe.think_prefix()]))
    r = await c.complete_json(_req())
    assert r.parsed["text"] == fe.DEFAULT_TEXT
    assert r.tokens_out == 61
    assert r.finish_reason == "stop"


@pytest.mark.parametrize(
    "out", [fe.LENGTH, fe.INVALID_JSON, fe.EXTRA_KEY, fe.valid("x" * 19), fe.NUMERAL]
)
@pytest.mark.asyncio
async def test_U_EVN_04_schema_invalid_not_counted(out: fe.Out) -> None:
    c, b = _client(fe.FakeEvon([out]))
    with pytest.raises(LlmError) as e:
        await c.complete_json(_req())
    assert e.value.code is ErrorCode.LLM_SCHEMA_INVALID
    assert b.failures == 0


@pytest.mark.parametrize(
    ("out", "code"),
    [
        (fe.http(503), ErrorCode.LLM_UNAVAILABLE),
        (fe.REFUSED, ErrorCode.LLM_UNAVAILABLE),
        (fe.slow(500), ErrorCode.LLM_TIMEOUT),
    ],
)
@pytest.mark.asyncio
async def test_U_EVN_05_unavailable_and_timeout_count_breaker(out: fe.Out, code: ErrorCode) -> None:
    c, b = _client(fe.FakeEvon([out]), timeout_s=0.05)  # 6 s scaled down for the test
    with pytest.raises(LlmError) as e:
        await c.complete_json(_req())
    assert e.value.code is code
    assert b.failures == 1


@pytest.mark.asyncio
async def test_deadline_kwarg_and_breaker_open() -> None:
    c, b = _client(fe.FakeEvon([fe.slow(300)]))
    with pytest.raises(LlmError) as e:
        await c.complete_json(_req(), timeout_s=0.05)
    assert e.value.code is ErrorCode.LLM_TIMEOUT
    b.open = True
    with pytest.raises(BreakerOpenError) as e2:
        await c.complete_json(_req())
    assert e2.value.code is ErrorCode.LLM_BREAKER_OPEN


@pytest.mark.asyncio
async def test_U_EVN_06_request_carries_only_band_and_wording() -> None:
    fake = fe.FakeEvon()
    c, _ = _client(fake)
    await c.complete_json(_req())
    user = fake.requests[0]["messages"][1]["content"]
    assert json.loads(user).keys() == {"band", "wording"}
    assert "ignore previous instructions" not in json.dumps(fake.requests[0])


@pytest.mark.asyncio
async def test_U_EVN_07_health() -> None:
    c, _ = _client(fe.FakeEvon())
    assert await c.health() is True
    c2, _ = _client(fe.FakeEvon(health_status=503, health_body={"status": "loading model"}))
    assert await c2.health() is False
    c3, _ = _client(fe.FakeEvon(health_body={"status": "busy"}))
    assert await c3.health() is False
