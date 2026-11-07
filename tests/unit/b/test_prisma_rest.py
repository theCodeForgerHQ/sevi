"""Prisma REST adapter (B spec §5.4, §13.4 U-PRS-20..22, F-PRS-20/21). respx only."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import httpx
import pytest
import respx

from sevi.adapters.prisma_rest import (
    PrismaRestClient,
    cache_fields,
    cache_key,
    form_fields,
    pcm16_wav,
)
from sevi.contracts.adapters import SttRestRequest
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.contracts.errors import BudgetExceeded, ErrorCode, SeviError, SttError
from sevi.fakes.fake_clock import FakeClock
from sevi.fakes.memory_ledger import MemoryBreaker, MemoryLedger, MemorySemaphore
from sevi.fakes.null_telemetry import NullTelemetry

URL = "https://api.vachana.ai/stt/v3"
PCM = b"\x00\x01" * 8000  # 1 s at 8 kHz
OK = {
    "success": True,
    "request_id": "5c1f",
    "transcript": "472",
    "model": "prisma-v3.0",
    "processing_time": 0.62,
}


def _req(**kw: object) -> SttRestRequest:
    base: dict[str, object] = {
        "wav_8k_pcm16": PCM,
        "lang": Lang.TA,
        "model_version": ModelVersion.V3,
        "format": "transcribe",
        "keyterms": ("ஒன்று", "onru", "one"),
    }
    base.update(kw)
    return SttRestRequest.model_validate(base)


def _client(
    tmp: Path | None, ledger: MemoryLedger | None = None
) -> tuple[PrismaRestClient, MemoryLedger, MemorySemaphore, FakeClock]:
    led, sem, clock = ledger or MemoryLedger(), MemorySemaphore(), FakeClock()
    c = PrismaRestClient(
        {KeyId.A: "KA"},
        sem,
        led,
        httpx.AsyncClient(),
        clock,
        NullTelemetry(),
        cache_dir=tmp,
        rng=random.Random(3),  # noqa: S311
    )
    return c, led, sem, clock


def _multipart_fields(request: httpx.Request) -> dict[str, str]:
    body = request.content.decode("latin-1")
    out = {}
    for part in body.split("--" + request.headers["content-type"].split("boundary=")[1]):
        if 'name="' in part and "filename" not in part:
            name = part.split('name="')[1].split('"')[0]
            out[name] = part.split("\r\n\r\n", 1)[1].rsplit("\r\n", 1)[0]
    return {k: v.encode("latin-1").decode() for k, v in out.items()}


@respx.mock
@pytest.mark.asyncio
async def test_U_PRS_20_multipart_fields_v3_and_v25() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=OK))
    c, _, sem, _ = _client(None)
    await c.transcribe(_req(), request_id="cal-1")
    sent = route.calls[0].request
    assert sent.headers["x-api-key-id"] == "KA"
    assert sent.headers["x-api-request-id"] == "cal-1"
    assert _multipart_fields(sent) == {
        "model": "prisma-v3.0",
        "language_code": "ta-IN,en-IN",
        "format": "transcribe",
        "keyterms": '["ஒன்று","onru","one"]',
    }
    assert b'filename="clip.wav"' in sent.content
    assert b"RIFF" in sent.content
    assert sem.taken == [(KeyId.A, "rest")]
    assert form_fields(_req(model_version=ModelVersion.V2_5)) == {
        "model": "prisma-v2.5",
        "language_code": "ta-IN",
        "format": "transcribe",
        "bias_list": "onru,one",
        "bias_score": "1.5",
    }


@respx.mock
@pytest.mark.asyncio
async def test_U_PRS_21_cache_hit_no_http_no_permit_no_ledger(tmp_path: Path) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=OK))
    c, led, sem, _ = _client(tmp_path)
    r1 = await c.transcribe(_req())
    r2 = await c.transcribe(_req())
    assert (r1.cached, r2.cached) == (False, True)
    assert route.call_count == 1
    assert len(sem.taken) == 1
    assert len(led.events) == 1
    assert led.events[0].stt_rest_calls == 1
    assert led.events[0].stt_s == pytest.approx(1.0)
    wav_sha = hashlib.sha256(pcm16_wav(PCM, 8000)).hexdigest()
    golden = {
        "endpoint": "stt/v3",
        "audio_sha256": wav_sha,
        "model": "prisma-v3.0",
        "language_code": "ta-IN,en-IN",
        "format": "transcribe",
        "keyterms": ["ஒன்று", "onru", "one"],
        "bias": None,
        "repeat_index": 0,
    }
    assert cache_fields(_req(), wav_sha) == golden
    canon = json.dumps(golden, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert r1.cache_key == cache_key(_req(), wav_sha) == hashlib.sha256(canon.encode()).hexdigest()
    assert cache_key(_req(repeat_index=1), wav_sha) != r1.cache_key


@respx.mock
@pytest.mark.asyncio
async def test_U_PRS_22_empty_transcript_valid_and_success_false() -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(200, json={**OK, "transcript": ""}),
            httpx.Response(200, json={"success": False}),
        ]
    )
    c, _, _, _ = _client(None)
    assert (await c.transcribe(_req())).transcript == ""
    with pytest.raises(SttError) as e:
        await c.transcribe(_req())
    assert e.value.code is ErrorCode.STT_REST_FAILED


@respx.mock
@pytest.mark.asyncio
async def test_F_PRS_20_504_x3_then_200_and_400_no_retry(tmp_path: Path) -> None:
    gw = httpx.Response(504, json={"error": "Gateway Timeout", "code": "UPSTREAM_TIMEOUT"})
    route = respx.post(URL).mock(side_effect=[gw, gw, gw, httpx.Response(200, json=OK)])
    c, led, _, clock = _client(tmp_path)
    r = await c.transcribe(_req())
    assert r.transcript == "472"
    assert route.call_count == 4
    exp = random.Random(3)  # noqa: S311
    want = [exp.uniform(0, 0.5) * 1000, exp.uniform(0, 1.0) * 1000, exp.uniform(0, 2.0) * 1000]
    assert clock.sleeps == pytest.approx(want)
    assert (await c.transcribe(_req())).cached
    before = route.call_count
    respx.post(URL).mock(return_value=httpx.Response(400, json={"detail": {}}))
    with pytest.raises(SttError):
        await c.transcribe(_req(repeat_index=2))
    assert route.call_count == before + 1
    assert led.reservations == {}  # 400 billed nothing: released


@respx.mock
@pytest.mark.asyncio
async def test_F_PRS_21_phone_record_no_retry_timeout_counts_breaker() -> None:
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    c, _, _, _ = _client(None)
    b = MemoryBreaker()
    with pytest.raises(SttError) as e:
        await c.transcribe(_req(), timeout_s=0.05, retries=0, breaker=b)
    assert e.value.code is ErrorCode.STT_REST_FAILED
    assert b.failures == 1


@respx.mock
@pytest.mark.asyncio
async def test_rest_auth_credits_budget_and_cal_stop() -> None:
    route = respx.post(URL).mock(side_effect=[httpx.Response(401), httpx.Response(402)])
    c, led, _, _ = _client(None)
    with pytest.raises(SttError) as e:
        await c.transcribe(_req())
    assert e.value.code is ErrorCode.STT_AUTH
    with pytest.raises(SttError) as e:
        await c.transcribe(_req())
    assert e.value.code is ErrorCode.STT_CREDITS
    assert KeyId.A in led.exhausted
    with pytest.raises(BudgetExceeded):  # exhausted key: refused before any request
        await c.transcribe(_req())
    assert route.call_count == 2
    c2, led2, _, _ = _client(None, MemoryLedger(budget=0.5))  # 1 s of audio > 0.5 credits
    with pytest.raises(BudgetExceeded):
        await c2.transcribe(_req())
    led2.budget, led2.cal_stopped = 100.0, True
    with pytest.raises(SeviError) as e:
        await c2.transcribe(_req())
    assert e.value.code is ErrorCode.CREDIT_STOP
    assert route.call_count == 2
