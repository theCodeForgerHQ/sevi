"""Fakes used by B's adapter and orchestrator tests (B spec §12.1, contracts §11.8)."""

from __future__ import annotations

import asyncio
import io
import json
import wave

import httpx
import pytest

from sevi.contracts.errors import ErrorCode, TwilioError
from sevi.contracts.telemetry import EventName
from sevi.fakes import fake_evon as fe
from sevi.fakes.fake_clock import FakeClock, ScaledClock
from sevi.fakes.fake_timbre import FakeTimbre
from sevi.fakes.fake_twilio_rest import FakeTwilioRest
from sevi.fakes.null_telemetry import NullTelemetry, RecordingTelemetry


@pytest.mark.asyncio
async def test_fake_clock_sleep_until_advances_instantly() -> None:
    c = FakeClock(start_mono=100.0)
    start_utc = c.now_utc()
    await c.sleep_until(156.0)
    assert c.mono_ms() == 156.0
    assert (c.now_utc() - start_utc).total_seconds() == pytest.approx(0.056)
    await c.sleep_until(10.0)  # in the past: no-op
    assert c.mono_ms() == 156.0
    assert c.sleeps == [56.0, 0.0]
    c.advance(1.5)
    assert c.mono_ms() == 1656.0
    with pytest.raises(ValueError, match="backwards"):
        c.advance_ms(-1)


@pytest.mark.asyncio
async def test_scaled_clock_runs_faster_than_real_time() -> None:
    c = ScaledClock(speed=100.0)
    t0 = c.mono_ms()
    await c.sleep_until(t0 + 500.0)  # 5 ms of real time
    assert c.mono_ms() >= t0 + 500.0


def test_telemetry_sinks() -> None:
    NullTelemetry().emit(EventName.STT_OPEN, x=1)
    rec = RecordingTelemetry()
    rec.emit(EventName.STT_OPEN, "INFO", header_mode="both")
    with rec.span(EventName.STT_OPEN):  # type: ignore[arg-type]
        pass
    assert rec.of(EventName.STT_OPEN) == [{"header_mode": "both"}]
    assert rec.names()[0] == "stt.open"


@pytest.mark.asyncio
async def test_fake_timbre_tone_by_audio_config_and_status_sequence() -> None:
    fake = FakeTimbre(status=[500, 200])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        url = "https://api.vachana.ai/api/v1/tts/inference"
        r1 = await http.post(url, json={"text": "x", "audio_config": {"container": "mulaw"}})
        assert r1.status_code == 500
        r2 = await http.post(url, json={"text": "x", "audio_config": {"container": "mulaw"}})
        assert r2.status_code == 200
        assert len(r2.content) == 4000  # 500 ms of 8 kHz mu-law
        r3 = await http.post(url, json={"text": "x", "audio_config": {"container": "wav"}})
        with wave.open(io.BytesIO(r3.content)) as w:
            assert w.getframerate() == 48000
    assert len(fake.requests) == 3


@pytest.mark.asyncio
async def test_fake_evon_script_health_and_recording() -> None:
    fake = fe.FakeEvon([fe.think_prefix(), fe.LENGTH, fe.http(503), fe.valid("ok text")])
    async with httpx.AsyncClient(transport=fake.transport(), base_url="http://evon") as http:
        r = await http.post("/v1/chat/completions", json={"model": "evon"})
        assert r.json()["choices"][0]["message"]["content"].startswith("<think>")
        r = await http.post("/v1/chat/completions", json={})
        assert r.json()["choices"][0]["finish_reason"] == "length"
        assert (await http.post("/v1/chat/completions", json={})).status_code == 503
        r = await http.post("/v1/chat/completions", json={})
        assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"text": "ok text"}
        assert (await http.get("/health")).json() == {"status": "ok"}
    assert fake.requests[0] == {"model": "evon"}


@pytest.mark.asyncio
async def test_fake_evon_refused_raises_connect_error() -> None:
    fake = fe.FakeEvon([fe.REFUSED])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        with pytest.raises(httpx.ConnectError):
            await http.post("http://evon/v1/chat/completions", json={})


@pytest.mark.asyncio
async def test_fake_twilio_rest_records_scripts_and_signs() -> None:
    fake = FakeTwilioRest()
    sid = await fake.create_call(
        to_e164="+919999999999", session_id="s1", base="https://x.test", t="tok"
    )
    assert sid.startswith("CA")
    assert len(sid) == 34
    assert fake.created[0].url == "https://x.test/twilio/voice/s1?t=tok"
    fake.fail_create(21219)
    with pytest.raises(TwilioError) as e:
        await fake.create_call(to_e164="+1", session_id="s1", base="b", t="t")
    assert e.value.code is ErrorCode.TWILIO_DIAL_FAILED
    assert e.value.details["twilio_code"] == 21219
    fake.timeout_create()
    with pytest.raises(TwilioError) as e:
        await fake.create_call(to_e164="+1", session_id="s1", base="b", t="t")
    assert e.value.details == {"maybe_created": True}
    form = {"CallSid": sid, "CallStatus": "ringing"}
    url = "https://x.test/twilio/status/s1?t=tok"
    sig = fake.sign(url, form)
    assert fake.validate(url, form, sig)
    assert not fake.validate(url, {**form, "CallStatus": "completed"}, sig)

    seen: dict[str, str] = {}

    async def app(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(app)) as http:
        r = await fake.emit_webhook(http, url, form)
    assert r.status_code == 204
    assert seen["x-twilio-signature"] == sig
    await asyncio.sleep(0)
