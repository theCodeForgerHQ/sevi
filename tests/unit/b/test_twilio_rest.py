"""Twilio trial REST adapter (B spec §5.7, contracts §8.6). A stub SDK client, no network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from twilio.base.exceptions import TwilioRestException  # type: ignore[import-untyped]

from sevi.adapters.twilio_rest import TwilioOptions, TwilioRest
from sevi.contracts.errors import ErrorCode, SeviError, TwilioError
from sevi.fakes.clock import FakeClock
from sevi.fakes.fake_twilio_rest import TEST_AUTH_TOKEN, FakeTwilioRest
from sevi.fakes.null_telemetry import NullTelemetry

SID = "AC" + "0" * 32
CALL = "CA" + "1" * 32
OPTS = TwilioOptions(SID, TEST_AUTH_TOKEN, "+15005550006")


class StubCalls:
    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.created: list[dict[str, Any]] = []
        self.ops: list[tuple[str, str]] = []

    def create(self, **kw: Any) -> Any:
        self.created.append(kw)
        nxt = self.script.pop(0) if self.script else None
        if isinstance(nxt, BaseException):
            raise nxt
        return SimpleNamespace(sid=CALL)

    def __call__(self, sid: str) -> Any:
        outer = self

        class One:
            def update(self, status: str) -> None:
                outer.ops.append(("update", status))
                nxt = outer.script.pop(0) if outer.script else None
                if isinstance(nxt, BaseException):
                    raise nxt

            def delete(self) -> None:
                outer.ops.append(("delete", sid))
                nxt = outer.script.pop(0) if outer.script else None
                if isinstance(nxt, BaseException):
                    raise nxt

        return One()


def _rest(script: list[Any] | None = None, **kw: Any) -> tuple[TwilioRest, StubCalls]:
    calls = StubCalls(script or [])
    client = SimpleNamespace(calls=calls, recordings=calls)
    return TwilioRest(OPTS, FakeClock(), NullTelemetry(), client=client, **kw), calls


def _exc(status: int, code: int = 20001) -> TwilioRestException:
    return TwilioRestException(status, "uri", msg="x", code=code)


@pytest.mark.asyncio
async def test_create_call_exact_args() -> None:
    r, calls = _rest()
    sid = await r.create_call(to_e164="+919876543210", session_id="s1", base="https://b", t="T")
    assert sid == CALL
    assert calls.created[0] == {
        "to": "+919876543210",
        "from_": "+15005550006",
        "url": "https://b/twilio/voice/s1?t=T",
        "method": "POST",
        "status_callback": "https://b/twilio/status/s1?t=T",
        "status_callback_event": ["initiated", "ringing", "answered", "completed"],
        "status_callback_method": "POST",
        "timeout": 30,
        "time_limit": 600,
        "record": False,
    }


@pytest.mark.asyncio
async def test_create_call_errors_never_retried() -> None:
    r, calls = _rest([_exc(400, 21219)])
    with pytest.raises(TwilioError) as e:
        await r.create_call(to_e164="+1", session_id="s", base="b", t="t")
    assert e.value.code is ErrorCode.TWILIO_DIAL_FAILED
    assert e.value.details == {"twilio_code": 21219, "status": 400}
    assert len(calls.created) == 1


@pytest.mark.asyncio
async def test_create_call_timeout_maybe_created() -> None:
    import time

    class Slow(StubCalls):
        def create(self, **kw: Any) -> Any:
            time.sleep(0.2)
            return SimpleNamespace(sid=CALL)

    r = TwilioRest(
        OPTS,
        FakeClock(),
        NullTelemetry(),
        client=SimpleNamespace(calls=Slow([])),
        sdk_timeout_s=0.05,
    )
    with pytest.raises(TwilioError) as e:
        await r.create_call(to_e164="+1", session_id="s", base="b", t="t")
    assert e.value.details == {"maybe_created": True}


@pytest.mark.asyncio
async def test_dial_guard_refuses_before_sdk_call() -> None:
    async def guard() -> None:
        raise SeviError(ErrorCode.TWILIO_TRIAL_MINUTES_STOP, "stop")

    r, calls = _rest(dial_guard=guard)
    with pytest.raises(SeviError) as e:
        await r.create_call(to_e164="+1", session_id="s", base="b", t="t")
    assert e.value.code is ErrorCode.TWILIO_TRIAL_MINUTES_STOP
    assert calls.created == []


@pytest.mark.asyncio
async def test_update_retries_5xx_and_delete_404_done() -> None:
    r, calls = _rest([_exc(503), None])
    await r.update_call(CALL, "completed")
    assert calls.ops == [("update", "completed"), ("update", "completed")]
    assert r.clock.sleeps == [500.0]  # type: ignore[attr-defined]
    r2, calls2 = _rest([_exc(404)])
    await r2.delete_call(CALL)
    assert calls2.ops == [("delete", CALL)]
    r3, calls3 = _rest([_exc(500), _exc(500), _exc(500)])
    with pytest.raises(TwilioError) as e:
        await r3.delete_recording("RE1")
    assert e.value.code is ErrorCode.TWILIO_REST_FAILED
    assert len(calls3.ops) == 3
    r4, calls4 = _rest([_exc(400)])
    with pytest.raises(TwilioError):
        await r4.update_call(CALL, "canceled")
    assert len(calls4.ops) == 1


@respx.mock
@pytest.mark.asyncio
async def test_fetch_recording_basic_auth_no_retry() -> None:
    url = "https://api.twilio.com/2010-04-01/Accounts/AC/Recordings/RE1"
    route = respx.get(url + ".wav").mock(return_value=httpx.Response(200, content=b"RIFFxx"))
    r, _ = _rest(http=httpx.AsyncClient())
    assert await r.fetch_recording_wav(url) == b"RIFFxx"
    assert route.calls[0].request.headers["authorization"].startswith("Basic ")
    respx.get(url + ".wav").mock(return_value=httpx.Response(500))
    with pytest.raises(TwilioError):
        await r.fetch_recording_wav(url)


def test_signature_validation_matches_fake_signer() -> None:
    r, _ = _rest()
    fake = FakeTwilioRest()
    url = "https://x.test/twilio/voice/s1?t=tok"
    form = {"CallSid": CALL, "Digits": "472"}
    assert r.validate(url, form, fake.sign(url, form))
    assert not r.validate(url, {**form, "Digits": "473"}, fake.sign(url, form))
