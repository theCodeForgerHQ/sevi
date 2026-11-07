"""Redaction and telemetry (contracts §10.1, §10.5; B spec §4.2-§4.3): U-OBS-01..06."""

from __future__ import annotations

import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import structlog
from pydantic import SecretStr

from sevi.app.logging import NUMBER_RE, redact, redact_processor, token_sha8
from sevi.app.telemetry import Telemetry, log_file_name
from sevi.contracts.telemetry import EventName, SpanName
from sevi.fakes.clock import FakeClock

PHONE_RE = re.compile(r"\+?\d[\d\s-]{9,}\d")


def _tel(
    tmp_path: Path, clock: FakeClock | None = None, **kw: object
) -> tuple[Telemetry, io.StringIO]:
    out = io.StringIO()
    tel = Telemetry(tmp_path / "logs", clock=clock or FakeClock(), stdout=out, strict=True, **kw)  # type: ignore[arg-type]
    return tel, out


def _lines(tmp_path: Path) -> list[dict[str, object]]:
    lines = []
    for p in sorted((tmp_path / "logs").glob("*.jsonl")):
        lines += [json.loads(x) for x in p.read_text().splitlines()]
    return lines


def test_U_OBS_01_twilio_status_form_redacted(tmp_path: Path) -> None:
    tel, _ = _tel(tmp_path)
    form = {
        "To": "+919876543210",
        "From": "+919812345678",
        "Called": "+919876543210",
        "Caller": "+919812345678",
        "CallerCity": "CHENNAI",
        "CallerState": "TN",
        "CalledZip": "600001",
        "ToCountry": "IN",
        "CallStatus": "completed",
        "note": "dialled +91 98765 43210 ok",
    }
    tel.emit(EventName.TWILIO_STATUS, "INFO", **form)
    tel.close()
    (line,) = _lines(tmp_path)
    raw = json.dumps(line)
    fields = line["fields"]
    assert isinstance(fields, dict)
    for k in (
        "To",
        "From",
        "Called",
        "Caller",
        "CallerCity",
        "CallerState",
        "CalledZip",
        "ToCountry",
    ):
        assert k not in fields
    assert fields["CallStatus"] == "completed"
    assert not PHONE_RE.search(raw)


def test_U_OBS_02_secret_keys_dropped_and_secretstr_masked() -> None:
    out = redact(
        {
            "ws_token": "abc",
            "api_key": "k",
            "x-api-key-id": "id",
            "foo_secret": "s",
            "ua": "Mozilla",
            "Report_Token": "r",
            "nested": {"auth_token": "t", "ok": 1},
            "pw": SecretStr("hunter2"),
            "kept": "yes",
        }
    )
    assert out == {"nested": {"ok": 1}, "pw": "**********", "kept": "yes"}
    assert redact_processor(None, "info", {"event": "x", "api_key": "k", "a": 1}) == {
        "event": "x",
        "a": 1,
    }
    assert token_sha8("tok") == token_sha8("tok")
    assert len(token_sha8("tok")) == 8


def test_U_OBS_03_number_scrubbed_in_strings() -> None:
    assert redact({"msg": "call 98765 43210 now"}) == {"msg": "call [REDACTED_NUMBER] now"}
    assert NUMBER_RE.sub("", "digits 123 only") == "digits 123 only"


def test_U_OBS_04_full_queue_never_raises_and_reports_drops(tmp_path: Path) -> None:
    clock = FakeClock()
    tel, out = _tel(tmp_path, clock, queue_max=3)
    for _ in range(5):
        tel.emit(EventName.ADMISSION_LEVEL, "INFO", level_name="L1")
    assert tel.dropped == 2
    errors = [json.loads(x) for x in out.getvalue().splitlines()]
    assert len(errors) == 1
    assert errors[0]["event"] == "error"
    assert errors[0]["error_code"] == "INTERNAL"
    assert errors[0]["fields"]["dropped"] == 1
    clock.advance(59)
    tel.emit(EventName.ADMISSION_LEVEL, "INFO")
    assert len(out.getvalue().splitlines()) == 1  # at most once per 60 s
    clock.advance(1)
    tel.emit(EventName.ADMISSION_LEVEL, "INFO")
    assert len(out.getvalue().splitlines()) == 2


def test_U_OBS_05_hourly_files_split_on_utc_hour(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 11, 7, 10, 59, 59, 999000, tzinfo=UTC))
    tel, _ = _tel(tmp_path, clock)
    tel.emit(EventName.ADMISSION_LEVEL, "INFO")
    clock.advance(0.001)
    tel.emit(EventName.ADMISSION_LEVEL, "INFO")
    tel.close()
    names = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert names == ["sevi-2026110710.jsonl", "sevi-2026110711.jsonl"]
    first = json.loads((tmp_path / "logs" / names[0]).read_text())
    assert first["ts"] == "2026-11-07T10:59:59.999Z"
    assert log_file_name(clock.now_utc()) == names[1]


def test_U_OBS_06_stdout_sink_filters(tmp_path: Path) -> None:
    tel, out = _tel(tmp_path)
    tel.emit(EventName.STT_ERROR, "WARNING", code="X")
    tel.emit(EventName.ADMISSION_LEVEL, "INFO")
    for ev in (EventName.RESULT_DECIDED, EventName.SESSION_ENDED, EventName.STT_TRANSCRIPT):
        tel.emit(ev, "INFO")
        tel.emit(ev, "ERROR")
    printed = [json.loads(x)["event"] for x in out.getvalue().splitlines()]
    assert printed == ["stt.error"]
    tel.close()
    assert len(_lines(tmp_path)) == 8  # everything still reaches the JSONL file


def test_level_filter_drops_debug_at_info(tmp_path: Path) -> None:
    tel, _ = _tel(tmp_path)
    tel.emit(EventName.STT_TRANSCRIPT, "DEBUG", text="one two three")
    tel.close()
    assert _lines(tmp_path) == []


def test_spans_and_context(tmp_path: Path) -> None:
    clock = FakeClock()
    tel, _ = _tel(tmp_path, clock, level="DEBUG")
    sid = "01890a5d-ac96-774b-bcce-b302099a8057"
    structlog.contextvars.bind_contextvars(session_id=sid)
    try:
        with tel.span(SpanName.STIM_RENDER, n=1):
            clock.advance(0.25)
        tel.mark_span(SpanName.MARK_RTT, 100.0, 50.0)
    finally:
        structlog.contextvars.clear_contextvars()
    tel.close()
    start, end, rtt = _lines(tmp_path)
    assert start["event"] == "span.start"
    assert start["session_id"] == sid
    assert end["event"] == "span.end"
    assert end["dur_ms"] == pytest.approx(250.0)
    assert end["span"] == "stim.render"
    assert rtt["dur_ms"] == 0.0


def test_unknown_event_strict_raises_lenient_drops(tmp_path: Path) -> None:
    tel, _ = _tel(tmp_path)
    with pytest.raises(ValueError, match="nope"):
        tel.emit("nope.event")
    out = io.StringIO()
    lenient = Telemetry(tmp_path / "l2", clock=FakeClock(), stdout=out)
    lenient.emit("nope.event")
    assert json.loads(out.getvalue())["error_code"] == "INTERNAL"
    with pytest.raises(NotImplementedError):
        lenient.subscribe()


@pytest.mark.asyncio
async def test_writer_task_flushes(tmp_path: Path) -> None:
    import asyncio

    tel, _ = _tel(tmp_path)
    task = asyncio.create_task(tel.run())
    tel.emit(EventName.ADMISSION_LEVEL, "INFO")
    for _ in range(50):
        await asyncio.sleep(0.05)
        if _lines(tmp_path):
            break
    task.cancel()
    tel.close()
    assert len(_lines(tmp_path)) == 1


@pytest.mark.parametrize(
    "key",
    [
        "X-Api-Key",
        "x-api-key",
        "Api-Key",
        "apiKey",
        "APIKEY",
        "api.key",
        "secret",
        "Secret",
        "clientSecret",
        "Authorization",
        "AUTHORIZATION",
        "X-Auth-Token",
        "wsToken",
        "ipHashKey",
        "hmac-key",
        "Password",
        "x_api_key_id",
        "Cookie",
        "User-Agent",
    ],
)
def test_U_OBS_02_key_names_any_case_or_separator_dropped(key: str) -> None:
    assert redact({key: "live-secret-value", "headers": {key: "v"}}) == {"headers": {}}


@pytest.mark.parametrize("key", ["key_id", "max_tokens", "monkey_count", "ok"])
def test_U_OBS_02_harmless_keys_kept(key: str) -> None:
    assert redact({key: 1}) == {key: 1}


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("client_ip", "203.0.113.9", "203.0.113.0/24"),
        ("ip", "2001:db8::1", "2001:db8::/48"),
        ("peer", "198.51.100.77", "198.51.100.0/24"),
        ("X-Forwarded-For", "203.0.113.9, 10.0.0.1", "203.0.113.0/24,10.0.0.0/24"),
        ("CF-Connecting-IP", "2001:db8:1:2::5", "2001:db8:1::/48"),
        ("clientIp", "not-an-ip", "[REDACTED_IP]"),
    ],
)
def test_U_OBS_07_raw_ips_truncated(key: str, value: str, expected: str) -> None:
    out = redact({key: value})
    assert out == {key: expected}
    assert value.split(",")[0] not in json.dumps(out) or value == "not-an-ip"


@pytest.mark.parametrize(
    "text",
    [
        "+919876543210",
        "98765 43210",
        "(987) 654-3210",
        "987.654.3210",
        "+91 (98765) 43210",
        "+1-987-654-3210",
    ],
)
def test_U_OBS_01_phone_formats_scrubbed(text: str) -> None:
    out = redact({"msg": f"call {text} now"})["msg"]
    assert out == "call [REDACTED_NUMBER] now"


def test_U_OBS_01_timestamps_and_small_numbers_survive() -> None:
    for s in ("2026-11-07T10:59:59.999Z", "srt -9.5 dB", "trial 12 of 24"):
        assert redact({"m": s}) == {"m": s}
