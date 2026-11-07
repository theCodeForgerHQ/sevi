"""U-TR-02..06, IT-TR-01/02/05/06: WebTransport over a loopback fake socket (B spec §6.2)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import json
import random
from typing import Any

import pytest
from b_fakes import Clock, RecordingTelemetry
from starlette.websockets import WebSocketDisconnect

from sevi.contracts.adapters import TClosed, TMark, TMedia, TransportEvent, TStart
from sevi.contracts.enums import SessionState
from sevi.contracts.errors import TransportError
from sevi.contracts.types import SessionId
from sevi.contracts.wire import UiKeypad
from sevi.transport.web_ws import AcceptView, WebTransport, check_accept, reject

SID = SessionId("01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b")
PAGES = "https://sevi.example.org"


class FakeWS:
    def __init__(self, *, never_drain: bool = False) -> None:
        self.inbox: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.sent: list[str] = []
        self.closed: list[tuple[int, str | None]] = []
        self.accepted = False
        self.never_drain = never_drain

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send_text(self, data: str) -> None:
        if self.never_drain:
            await asyncio.Event().wait()
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))

    def feed(self, obj: dict[str, Any] | str) -> None:
        self.inbox.put_nowait(obj if isinstance(obj, str) else json.dumps(obj))


def make(**kw: Any) -> tuple[WebTransport, FakeWS, Clock, RecordingTelemetry]:
    ws = FakeWS(never_drain=kw.pop("never_drain", False))
    clock, tel = Clock(), RecordingTelemetry()
    tr = WebTransport(ws, SID, clock, tel, send_timeout_s=kw.pop("send_timeout_s", 0.2), **kw)
    return tr, ws, clock, tel


async def collect(tr: WebTransport, timeout: float = 1.0) -> list[TransportEvent]:
    out: list[TransportEvent] = []

    async def run() -> None:
        async for ev in tr.events():
            out.append(ev)

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(run(), timeout)
    return out


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


def media_msg(seq: int, ts: int) -> dict[str, Any]:
    payload = base64.b64encode(b"\xff" * 160).decode()
    return {
        "event": "media",
        "sequenceNumber": str(seq),
        "media": {"timestamp": str(ts), "payload": payload},
    }


# ---------- accept (IT-TR-01 / IT-TR-02) ----------

AWAIT = AcceptView(SessionState.AWAIT_START, has_socket=False)


def _lookup(view: AcceptView | None):  # type: ignore[no-untyped-def]
    return lambda sid: view if sid == str(SID) else None


@pytest.mark.parametrize(
    ("sid", "origin", "reason"),
    [
        (str(SID), "https://evil.example", "bad_origin"),
        (str(SID), None, "bad_origin"),
        (None, PAGES, "sid_missing"),
        ("01933b6e-0000-7c2d-9e4b-2a1f0c9d8e7b", PAGES, "sid_unknown"),
    ],
)
def test_IT_TR_01_reject_4401(sid: str | None, origin: str | None, reason: str) -> None:
    r = check_accept(sid, origin, PAGES + "/", _lookup(AWAIT), pre_start_sockets=0)
    assert r is not None
    assert r.code == 4401
    assert r.reason == reason


def test_IT_TR_02_conflict_state_and_cap() -> None:
    sid = str(SID)
    r = check_accept(sid, PAGES, PAGES, _lookup(AcceptView(SessionState.AWAIT_START, True)), 0)
    assert r is not None
    assert r.code == 4409
    for st in (SessionState.TEST, SessionState.ENDED):
        r = check_accept(sid, PAGES, PAGES, _lookup(AcceptView(st, False)), 0)
        assert r is not None
        assert r.code == 4401
    r = check_accept(sid, PAGES, PAGES, _lookup(AWAIT), pre_start_sockets=2)
    assert r is not None
    assert r.code == 4429
    assert check_accept(sid, "HTTPS://SEVI.example.org/", PAGES, _lookup(AWAIT), 1) is None


@pytest.mark.asyncio
async def test_IT_TR_01_reject_accepts_then_closes() -> None:
    ws, tel = FakeWS(), RecordingTelemetry()
    r = check_accept(None, PAGES, PAGES, _lookup(AWAIT), 0)
    assert r is not None
    await reject(ws, r, tel)
    assert ws.accepted
    assert ws.closed == [(4401, "SESSION_TOKEN_INVALID")]
    assert tel.events[-1][0] == "ws.rejected"
    assert tel.events[-1][2]["reason"] == "sid_missing"


# ---------- inbound ----------


@pytest.mark.asyncio
async def test_U_TR_02_three_consecutive_bad_frames_close_4400() -> None:
    tr, ws, _, _ = make()
    tr.start()
    for _ in range(3):
        ws.feed("{bad")
    evs = await collect(tr)
    assert evs == [TClosed(code=4400, clean=False)]
    assert ws.closed
    assert ws.closed[0][0] == 4400
    assert any('"code":"TRANSPORT_PROTOCOL"' in s for s in ws.sent)


@pytest.mark.asyncio
async def test_U_TR_02_good_frame_resets_counter() -> None:
    tr, ws, _, _ = make()
    tr.start()
    for m in ("{bad", '{"event":"connected"}', "{bad", "{bad"):
        ws.feed(m)
    await settle()
    assert ws.closed == []
    await tr.close()


@pytest.mark.asyncio
async def test_U_TR_03_unknown_event_logged_no_close() -> None:
    tr, ws, _, tel = make()
    tr.start()
    ws.feed({"event": "weird"})
    await settle()
    assert "wire.unknown_message" in tel.names()
    assert ws.closed == []
    await tr.close()


@pytest.mark.asyncio
async def test_U_TR_04_clear_marks_cleared() -> None:
    tr, ws, _, _ = make()
    tr.start()
    await tr.send_mark("p0003")
    await tr.send_mark("p0004")  # never started on the page
    await tr.send_clear()
    await tr.send_mark("p0005")
    for i, n in enumerate(("p0003", "p0004", "p0005")):
        ws.feed({"event": "mark", "sequenceNumber": str(i + 1), "mark": {"name": n}})
    ws.inbox.put_nowait(WebSocketDisconnect(1000))
    evs = await collect(tr)
    marks = [e for e in evs if isinstance(e, TMark)]
    assert [(m.name, m.cleared) for m in marks] == [
        ("p0003", True),
        ("p0004", True),
        ("p0005", False),
    ]
    assert tr.outstanding_marks == set()


@pytest.mark.asyncio
async def test_U_TR_05_ping_pong_no_event() -> None:
    tr, ws, clock, _ = make()
    tr.start()
    clock.advance(10)
    ws.feed({"event": "kj", "sequenceNumber": "9", "kj": {"type": "ping"}})
    await settle()
    ws.inbox.put_nowait(WebSocketDisconnect(1000))
    evs = await collect(tr)
    assert evs == [TClosed(code=1000, clean=True)]
    assert tr.last_rx_mono == 10_000.0
    assert any('"type":"pong"' in s for s in ws.sent)


@pytest.mark.parametrize(
    ("stop", "code", "clean"),
    [(True, 1006, True), (False, 1006, False), (False, 1000, True), (False, 1001, True)],
)
@pytest.mark.asyncio
async def test_U_TR_06_clean_close(stop: bool, code: int, clean: bool) -> None:
    tr, ws, _, _ = make()
    tr.start()
    if stop:
        ws.feed({"event": "stop", "sequenceNumber": "5"})
    ws.inbox.put_nowait(WebSocketDisconnect(code))
    evs = await collect(tr)
    assert evs[-1] == TClosed(code=code, clean=clean)


@pytest.mark.asyncio
async def test_read_exception_without_stop_is_unclean() -> None:
    tr, ws, _, _ = make()
    tr.start()
    ws.inbox.put_nowait(RuntimeError("boom"))
    assert (await collect(tr))[-1] == TClosed(code=None, clean=False)


# ---------- outbound writer (IT-TR-05) ----------


@pytest.mark.asyncio
async def test_IT_TR_05_writer_stall_send_timeout() -> None:
    tr, ws, _, tel = make(never_drain=True, send_timeout_s=0.05)
    tr.start()
    await tr.send_ui(UiKeypad(enabled=True))
    evs = await collect(tr, timeout=1.05)  # SEND_TIMEOUT_S + 1
    assert evs[-1] == TClosed(code=None, clean=False)
    assert "ws.send_timeout" in tel.names()
    with pytest.raises(TransportError):
        await tr.send_media(b"\xff")
    await settle()
    assert ws.closed  # closed in the background


@pytest.mark.asyncio
async def test_IT_TR_05_65th_queued_message() -> None:
    tr, _, _, tel = make()  # writer not started: nothing drains
    for _ in range(64):
        await tr.send_mark("m")
    with pytest.raises(TransportError):
        await tr.send_mark("m")
    assert "ws.send_timeout" in tel.names()
    assert (await collect(tr))[-1] == TClosed(code=None, clean=False)


@pytest.mark.asyncio
async def test_writer_sends_in_order() -> None:
    tr, ws, _, _ = make()
    tr.start()
    await tr.send_media(b"\x01")
    await tr.send_mark("s001r0")
    await tr.close()
    assert [json.loads(s)["event"] for s in ws.sent] == ["media", "mark"]
    assert ws.closed[-1][0] == 1000
    await tr.close()  # idempotent
    with pytest.raises(TransportError):
        await tr.send_clear()


# ---------- capture time (IT-TR-06) and liveness ----------


@pytest.mark.asyncio
async def test_IT_TR_06_capture_time_filters_jitter_and_resets_at_start() -> None:
    tr, ws, clock, _ = make()
    rng = random.Random(7)  # noqa: S311 - test jitter
    base = clock.mono_ms()
    for i in range(50):
        ts = 20 * i
        clock.at((base + 1000 + ts + (20 if i == 0 else rng.uniform(20, 200))) / 1000)
        tr.handle_text(json.dumps(media_msg(i + 1, ts)))
    evs: list[TMedia] = []
    while not tr._events.empty():
        ev = tr._events.get_nowait()
        assert isinstance(ev, TMedia)
        evs.append(ev)
    caps = [e.capture_mono_ms for e in evs]
    assert all(c is not None for c in caps)
    # after the first frame (min offset = 20 ms jitter floor), spacing is exactly 20 ms
    assert all(abs((b - a) - 20.0) < 1e-6 for a, b in itertools.pairwise(caps) if a and b)
    clock.advance(5)
    tr.handle_text(
        json.dumps(
            {
                "event": "start",
                "sequenceNumber": "1",
                "start": {"streamSid": "WBx", "customParameters": {}},
            }
        )
    )
    assert isinstance(tr._events.get_nowait(), TStart)
    now = clock.mono_ms()
    tr.handle_text(json.dumps(media_msg(2, 0)))
    m = tr._events.get_nowait()
    assert isinstance(m, TMedia)
    assert m.capture_mono_ms == now
    assert tr.last_new_ts_mono == now


@pytest.mark.asyncio
async def test_rx_idle_liveness() -> None:
    tr, ws, clock, tel = make()
    tr.handle_text('{"event":"connected"}')
    clock.advance(34.9)
    assert not tr.rx_idle(clock.mono_ms(), 35)
    clock.advance(0.1)
    assert tr.rx_idle(clock.mono_ms(), 35)
    assert tr.rx_idle(clock.mono_ms(), 35)
    assert tel.names().count("ws.rx_idle") == 1
    tr.handle_text('{"event":"connected"}')
    assert not tr.rx_idle(clock.mono_ms(), 35)
