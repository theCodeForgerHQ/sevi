"""Browser `CallTransport` over the `/web/stream` WebSocket (B spec §6.2, contracts §6.1, §9.4).

- Every `send_*` only encodes and `put_nowait`s on a bounded outbound queue (64). One writer task
  sends each item under `asyncio.timeout(send_timeout_s)`. A timeout, a full queue or a send
  exception -> `fail()`: `ws.send_timeout`, `TClosed(code=None, clean=False)`, writer stops, the
  socket is closed in the background. Later `send_*` raise `TransportError(TRANSPORT_CLOSED)`.
- The reader task stamps `last_rx_mono` on every inbound message (`T_RX_IDLE` liveness) and maps
  each media frame to capture time: `capture_mono_ms = ts_ms + min(received - ts over the last
  50 frames)`, ring reset at each `start`.
- `check_accept` implements the S13/S13b admission of a socket: admitted `?sid=`, Pages Origin,
  one socket per AWAIT_START session, pre-start cap per client network.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Final, Protocol

from starlette.websockets import WebSocketDisconnect

from sevi.contracts.adapters import (
    Clock,
    TClosed,
    Telemetry,
    TMark,
    TMedia,
    TransportEvent,
    TStart,
    TStop,
)
from sevi.contracts.enums import Channel, SessionState
from sevi.contracts.errors import WS_CLOSE_CODE, ErrorCode, TransportError
from sevi.contracts.telemetry import EventName
from sevi.contracts.types import SessionId
from sevi.contracts.wire import ServerUiMessage, UiError, UiPong
from sevi.transport.base import (
    WireParseError,
    encode_clear,
    encode_mark,
    encode_media,
    encode_ui,
    parse_client_message,
    stream_sid_for,
)

OUTQ_MAX: Final = 64
CAPTURE_OFFSET_RING: Final = 50
MAX_CONSECUTIVE_PARSE_ERRORS: Final = 3
BACKGROUND_CLOSE_TIMEOUT_S: Final = 1.0
PRE_START_SOCKETS_PER_IP: Final = 2
CLEAN_CLOSE_CODES: Final = frozenset({1000, 1001})

_CLOSE_CODE_TO_ERROR: Final = {v: k for k, v in WS_CLOSE_CODE.items()}


class WebSocketLike(Protocol):
    """The subset of `starlette.websockets.WebSocket` the transport uses."""

    async def accept(self) -> None: ...
    async def receive_text(self) -> str: ...  # raises WebSocketDisconnect(code)
    async def send_text(self, data: str) -> None: ...
    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


# ---------- accept (S13 / S13b) ----------


@dataclass(frozen=True)
class AcceptView:
    """What the registry knows about the session named by `?sid=`."""

    state: SessionState
    has_socket: bool


@dataclass(frozen=True)
class Rejection:
    code: int
    error: ErrorCode
    reason: str


def _norm_origin(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def check_accept(
    sid: str | None,
    origin: str | None,
    pages_origin: str,
    lookup: Callable[[str], AcceptView | None],
    pre_start_sockets: int,
    pre_start_cap: int = PRE_START_SOCKETS_PER_IP,
) -> Rejection | None:
    """None = accept. Else the close code the route uses right after `accept()`."""

    def rej(err: ErrorCode, reason: str) -> Rejection:
        return Rejection(WS_CLOSE_CODE[err], err, reason)

    if origin is None or _norm_origin(origin) != _norm_origin(pages_origin):
        return rej(ErrorCode.SESSION_TOKEN_INVALID, "bad_origin")
    if not sid:
        return rej(ErrorCode.SESSION_TOKEN_INVALID, "sid_missing")
    view = lookup(sid)
    if view is None:
        return rej(ErrorCode.SESSION_TOKEN_INVALID, "sid_unknown")
    if view.state is not SessionState.AWAIT_START:
        return rej(ErrorCode.SESSION_TOKEN_INVALID, "not_await_start")
    if view.has_socket:
        return rej(ErrorCode.SESSION_STATE_CONFLICT, "socket_exists")
    if pre_start_sockets >= pre_start_cap:
        return rej(ErrorCode.RATE_LIMITED_IP, "pre_start_cap")
    return None


async def reject(ws: WebSocketLike, rejection: Rejection, telemetry: Telemetry) -> None:
    """Accept, then close at once with the rejection code; `ws.rejected`. No state change."""
    with contextlib.suppress(Exception):
        await ws.accept()
    with contextlib.suppress(Exception):
        await ws.close(rejection.code, rejection.error.value)
    telemetry.emit(EventName.WS_REJECTED, "INFO", reason=rejection.reason, code=rejection.code)


# ---------- transport ----------


class WebTransport:
    channel = Channel.BROWSER

    def __init__(
        self,
        ws: WebSocketLike,
        session_id: SessionId,
        clock: Clock,
        telemetry: Telemetry,
        *,
        send_timeout_s: float,
        queue_max: int = OUTQ_MAX,
    ) -> None:
        self._ws = ws
        self._clock = clock
        self._tel = telemetry
        self._send_timeout_s = send_timeout_s
        self.stream_sid = stream_sid_for(session_id)
        self._outq: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        self._events: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._writer: asyncio.Task[None] | None = None
        self._reader: asyncio.Task[None] | None = None
        self._bg: set[asyncio.Task[None]] = set()
        self._failed = False
        self._closing = False
        self._closed_pushed = False
        self._parse_errors = 0
        self._offsets: deque[float] = deque(maxlen=CAPTURE_OFFSET_RING)
        self._max_ts: int | None = None
        self._cleared: set[str] = set()
        self._idle_reported = False
        self.stop_received = False
        self.outstanding_marks: set[str] = set()
        self.last_rx_mono: float = clock.mono_ms()
        self.last_new_ts_mono: float | None = None

    # ----- lifecycle

    def start(self) -> None:
        """Start the writer and reader tasks (call once, right after accept)."""
        if self._writer is None:
            self._writer = asyncio.create_task(self._write_loop(), name="ws_writer")
            self._reader = asyncio.create_task(self._read_loop(), name="ws_reader")

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Idempotent. A code != 1000 first sends ui `error` (best effort through the queue)."""
        if self._closing:
            return
        self._closing = True
        if code != 1000 and not self._failed:
            err = _CLOSE_CODE_TO_ERROR.get(code, ErrorCode.INTERNAL)
            with contextlib.suppress(asyncio.QueueFull):
                self._outq.put_nowait(
                    encode_ui(
                        self.stream_sid, UiError(code=err, message=(reason or err.value)[:300])
                    )
                )
        if self._writer is not None and not self._failed:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._send_timeout_s):
                    await self._outq.join()
        await self._stop_tasks()
        with contextlib.suppress(Exception):
            async with asyncio.timeout(BACKGROUND_CLOSE_TIMEOUT_S):
                await self._ws.close(code, reason or None)
        self._push_closed(code, clean=code in CLEAN_CLOSE_CODES)

    async def _stop_tasks(self) -> None:
        for t in (self._writer, self._reader):
            if t is not None and not t.done() and t is not asyncio.current_task():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    # ----- outbound (enqueue only)

    def _enqueue(self, item: str) -> None:
        if self._failed or self._closing:
            raise TransportError(ErrorCode.TRANSPORT_CLOSED, "transport closed")
        try:
            self._outq.put_nowait(item)
        except asyncio.QueueFull:
            self._fail("queue_full")
            raise TransportError(ErrorCode.TRANSPORT_CLOSED, "outbound queue full") from None

    async def send_media(self, ulaw: bytes) -> None:
        self._enqueue(encode_media(self.stream_sid, ulaw))

    async def send_mark(self, name: str) -> None:
        self._enqueue(encode_mark(self.stream_sid, name))
        self.outstanding_marks.add(name)

    async def send_clear(self) -> None:
        self._enqueue(encode_clear(self.stream_sid))
        self._cleared |= self.outstanding_marks

    async def send_ui(self, msg: ServerUiMessage) -> None:
        self._enqueue(encode_ui(self.stream_sid, msg))

    async def _write_loop(self) -> None:
        while True:
            item = await self._outq.get()
            try:
                async with asyncio.timeout(self._send_timeout_s):
                    await self._ws.send_text(item)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                self._fail("send_timeout")
                return
            except Exception:
                self._fail("send_error")
                return
            finally:
                self._outq.task_done()

    def _fail(self, reason: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._tel.emit(
            EventName.WS_SEND_TIMEOUT, "WARNING", queued=self._outq.qsize(), reason=reason
        )
        self._push_closed(None, clean=False)
        writer = self._writer
        if writer is not None and writer is not asyncio.current_task() and not writer.done():
            writer.cancel()
        task = asyncio.get_running_loop().create_task(self._background_close())
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _background_close(self) -> None:
        with contextlib.suppress(Exception):
            async with asyncio.timeout(BACKGROUND_CLOSE_TIMEOUT_S):
                await self._ws.close(1011, None)

    # ----- inbound

    def events(self) -> AsyncIterator[TransportEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[TransportEvent]:
        while True:
            ev = await self._events.get()
            yield ev
            if isinstance(ev, TClosed):
                return

    def _push_closed(self, code: int | None, *, clean: bool) -> None:
        if not self._closed_pushed:
            self._closed_pushed = True
            self._events.put_nowait(TClosed(code=code, clean=clean))

    async def _read_loop(self) -> None:
        try:
            while True:
                raw = await self._ws.receive_text()
                self.handle_text(raw)
                if self._closed_pushed:
                    return
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect as e:
            self._push_closed(e.code, clean=self.stop_received or e.code in CLEAN_CLOSE_CODES)
            self._stop_writer()
        except Exception:
            self._push_closed(None, clean=self.stop_received)
            self._stop_writer()

    def _stop_writer(self) -> None:
        """The peer is gone: nothing more can be sent."""
        self._closing = True
        if self._writer is not None and not self._writer.done():
            self._writer.cancel()

    def handle_text(self, raw: str) -> None:
        """Process one inbound text frame (the reader task calls this; tests may too)."""
        now = self._clock.mono_ms()
        self.last_rx_mono = now
        self._idle_reported = False
        try:
            parsed = parse_client_message(raw, now)
        except WireParseError as e:
            self._parse_errors += 1
            if self._parse_errors >= MAX_CONSECUTIVE_PARSE_ERRORS:
                self._protocol_close(str(e))
            return
        self._parse_errors = 0
        if parsed is None:
            self._tel.emit(EventName.WIRE_UNKNOWN_MESSAGE, "DEBUG", dep="browser")
            return
        if parsed == "connected":
            return
        if parsed == "ping":
            with contextlib.suppress(TransportError):
                self._enqueue(encode_ui(self.stream_sid, UiPong()))
            return
        ev: TransportEvent = parsed
        if isinstance(ev, TStart):
            self._offsets.clear()
            self._max_ts = None
        elif isinstance(ev, TMedia):
            ev = self._stamp_capture(ev, now)
        elif isinstance(ev, TMark):
            cleared = ev.name in self._cleared
            self._cleared.discard(ev.name)
            self.outstanding_marks.discard(ev.name)
            ev = ev.model_copy(update={"cleared": cleared})
        elif isinstance(ev, TStop):
            self.stop_received = True
        self._events.put_nowait(ev)

    def _stamp_capture(self, m: TMedia, now: float) -> TMedia:
        if m.ts_ms is None:
            return m.model_copy(update={"capture_mono_ms": m.received_mono_ms})
        self._offsets.append(m.received_mono_ms - m.ts_ms)
        cap = m.ts_ms + min(self._offsets)
        if self._max_ts is None or m.ts_ms > self._max_ts:
            self._max_ts = m.ts_ms
            self.last_new_ts_mono = now
        return m.model_copy(update={"capture_mono_ms": cap})

    def _protocol_close(self, detail: str) -> None:
        self._parse_errors = 0
        task = asyncio.get_running_loop().create_task(
            self.close(WS_CLOSE_CODE[ErrorCode.TRANSPORT_PROTOCOL], "TRANSPORT_PROTOCOL")
        )
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ----- liveness

    def rx_idle(self, now_mono_ms: float, t_rx_idle_s: float) -> bool:
        """True once no inbound message arrived for `t_rx_idle_s` (S17/S65 is the orchestrator's
        timer; this emits `ws.rx_idle` once per idle period)."""
        idle = now_mono_ms - self.last_rx_mono >= t_rx_idle_s * 1000.0
        if idle and not self._idle_reported:
            self._idle_reported = True
            self._tel.emit(
                EventName.WS_RX_IDLE, "WARNING", idle_ms=round(now_mono_ms - self.last_rx_mono)
            )
        return idle
