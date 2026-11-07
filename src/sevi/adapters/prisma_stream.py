"""Prisma streaming STT client (B spec §5.3, contracts §8.2).

- `build_headers` (§5.3.1), `validate_connected` (§8.2.3), `map_server_message` (§8.2.4),
  `bind_transcript` (U-PRS-30) are pure.
- `PrismaStreamClient.open` (§5.3.2): breaker check, **ledger reservation before any network
  I/O** (`BudgetExceeded` propagates), connect + `connected` within `connect_timeout_s`, up to 2
  retries with full jitter on RATE_LIMITED / CONNECT_TIMEOUT.
- `PrismaStreamSession` (§5.3.3-§5.3.5): one FIFO pacer queue for audio and commit (commit is
  never written before the last queued frame), 56 ms pacing, send timeouts, gated-zero drop
  under backlog, a reader that binds transcripts to frame tags, idempotent close.
- Ledger (§5.3.7): the reservation is topped up by `STREAM_RESERVE_S` whenever the billed
  seconds `max(frames_written x 0.064, now - opened)` reach it (checked on every written item
  and at close); `BudgetExceeded` on a top-up ends the stream (`SttClosed(reason="budget")`).
  At close the actual billed seconds are committed.

Permits: the caller holds the KeySemaphore 'stream' permit (a voice slot IS that permit, R-14;
contracts §5.2 `STTClient.open`). Breaker counting of mid-session closes is the orchestrator's
(§5.3.6); `open()` counts its own failures.
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import json
import random
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from sevi.adapters.deps import BillingLedger, BreakerSource, Purpose, Reservation, secret
from sevi.contracts.adapters import (
    Clock,
    FrameTag,
    SttClosed,
    SttCommitSent,
    SttConnected,
    SttErrorMsg,
    SttEvent,
    SttProcessing,
    SttSessionConfig,
    SttSpeechStart,
    SttTranscript,
    Telemetry,
)
from sevi.contracts.constants import PRISMA_FRAME_BYTES
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.contracts.errors import BreakerOpenError, BudgetExceeded, ErrorCode, SttError
from sevi.contracts.telemetry import EventName, SpanName

PRISMA_CHUNK_SAMPLES = 512  # documented: 1,024-byte frames at every rate
WS_URL = "wss://api.vachana.ai/stt/v3/stream"
PACE_MS = 56.0
BACKLOG_WARN_FRAMES = 4
GATED_DROP_BACKLOG_FRAMES = 8
HEADER_BUDGET = 7000
EVENTS_MAX = 256
STREAM_RESERVE_S = 60.0
TOP_UP_MARGIN_S = 5.0  # top up while this much reserved time is still left
DRAIN_MS = 500
CLOSE_TIMEOUT_S = 1.5
KEYTERM_HEADER_NAMES = ("keyterms", "x-keyterms", "x-keywords")
CLOSE_CODE_ERRORS = {1001: ErrorCode.STT_SESSION_LIMIT, 1008: ErrorCode.STT_HANDSHAKE_REJECTED}


@dataclass(frozen=True)
class PrismaOptions:
    ws_url: str = WS_URL
    header_mode: Literal["both", "doc", "sdk"] = "both"
    keyterms_enabled: bool = True
    keyterms_headers: tuple[str, ...] = KEYTERM_HEADER_NAMES
    send_enable_itn: bool = True
    connect_timeout_s: float = 3.0
    send_timeout_s: float = 5.0

    @classmethod
    def from_settings(cls, s: Any) -> PrismaOptions:
        """`s` = the root `Settings` (uses `s.prisma` and `s.session.send_timeout_s`)."""
        p = s.prisma
        return cls(
            ws_url=p.ws_url,
            header_mode=p.header_mode,
            keyterms_enabled=p.keyterms_enabled,
            keyterms_headers=tuple(p.keyterms_headers),
            send_enable_itn=p.send_enable_itn,
            connect_timeout_s=p.connect_timeout_s,
            send_timeout_s=s.session.send_timeout_s,
        )


def prisma_model(mv: ModelVersion) -> str:
    return "prisma-v3.0" if mv is ModelVersion.V3 else "prisma-v2.5"


def prisma_language_code(lang: Lang, mv: ModelVersion) -> str:
    """C§4.13: ta v3 "ta-IN,en-IN", ta v2.5 "ta-IN", en (both) "en-IN"."""
    if lang is Lang.EN:
        return "en-IN"
    return f"{lang.value}-IN,en-IN" if mv is ModelVersion.V3 else f"{lang.value}-IN"


# ---------------------------------------------------------------- headers (§5.3.1)
def _size(h: Sequence[tuple[str, str]]) -> int:
    return sum(len(n) + len(v) + 4 for n, v in h)


def build_headers(cfg: SttSessionConfig, o: PrismaOptions, api_key: str) -> list[tuple[str, str]]:
    v3 = cfg.model_version is ModelVersion.V3
    doc = o.header_mode in ("doc", "both")
    sdk = o.header_mode in ("sdk", "both")
    lang = cfg.lang.value
    head: list[tuple[str, str]] = [("x-api-key-id", api_key)]
    if doc:
        head.append(("language_code", prisma_language_code(cfg.lang, cfg.model_version)))
    if sdk:
        head.append(("lang_code", f"{lang}-IN"))
    m = prisma_model(cfg.model_version)
    if doc:
        head.append(("model", m))
    if sdk:
        head.append(("x-model", m))
    f = cfg.format if v3 else "transcribe"
    if doc:
        head.append(("format", f))
    if sdk:
        head.append(("x-format", f))
        if o.send_enable_itn and f == "transcribe":
            head.append(("x-enable-itn", "true"))
    kt_names: list[str] = []
    terms = list(cfg.keyterms)
    if v3 and o.keyterms_enabled and terms:
        allowed = ({"keyterms", "x-keyterms"} if doc else set()) | (
            {"x-keywords"} if sdk else set()
        )
        kt_names = [n for n in o.keyterms_headers if n in allowed]
    elif not v3:
        if o.keyterms_enabled and doc and terms:
            ascii_terms = [k for k in terms if k.isascii() and k.isalpha()][:100]
            head += [("bias_list", ",".join(ascii_terms)), ("bias_score", "1.5")]
        if doc:
            head.append(("itn_native_numerals", "false"))
    tail = [
        ("x-sample-rate", str(cfg.sample_rate_hz)),
        ("x-vad-threshold", f"{cfg.vad.threshold:g}"),
        ("x-min-silence-ms", str(cfg.vad.min_silence_ms)),
        ("x-min-speech-ms", str(cfg.vad.min_speech_ms)),
        ("x-min-segment-ms", str(cfg.vad.min_segment_ms)),
        ("x-idle-timeout-sec", str(cfg.idle_timeout_s)),
        ("x-api-request-id", cfg.request_id),
    ]

    def kt(names: list[str], ts: list[str]) -> list[tuple[str, str]]:
        val = json.dumps(ts, ensure_ascii=True, separators=(",", ":"))
        return [(n, val) for n in names]

    if kt_names:
        # budget: drop keyterm headers from the end (keep the first), then truncate the list
        while len(kt_names) > 1 and _size(head + kt(kt_names, terms) + tail) > HEADER_BUDGET:
            kt_names.pop()
        if _size(head + kt(kt_names, terms) + tail) > HEADER_BUDGET:
            lo, hi = 0, len(terms)
            while lo < hi:  # longest prefix that fits
                mid = (lo + hi + 1) // 2
                if _size(head + kt(kt_names, terms[:mid]) + tail) <= HEADER_BUDGET:
                    lo = mid
                else:
                    hi = mid - 1
            terms = terms[:lo]
        head += kt(kt_names, terms)
    return head + tail


# ---------------------------------------------------------------- connected (§8.2.3)
def frame_ms(sample_rate_hz: int) -> float:
    """Duration of one 1,024-byte (512-sample s16le) frame: 64 ms at 8 kHz, 32 ms at 16 kHz."""
    return PRISMA_CHUNK_SAMPLES * 1000.0 / sample_rate_hz


def validate_connected(
    cfg: SttSessionConfig, config: Mapping[str, Any] | None
) -> Literal["ok", "unverified"]:
    """C§8.2.3 echo check, narrowed to what Prisma documents (docs.gnani.ai stt-websocket):

    - v2.5 echoes only ``sample_rate`` and ``chunk_size``; only ``sample_rate`` is checked.
    - v3.0 also echoes ``format``, ``language_code`` and ``language_candidates``; all checked.
    ``chunk_size``, when present, must be 512.
    """
    if config is None:
        return "unverified"
    v3 = cfg.model_version is ModelVersion.V3
    problems = []
    if config.get("sample_rate") != cfg.sample_rate_hz:
        problems.append("sample_rate")
    if "chunk_size" in config and config.get("chunk_size") != PRISMA_CHUNK_SAMPLES:
        problems.append("chunk_size")
    if v3:
        if config.get("format") != cfg.format:
            problems.append("format")
        lc = str(config.get("language_code") or "")
        tag = f"{cfg.lang.value}-IN"
        cands = config.get("language_candidates") or []
        if tag not in lc and cfg.lang.value not in cands:
            problems.append("language")
    if problems:
        raise SttError(
            ErrorCode.STT_CONFIG_MISMATCH,
            f"config echo mismatch: {','.join(problems)}",
            details={"fields": ",".join(problems)},
        )
    return "ok"


# ---------------------------------------------------------------- mapping (§8.2.4)
def _int_or_none(v: object) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return int(v)


def map_server_message(
    raw: str, received_mono_ms: float, telemetry: Telemetry | None = None
) -> SttEvent | None:
    """None = ignored (unknown type, garbage, or a transcript without `text` -> STT_SCHEMA)."""
    try:
        msg = json.loads(raw)
    except ValueError:
        msg = None
    if not isinstance(msg, dict):
        if telemetry:
            telemetry.emit(EventName.WIRE_UNKNOWN_MESSAGE, "DEBUG", dep="prisma", kind="garbage")
        return None
    t = msg.get("type")
    if t == "connected":
        cfg = msg.get("config")
        return SttConnected(
            request_id=msg.get("request_id"), config_echo=cfg if isinstance(cfg, dict) else {}
        )
    if t == "speech_start":
        uid = msg.get("utterance_id")
        return SttSpeechStart(utterance_id=str(uid) if uid is not None else None)
    if t == "processing":
        return SttProcessing()
    if t == "transcript":
        text = msg.get("text")
        if not isinstance(text, str):
            if telemetry:
                telemetry.emit(EventName.STT_ERROR, "WARNING", code=ErrorCode.STT_SCHEMA.value)
            return None
        sid = msg.get("segment_id")
        dl = msg.get("detected_language_code")
        return SttTranscript(
            text=text,
            segment_id=str(sid) if sid is not None else None,
            segment_index=_int_or_none(msg.get("segment_index")),
            audio_duration_ms=_int_or_none(msg.get("audio_duration_ms")),
            latency_ms=_int_or_none(msg.get("latency")),
            detected_language_code=str(dl) if dl is not None else None,
            received_mono_ms=received_mono_ms,
        )
    if t == "error":
        return SttErrorMsg(message=str(msg.get("message", ""))[:300])
    if telemetry:
        telemetry.emit(EventName.WIRE_UNKNOWN_MESSAGE, "DEBUG", dep="prisma", kind=str(t))
    return None


# ---------------------------------------------------------------- binding (U-PRS-30)
@dataclass
class OffsetMap:
    """Run-length map of the sent stream: a new run starts only when the tag changes."""

    starts: list[float] = field(default_factory=list)
    tags: list[FrameTag | None] = field(default_factory=list)

    def append(self, written_ms: float, tag: FrameTag | None) -> None:
        if not self.tags or self.tags[-1] != tag:
            self.starts.append(written_ms)
            self.tags.append(tag)

    def tag_at(self, t_ms: float) -> FrameTag | None:
        if t_ms < 0 or not self.starts:
            return None
        i = bisect.bisect_right(self.starts, t_ms) - 1
        return self.tags[i] if i >= 0 else None


def bind_transcript(
    ev: SttTranscript, offsets: OffsetMap, written_ms: float, min_silence_ms: int
) -> SttTranscript:
    lat = ev.latency_ms if ev.latency_ms is not None else 400
    dur = ev.audio_duration_ms if ev.audio_duration_ms is not None else 600
    end = written_ms - lat - min_silence_ms
    mid = end - dur / 2
    return ev.model_copy(update={"stream_mid_ms": mid, "bound": offsets.tag_at(mid)})


# ---------------------------------------------------------------- session
@dataclass
class _Audio:
    pcm: bytes
    tag: FrameTag | None
    gated_zero: bool


class _Commit:
    pass


class _EventBuffer:
    """Bounded: when full, the oldest non-transcript event is dropped (transcripts never)."""

    def __init__(self, maxsize: int = EVENTS_MAX) -> None:
        self._d: deque[SttEvent] = deque()
        self._maxsize = maxsize
        self._ready = asyncio.Event()
        self.dropped = 0

    def put_nowait(self, ev: SttEvent) -> None:
        if len(self._d) >= self._maxsize:
            for i, old in enumerate(self._d):
                if not isinstance(old, SttTranscript):
                    del self._d[i]
                    self.dropped += 1
                    break
        self._d.append(ev)
        self._ready.set()

    async def get(self) -> SttEvent:
        while not self._d:
            self._ready.clear()
            await self._ready.wait()
        return self._d.popleft()


class PrismaStreamSession:
    """Implements contracts §5.2 `STTSession`."""

    def __init__(
        self,
        ws: Any,
        cfg: SttSessionConfig,
        *,
        request_id: str,
        config_echo: dict[str, Any],
        clock: Clock,
        telemetry: Telemetry,
        send_timeout_s: float = 5.0,
        ledger: BillingLedger | None = None,
        reservation: Reservation | None = None,
        reserved_s: float = 0.0,
        opened_mono_ms: float | None = None,
    ) -> None:
        self._ws = ws
        self.cfg = cfg
        self._request_id = request_id
        self._config_echo = config_echo
        self.clock = clock
        self.telemetry = telemetry
        self.send_timeout_s = send_timeout_s
        self._ledger = ledger
        self._reservation = reservation
        self._reserved_s = reserved_s
        self._committed = False
        self.opened_mono_ms = clock.mono_ms() if opened_mono_ms is None else opened_mono_ms
        self._q: asyncio.Queue[_Audio | _Commit] = asyncio.Queue()
        self._events = _EventBuffer()
        self.offsets = OffsetMap()
        self.frames_written = 0
        self.written_ms = 0.0
        self.dropped_gated = 0
        self._last_send = float("-inf")
        self._last_backlog_warn = float("-inf")
        self._last_drop_warn = float("-inf")
        self._accepting = True
        self._closing = False
        self._closed_emitted = False
        self._writer = asyncio.create_task(self._write_loop())
        self._reader = asyncio.create_task(self._read_loop())

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def config_echo(self) -> dict[str, Any]:
        return self._config_echo

    @property
    def backlog(self) -> int:
        return self._q.qsize()

    def billed_s(self) -> float:
        wall = (self.clock.mono_ms() - self.opened_mono_ms) / 1000.0
        return max(self.frames_written * frame_ms(self.cfg.sample_rate_hz) / 1000.0, wall)

    # ---------- producer side (synchronous, never blocks) ----------
    def send_frame(self, pcm: bytes, tag: FrameTag | None, *, gated_zero: bool = False) -> None:
        if len(pcm) != PRISMA_FRAME_BYTES:
            raise ValueError(f"frame must be {PRISMA_FRAME_BYTES} bytes, got {len(pcm)}")
        if not self._accepting:
            raise SttError(ErrorCode.STT_DISCONNECTED, "stream closed")
        self._q.put_nowait(_Audio(pcm, tag, gated_zero))
        n = self._q.qsize()
        now = self.clock.mono_ms()
        if n > BACKLOG_WARN_FRAMES and now - self._last_backlog_warn >= 1000.0:
            self._last_backlog_warn = now
            self.telemetry.emit(EventName.UPLINK_BACKLOG, "WARNING", frames=n)

    def commit(self) -> None:
        if not self._accepting:
            raise SttError(ErrorCode.STT_DISCONNECTED, "stream closed")
        self._q.put_nowait(_Commit())

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self._events.get()
            yield ev
            if isinstance(ev, SttClosed):
                return

    # ---------- writer (pacer) ----------
    def _emit_closed(self, code: int | None, reason: str | None) -> None:
        if self._closed_emitted:
            return
        self._closed_emitted = True
        self._accepting = False
        self._events.put_nowait(SttClosed(code=code, reason=reason))
        expected = self._closing and code in (1000, None)
        err = None if expected else CLOSE_CODE_ERRORS.get(code or 0, ErrorCode.STT_DISCONNECTED)
        self.telemetry.emit(
            EventName.STT_CLOSED,
            "INFO" if expected else "WARNING",
            code=code,
            reason=reason,
            error=err.value if err else None,
            request_id=self._request_id,
        )

    async def _send(self, data: bytes | str) -> None:
        async with asyncio.timeout(self.send_timeout_s):
            await self._ws.send(data)

    async def _write_loop(self) -> None:
        try:
            while True:
                item = await self._q.get()
                if isinstance(item, _Audio):
                    if item.gated_zero and self._q.qsize() + 1 > GATED_DROP_BACKLOG_FRAMES:
                        self.dropped_gated += 1
                        now = self.clock.mono_ms()
                        if now - self._last_drop_warn >= 1000.0:
                            self._last_drop_warn = now
                            self.telemetry.emit(
                                EventName.UPLINK_DROP_GATED, "WARNING", frames=self.dropped_gated
                            )
                        self._q.task_done()
                        continue
                    await self.clock.sleep_until(
                        max(self.clock.mono_ms(), self._last_send + PACE_MS)
                    )
                    await self._top_up_if_due(frame_ms(self.cfg.sample_rate_hz) / 1000.0)
                    await self._send(item.pcm)
                    self.offsets.append(self.written_ms, item.tag)
                    self.written_ms += frame_ms(self.cfg.sample_rate_hz)
                    self.frames_written += 1
                    self._last_send = self.clock.mono_ms()
                else:
                    await self._send('{"type":"commit"}')
                    t = self.clock.mono_ms()
                    self._events.put_nowait(SttCommitSent(t_mono_ms=t))
                    self.telemetry.emit(EventName.STT_COMMIT, "DEBUG", barrier_wait=True)
                self._q.task_done()
        except TimeoutError:
            self.telemetry.emit(EventName.WS_SEND_TIMEOUT, "WARNING", dep="prisma")
            self._emit_closed(None, "send_timeout")
        except ConnectionClosed as e:
            rc = e.rcvd
            self._emit_closed(rc.code if rc else None, rc.reason if rc else "closed")
        except _BudgetStop:
            self._emit_closed(None, "budget")
            with contextlib.suppress(Exception):
                await self._ws.close(1000, "budget")

    async def _top_up_if_due(self, next_s: float = 0.0) -> None:
        """Called before every frame: top up once the billed seconds plus the next frame come
        within TOP_UP_MARGIN_S of the reservation, so no audio is ever sent unreserved."""
        if self._ledger is None or self._reservation is None or self._committed:
            return
        if self.billed_s() + next_s < self._reserved_s - TOP_UP_MARGIN_S:
            return
        try:
            await self._ledger.top_up(self._reservation, stt_s=STREAM_RESERVE_S)
        except BudgetExceeded:
            raise _BudgetStop from None
        self._reserved_s += STREAM_RESERVE_S

    # ---------- reader ----------
    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes | bytearray):
                    continue
                ev = map_server_message(raw, self.clock.mono_ms(), self.telemetry)
                if ev is None:
                    continue
                if isinstance(ev, SttTranscript):
                    ev = bind_transcript(
                        ev, self.offsets, self.written_ms, self.cfg.vad.min_silence_ms
                    )
                    self.telemetry.emit(
                        EventName.STT_TRANSCRIPT,
                        "DEBUG",
                        segment_index=ev.segment_index,
                        latency_ms=ev.latency_ms,
                        bound=ev.bound is not None,
                    )
                elif isinstance(ev, SttErrorMsg):
                    self.telemetry.emit(EventName.STT_ERROR, "WARNING", message=ev.message)
                self._events.put_nowait(ev)
            code = getattr(self._ws, "close_code", None)
            reason = getattr(self._ws, "close_reason", None)
            self._emit_closed(code, reason)
        except ConnectionClosed as e:
            rc = e.rcvd
            self._emit_closed(rc.code if rc else None, rc.reason if rc else "closed")

    # ---------- close (§5.3.5) ----------
    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._accepting = False
        with contextlib.suppress(Exception):
            async with asyncio.timeout(CLOSE_TIMEOUT_S):
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(DRAIN_MS / 1000.0):
                        while not self._q.empty() and not self._writer.done():
                            await asyncio.sleep(0.005)
                await self._ws.close(1000)
        for t in (self._writer, self._reader):
            t.cancel()
        for t in (self._writer, self._reader):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._emit_closed(1000, "client close")
        await self.finalize_ledger()

    async def finalize_ledger(self) -> None:
        """Commit the billed connection seconds once (also safe if never reserved)."""
        if self._committed or self._ledger is None or self._reservation is None:
            return
        self._committed = True
        await self._ledger.commit(self._reservation, stt_s=round(self.billed_s(), 3))


class _BudgetStop(Exception):
    pass


# ---------------------------------------------------------------- client
class PrismaStreamClient:
    """Implements contracts §5.2 `STTClient`."""

    def __init__(
        self,
        opts: PrismaOptions,
        keys: Mapping[KeyId, object],
        breakers: BreakerSource,
        ledger: BillingLedger,
        clock: Clock,
        telemetry: Telemetry,
        *,
        purpose: Callable[[KeyId], Purpose] | Purpose | None = None,
        connect: Callable[..., Any] = ws_connect,
        rng: random.Random | None = None,
    ) -> None:
        self.opts = opts
        self.keys = keys
        self.breakers = breakers
        self.ledger = ledger
        self.clock = clock
        self.telemetry = telemetry
        self._purpose = purpose
        self._connect = connect
        self.rng = rng or random.Random()  # noqa: S311 (backoff jitter)

    def purpose_for(self, key_id: KeyId) -> Purpose:
        """Default: Key B (public voice) -> public_voice; Key A (VT / interview) -> vt."""
        if self._purpose is None:
            return "public_voice" if key_id is KeyId.B else "vt"
        return self._purpose(key_id) if callable(self._purpose) else self._purpose

    async def open(self, cfg: SttSessionConfig) -> PrismaStreamSession:
        breaker = self.breakers.get("prisma", cfg.key_id)
        if breaker.is_open():
            raise BreakerOpenError(ErrorCode.STT_BREAKER_OPEN, "prisma breaker open")
        headers = build_headers(cfg, self.opts, secret(self.keys[cfg.key_id]))
        kt = [n for n, _ in headers if n in KEYTERM_HEADER_NAMES]
        self.telemetry.emit(
            EventName.STT_OPEN,
            "INFO",
            header_mode=self.opts.header_mode,
            keyterm_headers=kt,
            n_keyterms=len(json.loads(dict(headers)[kt[0]])) if kt else 0,
            request_id=cfg.request_id,
        )
        # credit safety: reserve BEFORE any network I/O; BudgetExceeded propagates untouched
        reservation = await self.ledger.reserve(
            cfg.key_id, self.purpose_for(cfg.key_id), stt_s=STREAM_RESERVE_S
        )
        socket_s = 0.0  # seconds a socket was open during failed attempts (conservative bill)
        attempt = 0
        loop = asyncio.get_running_loop()
        while True:
            code: ErrorCode
            retryable = False
            ws: Any = None
            t_open = 0.0
            try:
                with self.telemetry.span(SpanName.PRISMA_OPEN, attempt=attempt + 1):
                    t0 = loop.time()
                    async with asyncio.timeout(self.opts.connect_timeout_s):
                        ws = await self._connect(
                            self.opts.ws_url,
                            additional_headers=headers,
                            open_timeout=self.opts.connect_timeout_s,
                            max_size=2**20,
                            ping_interval=20,
                            ping_timeout=20,
                            close_timeout=1,
                        )
                        t_open = loop.time()
                        first = await ws.recv()
                    ev = map_server_message(first, self.clock.mono_ms(), self.telemetry)
                    if isinstance(ev, SttErrorMsg):
                        with contextlib.suppress(Exception):
                            async with asyncio.timeout(0.5):
                                await ws.wait_closed()
                        if getattr(ws, "close_code", None) == 1008:
                            breaker.force_open()
                            raise _Fail(ErrorCode.STT_HANDSHAKE_REJECTED, False, counted=True)
                        if "limit" in ev.message.lower():
                            raise _Fail(ErrorCode.STT_RATE_LIMITED, True)
                        raise _Fail(ErrorCode.STT_HANDSHAKE_REJECTED, False)
                    if not isinstance(ev, SttConnected):
                        raise _Fail(ErrorCode.STT_SCHEMA, False)
                    echo_raw = json.loads(first).get("config")
                    try:
                        verdict = validate_connected(
                            cfg, echo_raw if isinstance(echo_raw, dict) else None
                        )
                    except SttError:
                        self.telemetry.emit(EventName.STT_CONFIG_MISMATCH, "WARNING", echo=echo_raw)
                        await self._abandon(ws)
                        socket_s += loop.time() - t_open
                        await self.ledger.commit(reservation, stt_s=round(socket_s, 3))
                        raise
                    if verdict == "unverified":
                        self.telemetry.emit(EventName.STT_CONFIG_UNVERIFIED, "WARNING")
                    self.telemetry.emit(
                        EventName.STT_CONNECTED,
                        "INFO",
                        request_id=ev.request_id,
                        dur_ms=(loop.time() - t0) * 1000.0,
                    )
                    opened = self.clock.mono_ms() - socket_s * 1000.0
                    return PrismaStreamSession(
                        ws,
                        cfg,
                        request_id=ev.request_id or cfg.request_id,
                        config_echo=ev.config_echo,
                        clock=self.clock,
                        telemetry=self.telemetry,
                        send_timeout_s=self.opts.send_timeout_s,
                        ledger=self.ledger,
                        reservation=reservation,
                        reserved_s=STREAM_RESERVE_S,
                        opened_mono_ms=opened,
                    )
            except InvalidStatus as e:
                st = e.response.status_code
                if st in (401, 403):
                    breaker.force_open()
                    await self.ledger.release(reservation)
                    raise SttError(
                        ErrorCode.STT_AUTH, f"HTTP {st}", details={"status": st}
                    ) from None
                if st == 402:
                    breaker.force_open()
                    await self.ledger.mark_exhausted(cfg.key_id)
                    await self.ledger.release(reservation)
                    raise SttError(
                        ErrorCode.STT_CREDITS, "HTTP 402", details={"status": st}
                    ) from None
                code, retryable = (
                    (ErrorCode.STT_RATE_LIMITED, True)
                    if st == 429
                    else (ErrorCode.STT_CONNECT_TIMEOUT, True)
                )
            except _Fail as f:
                await self._abandon(ws)
                socket_s += loop.time() - t_open if t_open else 0.0
                if not f.retryable:
                    await self._settle(reservation, socket_s)
                    if not f.counted:
                        breaker.record_failure()
                    raise SttError(f.code, f"prisma open: {f.code.value}") from None
                code, retryable = f.code, True
            except (TimeoutError, OSError, ConnectionClosed):
                await self._abandon(ws)
                socket_s += loop.time() - t_open if t_open else 0.0
                code, retryable = ErrorCode.STT_CONNECT_TIMEOUT, True
            if retryable and attempt < 2:
                delay = self.rng.uniform(0, min(2.0, 0.5 * 2**attempt))
                await self.clock.sleep_until(self.clock.mono_ms() + delay * 1000.0)
                attempt += 1
                continue
            breaker.record_failure()
            await self._settle(reservation, socket_s)
            raise SttError(code, f"prisma open failed after {attempt + 1} attempts")

    async def _settle(self, r: Reservation, socket_s: float) -> None:
        if socket_s > 0:
            await self.ledger.commit(r, stt_s=round(socket_s, 3))
        else:
            await self.ledger.release(r)

    @staticmethod
    async def _abandon(ws: Any) -> None:
        if ws is None:
            return
        with contextlib.suppress(Exception):
            async with asyncio.timeout(1.0):
                await ws.close(1000)


class _Fail(Exception):
    def __init__(self, code: ErrorCode, retryable: bool, *, counted: bool = False) -> None:
        super().__init__(code.value)
        self.code = code
        self.retryable = retryable
        self.counted = counted
