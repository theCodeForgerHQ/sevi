"""Fake Prisma streaming STT server (contracts §11.8, B spec §12.1).

A real `websockets` server on 127.0.0.1 speaking the §8.2 wire format, so the adapter is tested
over a socket with zero Gnani credits.

Scripted mode: a frame is *voiced* when its RMS is above -40 dBFS. Rule k (k-th utterance)
fires after >= `after_voiced_frames` voiced frames followed by >= `then_silence_ms` of silent
frames: `processing` (if `processing_first`), then after `latency_ms` a final `transcript`.
A `commit` while voiced frames are pending emits (`processing` iff `commit_emits_processing`)
+ the transcript of the current rule; otherwise a commit emits nothing.

Knobs (C§11.8): `delay_ms` (int, or {utterance_index: ms}), `drop_after_frames` + `close_code`,
`http_status` (reject the upgrade), `garbage_json`, `never_final`, `error_after_commit`,
`commit_emits_processing`, `split_segments`, `config_echo` (override), `omit_config`,
`connected_delay_ms`, `empty_text`, `text_source`, `drop_processing`, `stall_writes` (stop
reading the socket after `connected`, so the client's writes eventually block),
`handshake_error` (send an `error` message instead of `connected`, then close with
`close_code`, default 1008) and `replay` (messages sent verbatim after `connected`; recorded
fixture replay, C-PRS-01). `clock`: the transcript latency and `connected_delay_ms` wait on
that `Clock` (the app's in demo mode, so a scaled session clock scales them too); default
real time.

Each connection records its lower-cased request headers, every binary frame with its stream
offset (`(offset_ms, frame)`, 64 ms per 1,024-byte frame), commit offsets and sent messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from sevi.contracts.adapters import Clock
from sevi.contracts.constants import PRISMA_FRAME_BYTES

VOICED_DBFS = -40.0


@dataclass(frozen=True)
class Rule:
    text: str
    after_voiced_frames: int = 8
    then_silence_ms: int = 300
    processing_first: bool = True
    latency_ms: int = 300


@dataclass
class FakePrismaConnection:
    headers: dict[str, str]
    header_list: list[tuple[str, str]]
    request_id: str
    frames: list[tuple[float, bytes]] = field(default_factory=list)
    commits: list[float] = field(default_factory=list)
    sent: list[dict[str, Any]] = field(default_factory=list)
    closed_code: int | None = None

    @property
    def frame_ms(self) -> float:
        """512 samples per 1,024-byte frame at the negotiated x-sample-rate (docs)."""
        return 512 * 1000.0 / int(self.headers.get("x-sample-rate", "16000"))

    @property
    def stream_ms(self) -> float:
        return len(self.frames) * self.frame_ms


def frame_dbfs(frame: bytes) -> float:
    x = np.frombuffer(frame[: len(frame) - len(frame) % 2], dtype="<i2").astype(np.float64)
    if x.size == 0:
        return -200.0
    r = math.sqrt(float(np.mean(x * x))) / 32768.0
    return 20.0 * math.log10(r) if r > 0 else -200.0


def voiced_frame(amplitude: int = 8000, n_bytes: int = PRISMA_FRAME_BYTES) -> bytes:
    """A 400 Hz tone frame (well above -40 dBFS) for tests."""
    n = n_bytes // 2
    t = np.arange(n) / 8000.0
    return (amplitude * np.sin(2 * np.pi * 400.0 * t)).astype("<i2").tobytes()


def silent_frame(n_bytes: int = PRISMA_FRAME_BYTES) -> bytes:
    return bytes(n_bytes)


class FakePrisma:
    def __init__(
        self,
        script: Sequence[Rule] = (),
        *,
        delay_ms: int | Mapping[int, int] = 0,
        drop_after_frames: int | None = None,
        close_code: int | None = None,
        http_status: int | None = None,
        garbage_json: bool = False,
        never_final: bool = False,
        error_after_commit: bool = False,
        commit_emits_processing: bool = True,
        split_segments: bool = False,
        config_echo: dict[str, Any] | None = None,
        omit_config: bool = False,
        connected_delay_ms: int = 0,
        empty_text: bool = False,
        text_source: Callable[[int], str] | None = None,
        drop_processing: bool = False,
        stall_writes: bool = False,
        handshake_error: str | None = None,
        replay: Sequence[dict[str, Any]] = (),
        clock: Clock | None = None,
    ) -> None:
        self.script = list(script)
        self.delay_ms = delay_ms
        self.drop_after_frames = drop_after_frames
        self.close_code = close_code
        self.http_status = http_status
        self.garbage_json = garbage_json
        self.never_final = never_final
        self.error_after_commit = error_after_commit
        self.commit_emits_processing = commit_emits_processing
        self.split_segments = split_segments
        self.config_echo = config_echo
        self.omit_config = omit_config
        self.connected_delay_ms = connected_delay_ms
        self.empty_text = empty_text
        self.text_source = text_source
        self.drop_processing = drop_processing
        self.stall_writes = stall_writes
        self.handshake_error = handshake_error
        self.replay = list(replay)
        # Transcript latency runs on this clock (the app's, in demo mode) so a scaled or fake
        # session clock also scales it; None = real time (adapter tests).
        self.clock = clock
        self.connections: list[FakePrismaConnection] = []
        self.rejected: list[dict[str, str]] = []  # headers of upgrades rejected by http_status
        self._server: Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    # ---------- lifecycle ----------
    async def start(self) -> str:
        self._server = await serve(
            self._handler, "127.0.0.1", 0, process_request=self._process_request
        )
        port = next(iter(self._server.sockets)).getsockname()[1]
        return f"ws://127.0.0.1:{port}/stt/v3/stream"

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> FakePrisma:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ---------- protocol ----------
    def _process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        if self.http_status is not None:
            self.rejected.append({k.lower(): v for k, v in request.headers.raw_items()})
            return conn.respond(self.http_status, f"fake prisma: HTTP {self.http_status}\n")
        return None

    def _echo(self, h: Mapping[str, str]) -> dict[str, Any]:
        """Documented `connected.config`: v2.5 has only sample_rate + chunk_size; v3.0 adds
        language_code, format and language_candidates."""
        echo: dict[str, Any] = {
            "sample_rate": int(h.get("x-sample-rate", "16000")),
            "chunk_size": 512,
        }
        if (h.get("model") or h.get("x-model") or "prisma-v2.5") == "prisma-v3.0":
            lc = h.get("language_code") or h.get("lang_code") or "ta-IN"
            echo["language_code"] = lc
            echo["format"] = h.get("format") or h.get("x-format") or "verbatim"
            echo["language_candidates"] = [c.split("-")[0] for c in lc.split(",")]
        if self.config_echo is not None:
            echo.update(self.config_echo)
        return echo

    async def _send(
        self, ws: ServerConnection, c: FakePrismaConnection, msg: dict[str, Any]
    ) -> None:
        c.sent.append(msg)
        await ws.send(json.dumps(msg, ensure_ascii=False))

    def _spawn(self, coro: Any) -> None:
        t: asyncio.Task[None] = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _rule(self, k: int) -> Rule:
        if self.text_source is not None:
            base = self.script[min(k, len(self.script) - 1)] if self.script else Rule("")
            return Rule(
                self.text_source(k),
                base.after_voiced_frames,
                base.then_silence_ms,
                base.processing_first,
                base.latency_ms,
            )
        if not self.script:
            return Rule("")
        return self.script[min(k, len(self.script) - 1)]

    async def _wait_ms(self, ms: float) -> None:
        if self.clock is None:
            await asyncio.sleep(ms / 1000.0)
        else:
            await self.clock.sleep_until(self.clock.mono_ms() + ms)

    def _delay(self, k: int) -> int:
        if isinstance(self.delay_ms, Mapping):
            return int(self.delay_ms.get(k, 0))
        return int(self.delay_ms)

    async def _handler(self, ws: ServerConnection) -> None:
        req = ws.request
        assert req is not None
        pairs = [(k.lower(), v) for k, v in req.headers.raw_items()]
        h = dict(pairs)
        rid = str(uuid.uuid4())
        c = FakePrismaConnection(headers=h, header_list=pairs, request_id=rid)
        self.connections.append(c)
        state = _UttState()
        try:
            if self.connected_delay_ms:
                await self._wait_ms(self.connected_delay_ms)
            if self.handshake_error is not None:
                await self._send(ws, c, {"type": "error", "message": self.handshake_error})
                await ws.close(self.close_code or 1008, "rejected")
                return
            connected: dict[str, Any] = {
                "type": "connected",
                "message": "STT service ready",
                "request_id": rid,
                "timestamp": "2026-11-04T05:10:00.123Z",
            }
            if not self.omit_config:
                connected["config"] = self._echo(h)
            await self._send(ws, c, connected)
            if self.garbage_json:
                c.sent.append({"raw": "{not json"})
                await ws.send("{not json")
                await self._send(ws, c, {"type": "vad_end"})
            for msg in self.replay:
                await self._send(ws, c, msg)
            if self.stall_writes:
                await ws.wait_closed()
                return
            async for raw in ws:
                if isinstance(raw, bytes):
                    await self._on_frame(ws, c, state, raw)
                    if (
                        self.drop_after_frames is not None
                        and len(c.frames) >= self.drop_after_frames
                    ):
                        await ws.close(self.close_code or 1011, "fake drop")
                        return
                else:
                    await self._on_text(ws, c, state, raw)
        except ConnectionClosed:
            pass
        finally:
            c.closed_code = ws.close_code

    async def _on_frame(
        self, ws: ServerConnection, c: FakePrismaConnection, st: _UttState, frame: bytes
    ) -> None:
        c.frames.append((c.stream_ms, frame))
        if frame_dbfs(frame) > VOICED_DBFS:
            if st.voiced == 0:
                await self._send(ws, c, {"type": "speech_start", "utterance_id": f"u-{st.k}"})
            st.voiced += 1
            st.silence_ms = 0.0
            return
        if st.voiced == 0:
            return
        st.silence_ms += c.frame_ms
        rule = self._rule(st.k)
        if st.voiced >= rule.after_voiced_frames and st.silence_ms >= rule.then_silence_ms:
            k, dur = st.k, st.voiced * c.frame_ms
            st.k += 1
            st.voiced = 0
            st.silence_ms = 0.0
            text = "" if self.empty_text else rule.text
            if self.split_segments:
                words = text.split()
                half = max(1, len(words) // 2) if words else 0
                text, st.pending_tail = " ".join(words[:half]), " ".join(words[half:])
                st.pending_k = k
            self._spawn(self._emit(ws, c, st, k, text, dur, rule, rule.processing_first))

    async def _on_text(
        self, ws: ServerConnection, c: FakePrismaConnection, st: _UttState, raw: str
    ) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if msg.get("type") != "commit":
            return
        c.commits.append(c.stream_ms)
        if self.error_after_commit:
            await self._send(ws, c, {"type": "error", "message": "STT engine failed"})
        if st.voiced > 0:
            k, dur = st.k, st.voiced * c.frame_ms
            st.k += 1
            st.voiced = 0
            st.silence_ms = 0.0
            rule = self._rule(k)
            text = "" if self.empty_text else rule.text
            self._spawn(self._emit(ws, c, st, k, text, dur, rule, self.commit_emits_processing))
        elif st.pending_tail is not None:
            tail, k = st.pending_tail, st.pending_k
            st.pending_tail = None
            rule = self._rule(k)
            self._spawn(self._emit(ws, c, st, k, tail, 600.0, rule, False))

    async def _emit(
        self,
        ws: ServerConnection,
        c: FakePrismaConnection,
        st: _UttState,
        k: int,
        text: str,
        dur_ms: float,
        rule: Rule,
        processing: bool,
    ) -> None:
        with contextlib.suppress(ConnectionClosed):
            if processing and not self.drop_processing:
                await self._send(ws, c, {"type": "processing"})
            await self._wait_ms(rule.latency_ms + self._delay(k))
            if self.never_final:
                return
            st.segment += 1
            await self._send(
                ws,
                c,
                {
                    "type": "transcript",
                    "text": text,
                    "audio_duration_ms": int(dur_ms),
                    "segment_id": f"{c.request_id}-seg-{st.segment}",
                    "segment_index": st.segment,
                    "latency": rule.latency_ms,
                    "is_final": True,
                },
            )


@dataclass
class _UttState:
    k: int = 0  # utterance index (rule index)
    voiced: int = 0  # voiced frames since the last emit
    silence_ms: float = 0.0  # silence after voice
    segment: int = 0
    pending_tail: str | None = None  # split_segments: text sent after the next commit
    pending_k: int = 0
