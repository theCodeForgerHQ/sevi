"""Whole-app E2E kit: the real FastAPI app on uvicorn (127.0.0.1, ephemeral port) and a scripted
browser speaking the `/web/stream` wire protocol over a real WebSocket (contracts §9.4, B spec
§6.2). No network beyond localhost, no keys: demo mode runs the in-process fakes (Prisma on a
local websockets server, Evon on an httpx mock transport) and the synthetic test bank.

The browser echoes every `mark` as soon as the clip arrives (instant playback), answers
consent, types the presented triplet on each open keypad window (read from the live session:
a test-only peek) and, in voice mode, streams μ-law "mic" audio: a tone during each answer
window, silence otherwise. The demo Prisma fake transcribes with the presented digits.

Time runs on `E2EClock` (real time x `speed`); its wall clock can also jump (72 h report
expiry) without waiting.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import socket
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from sevi.app import main as app_main
from sevi.app.main import create_app
from sevi.app.settings import Settings
from sevi.app.state import AppState
from sevi.core.dsp import ulaw_encode
from sevi.fakes.clock import ScaledClock

REPO = Path(__file__).resolve().parents[3]
TEST_BANK_DIR = REPO / "stimuli" / "test-bank"
PAGES = "http://127.0.0.1:8000"
SPEED = 10.0
CHUNK_MS = 200  # one media message = 200 ms of 8 kHz μ-law (1,600 bytes, the wire maximum)
SILENCE = b"\xff" * (CHUNK_MS * 8)


def _tone() -> bytes:
    t = np.arange(CHUNK_MS * 8) / 8000.0
    return ulaw_encode((8000 * np.sin(2 * np.pi * 400 * t)).astype(np.int16))


TONE = _tone()


class E2EClock(ScaledClock):
    """`ScaledClock` whose wall clock can also be jumped forward without waiting."""

    def __init__(self, speed: float = SPEED) -> None:
        super().__init__(speed)
        self._jump = timedelta(0)

    def jump(self, **delta: float) -> None:
        self._jump += timedelta(**delta)

    def now_utc(self) -> datetime:
        return super().now_utc() + self._jump


_BANKS: dict[tuple[str, str], Any] = {}
_real_load_banks = app_main.load_banks


def _cached_load_banks(s: Settings, telemetry: Any) -> Any:
    key = (str(s.result.bank_dir), s.env)
    if key not in _BANKS:
        _BANKS[key] = _real_load_banks(s, telemetry)
    return dict(_BANKS[key])


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def demo_settings(tmp: Path, **over: Any) -> Settings:
    """What `SEVI_DEMO_MODE=1 make serve` runs: demo env, fakes, the synthetic test bank."""
    base: dict[str, Any] = {
        "env": "demo",
        "demo_mode": True,
        "data_dir": tmp / "data",
        "log_dir": tmp / "logs",
        "log_level": "WARNING",
        "pages_origin": PAGES,
        "public_base_url": PAGES,
        "result": {
            "bank_dir": TEST_BANK_DIR,
            "enabled_langs": "ta",
            "experimental_langs": "en",
            "allow_unfrozen_bank": True,
        },
        "browser_voice": {"enabled": True},
        "admission": {"ip_hash_key": "k" * 32},
        "ledger": {"credits_per_stt_s": 1.0, "budget_credits_per_key": 100_000.0},
    }
    return Settings(_env_file=None, **_merge(base, over))  # type: ignore[call-arg]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@dataclass
class Server:
    app: FastAPI
    base: str
    clock: E2EClock

    @property
    def st(self) -> AppState:
        state: AppState = self.app.state.sevi
        return state

    def http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base, timeout=10.0)


@contextmanager
def serving(tmp: Path, *, clock: E2EClock | None = None, **over: Any) -> Iterator[Server]:
    """uvicorn on 127.0.0.1 in a thread (its own loop), lifespan on (real `startup()`)."""
    clock = clock or E2EClock()
    app_main.load_banks = _cached_load_banks
    app = create_app(demo_settings(tmp, **over), clock=clock)
    port = _free_port()
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, lifespan="on")
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        for _ in range(500):
            if server.started:
                break
            if not th.is_alive():
                raise RuntimeError("uvicorn failed to start")
            threading.Event().wait(0.02)
        else:
            raise RuntimeError("uvicorn did not start")
        yield Server(app, f"http://127.0.0.1:{port}", clock)
    finally:
        server.should_exit = True
        th.join(timeout=15)
        app_main.load_banks = _real_load_banks


def create_body(mode: str = "keypad", *, earphones: bool = True) -> dict[str, Any]:
    return {
        "lang": "ta",
        "mode": mode,
        "earphones": earphones,
        "client": {
            "browser_family": "Chrome",
            "os_family": "Android",
            "ua": "Mozilla/5.0 (Linux; Android 14) e2e",
            "audio_worklet": True,
            "sample_rate_hz": 48000,
            "page_version": "abcdef123456",
        },
    }


Answer = Callable[[tuple[int, ...] | None], str | None]
Hook = Callable[["Browser", dict[str, Any]], Any]


def correct(triplet: tuple[int, ...] | None) -> str | None:
    return None if triplet is None else "".join(str(d) for d in triplet)


@dataclass
class Transcript:
    ui: list[dict[str, Any]] = field(default_factory=list)
    marks: list[str] = field(default_factory=list)
    media_in: int = 0
    media_out: int = 0
    close_code: int | None = None

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [m for m in self.ui if m.get("type") == kind]

    def last(self, kind: str) -> dict[str, Any] | None:
        got = self.of(kind)
        return got[-1] if got else None


class Browser:
    def __init__(self, srv: Server, created: dict[str, Any]) -> None:
        self.srv = srv
        self.sid: str = created["session_id"]
        self.token: str = created["ws_token"]
        self.t = Transcript()
        self.ws: ClientConnection | None = None
        self._seq = 0
        self._mic: asyncio.Task[None] | None = None
        self.speak_until = 0.0  # E2EClock mono ms
        self.mic_muted = False

    # ----- wire -----
    async def _send(self, event: str, body: dict[str, Any] | None = None) -> None:
        assert self.ws is not None
        self._seq += 1
        msg: dict[str, Any] = {"event": event, "sequenceNumber": str(self._seq)}
        if body is not None:
            msg[event] = body
        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(json.dumps(msg))

    async def start(self) -> None:
        params = {"token": self.token, "session_id": self.sid}
        await self._send("start", {"streamSid": f"web-{self.sid}", "customParameters": params})

    async def ui(self, type_: str, **kw: Any) -> None:
        await self._send("kj", {"type": type_, **kw})

    async def dtmf(self, keys: str) -> None:
        for k in keys:
            await self._send("dtmf", {"digit": k})

    # ----- peeks (test only) -----
    @property
    def session(self) -> Any:
        h = self.srv.st.port.get(self.sid)
        return getattr(h, "sess", None)

    def presented(self) -> tuple[int, ...] | None:
        pres = getattr(self.session, "pres", None)
        return None if pres is None else tuple(pres.spec.triplet)

    # ----- mic -----
    async def _mic_loop(self) -> None:
        clock = self.srv.clock
        ts = int(clock.mono_ms())
        while True:
            now = clock.mono_ms()
            voiced = now < self.speak_until and not self.mic_muted
            payload = base64.b64encode(TONE if voiced else SILENCE).decode()
            await self._send("media", {"payload": payload, "timestamp": str(ts)})
            self.t.media_out += 1
            ts += CHUNK_MS
            await asyncio.sleep(CHUNK_MS / 1000.0 / clock.speed)

    def speak(self, ms: float = 1200.0, after_ms: float = 200.0) -> None:
        now = self.srv.clock.mono_ms()
        self.speak_until = now + after_ms + ms

    # ----- session -----
    @asynccontextmanager
    async def connect(self, origin: str = PAGES) -> AsyncIterator[Browser]:
        url = self.srv.base.replace("http://", "ws://") + f"/web/stream?sid={self.sid}"
        async with connect(url, origin=origin, max_size=2**20) as ws:  # type: ignore[arg-type]
            self.ws = ws
            try:
                yield self
            finally:
                if self._mic is not None:
                    self._mic.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._mic

    async def run(
        self,
        *,
        answer: Answer | None = None,
        on_ui: Hook | None = None,
        consent: bool = True,
        timeout_s: float = 60.0,
    ) -> Transcript:
        """Drive until the server closes the socket. An `on_ui` hook returning True replaces
        the default reaction to that message."""
        assert self.ws is not None
        await self.start()

        async def loop() -> None:
            assert self.ws is not None
            try:
                async for raw in self.ws:
                    msg = json.loads(raw)
                    ev = msg.get("event")
                    if ev == "mark":
                        self.t.marks.append(msg["mark"]["name"])
                        await self._send("mark", {"name": msg["mark"]["name"]})
                    elif ev == "media":
                        self.t.media_in += 1
                    elif ev == "kj":
                        await self._on_ui(msg["kj"], answer, on_ui, consent)
            except ConnectionClosed:
                pass
            self.t.close_code = self.ws.close_code

        await asyncio.wait_for(loop(), timeout_s)
        return self.t

    async def _on_ui(
        self, ui: dict[str, Any], answer: Answer | None, on_ui: Hook | None, consent: bool
    ) -> None:
        self.t.ui.append(ui)
        if on_ui is not None:
            r = on_ui(self, ui)
            if asyncio.iscoroutine(r):
                r = await r
            if r:
                return
        kind = ui.get("type")
        if kind == "consent":
            await self.ui("consent_given" if consent else "consent_declined")
        elif kind == "mic" and ui.get("start") and self._mic is None:
            self._mic = asyncio.create_task(self._mic_loop())
        elif kind == "keypad" and ui.get("enabled"):
            if self.voice_now():
                self.speak()
            else:
                keys = (answer or correct)(self.presented())
                if keys:
                    await self.dtmf(keys)

    def voice_now(self) -> bool:
        modes = self.t.of("mode")
        if modes:
            return bool(modes[-1]["mode"] == "voice")
        return self._mic is not None


async def admit(srv: Server, mode: str = "keypad", **kw: Any) -> dict[str, Any]:
    async with srv.http() as c:
        r = await c.post("/api/v1/sessions", json=create_body(mode, **kw))
    assert r.status_code in (200, 201, 202), r.text
    body: dict[str, Any] = r.json()
    return body
