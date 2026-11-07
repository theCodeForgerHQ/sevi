"""Deterministic harness for the orchestrator (B spec §13.7): memory transport, scripted STT,
capture and uplink fakes, a recording telemetry and a FakeClock-driven event driver.

No network, no real Gnani/Evon calls. Real A core (staircase, bank, scorer, classifier).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sevi.app.orchestrator import (
    BrowserSession,
    Orchestrator,
    OrchestratorDeps,
)
from sevi.app.prompts import placeholder_prompts
from sevi.app.settings import Settings, Switches
from sevi.contracts.adapters import (
    FrameTag,
    SttClosed,
    SttCommitSent,
    SttEvent,
    SttSessionConfig,
    SttTranscript,
    TClosed,
    TDtmf,
    TMark,
    TMedia,
    TStart,
    TUi,
)
from sevi.contracts.enums import Channel, EvonOutcome, KeyId, Lang, Mode, PresentationState
from sevi.contracts.errors import ErrorCode, SttError
from sevi.contracts.result import ResultDecision
from sevi.contracts.session import SessionRecord
from sevi.contracts.trial import TrialRecord
from sevi.contracts.wire import BrowserUiEvent, ClientHello, CreateSessionRequest
from sevi.core.result import build_classifier, t8_decision
from sevi.core.scoring import load_lexicons
from sevi.core.staircase import PROCEDURE
from sevi.core.stimulus import BankStimulusProvider
from sevi.core.wording import load_copy
from sevi.fakes.clock import FakeClock

_SCORER = None
_CLASSIFIER = None
_COPY = None


def shared_core() -> tuple[Any, Any, Any]:
    global _SCORER, _CLASSIFIER, _COPY
    if _SCORER is None:
        _SCORER = load_lexicons()
        _CLASSIFIER = build_classifier()
        _COPY = load_copy()
    return _SCORER, _CLASSIFIER, _COPY


# ---------- fakes ----------
class MemoryTransport:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Any]] = []
        self.closed: int | None = None

    @property
    def channel(self) -> Channel:
        return Channel.BROWSER

    async def send_media(self, ulaw: bytes) -> None:
        self.sent.append(("media", len(ulaw)))

    async def send_mark(self, name: str) -> None:
        self.sent.append(("mark", name))

    async def send_clear(self) -> None:
        self.sent.append(("clear", None))

    async def send_ui(self, msg: Any) -> None:
        self.sent.append(("ui", msg))

    async def events(self) -> AsyncIterator[Any]:  # pragma: no cover - the driver feeds events
        if False:
            yield None

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed is None:
            self.closed = code
            self.sent.append(("close", code))

    def marks(self) -> list[str]:
        return [v for k, v in self.sent if k == "mark"]

    def ui(self, typ: str) -> list[Any]:
        return [v for k, v in self.sent if k == "ui" and v.type == typ]


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def emit(self, event: Any, level: str = "INFO", **fields: Any) -> None:
        self.events.append((str(event), level, fields))

    @contextmanager
    def span(self, name: Any, **fields: Any) -> Iterator[None]:
        yield

    def mark_span(self, name: Any, start_mono_ms: float, end_mono_ms: float, **fields: Any) -> None:
        self.events.append((f"span:{name}", "DEBUG", {"dur_ms": end_mono_ms - start_mono_ms}))

    def names(self) -> list[str]:
        return [e for e, _, _ in self.events]


class MemoryStore:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionRecord] = {}
        self.trials: list[TrialRecord] = []
        self.puts = 0

    async def put_session(self, rec: SessionRecord) -> None:
        self.puts += 1
        self.sessions[rec.session_id] = rec

    async def append_trial(self, trial: TrialRecord) -> None:
        self.trials.append(trial)


@dataclass(frozen=True)
class Speech:
    present: bool
    speech_ms: int
    last_mono_ms: float | None


class ScriptedCapture:
    """Speech is wherever the test says it is (capture-time intervals)."""

    def __init__(self) -> None:
        self.speech: list[tuple[float, float]] = []
        self.early: float | None = None
        self.echo = False
        self.fed = 0

    def feed(self, frame_dbfs: float, mono_ms: float, gated: bool, playing: bool) -> None:
        self.fed += 1

    def floor_dbfs(self) -> float:
        return -55.0

    def speech_in(self, window: tuple[float, float]) -> Speech:
        a, b = window
        ms = 0.0
        last: float | None = None
        for s, e in self.speech:
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                ms += hi - lo
                last = hi
        return Speech(ms >= 300, int(ms), last)

    def any_speech_after(self, t_ms: float, until_ms: float) -> bool:
        return any(e > t_ms and s < until_ms for s, e in self.speech)

    def early_onset(self, start_ms: float, end_ms: float) -> float | None:
        e = self.early
        return e if e is not None and start_ms <= e < end_ms else None

    def add_echo_interval(self, start_ms: float, end_ms: float) -> None:
        return None

    def echo_coupling_db(self) -> float | None:
        return 12.0 if self.echo else 2.0

    def echo_mode(self) -> bool:
        return self.echo


@dataclass(frozen=True)
class PFrame:
    pcm: bytes
    tag: FrameTag | None


@dataclass(frozen=True)
class CFrame:
    dbfs: float
    mono_ms: float


class FakeUplink:
    """One Prisma frame per media message, tagged/gated by capture time."""

    def __init__(self, gated_at: Any, tag_at: Any) -> None:
        self.gated_at = gated_at
        self.tag_at = tag_at
        self.dropped = 0

    def on_media(self, m: TMedia) -> tuple[list[CFrame], list[PFrame]]:
        t = m.capture_mono_ms if m.capture_mono_ms is not None else m.received_mono_ms
        gated = self.gated_at(t)
        pcm = b"\x00" * 1024 if gated else b"\x01" * 1024
        return [CFrame(-40.0, t)], [PFrame(pcm, self.tag_at(t))]

    def flush_tail(self) -> PFrame | None:
        return PFrame(b"\x00" * 1024, None)

    def drop_pending(self) -> None:
        self.dropped += 1


class FakeSttSession:
    def __init__(self, cfg: SttSessionConfig, clock: FakeClock) -> None:
        self.cfg = cfg
        self.clock = clock
        self.frames: list[FrameTag | None] = []
        self.commits = 0
        self.closed = False
        self.q: asyncio.Queue[SttEvent | None] = asyncio.Queue()

    @property
    def request_id(self) -> str:
        return self.cfg.request_id

    @property
    def config_echo(self) -> dict[str, Any]:
        return {}

    def send_frame(self, pcm: bytes, tag: FrameTag | None) -> None:
        if self.closed:
            raise SttError(ErrorCode.STT_DISCONNECTED, "closed")
        self.frames.append(tag)

    def commit(self) -> None:
        self.commits += 1
        self.q.put_nowait(SttCommitSent(t_mono_ms=self.clock.mono_ms()))

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev
            if isinstance(ev, SttClosed):
                return

    async def close(self) -> None:
        self.closed = True
        self.q.put_nowait(None)

    def push(self, ev: SttEvent) -> None:
        self.q.put_nowait(ev)


class FakeSttClient:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.sessions: list[FakeSttSession] = []
        self.fail: list[BaseException | None] = []
        self.hold = False  # never complete open() (start_timeout path)
        self._hold_ev = asyncio.Event()

    async def open(self, cfg: SttSessionConfig) -> FakeSttSession:
        if self.hold:
            await self._hold_ev.wait()
        if self.fail:
            err = self.fail.pop(0)
            if err is not None:
                raise err
        s = FakeSttSession(cfg, self.clock)
        self.sessions.append(s)
        return s

    @property
    def current(self) -> FakeSttSession:
        return self.sessions[-1]


@dataclass(frozen=True)
class Explained:
    evon: EvonOutcome
    text: str | None


@dataclass
class FakeExplainer:
    text: str | None = "A calm explanation."
    evon: EvonOutcome = EvonOutcome.LIVE
    hang: bool = False
    calls: list[str] = field(default_factory=list)

    def eligible(self, channel: Channel, decision: ResultDecision, evon_switch: bool) -> bool:
        return evon_switch

    async def explain(self, decision: ResultDecision, lang: Lang) -> Explained:
        self.calls.append(decision.band.value)
        if self.hang:
            await asyncio.sleep(3600)
        return Explained(self.evon, self.text)


class Permit:
    def __init__(self) -> None:
        self.released = 0

    def release(self) -> None:
        self.released += 1


# ---------- build ----------
@dataclass
class Rig:
    orch: Orchestrator
    sess: BrowserSession
    token: str
    clock: FakeClock
    t: MemoryTransport
    tm: RecordingTelemetry
    store: MemoryStore
    stt: FakeSttClient
    capture: ScriptedCapture
    explainer: FakeExplainer
    permit: Permit
    deps: OrchestratorDeps
    auto_media: bool = False
    next_media_at: float = 0.0
    ts: int = 0

    # ----- driving -----
    async def pump(self) -> None:
        """Let background tasks run and dispatch everything that is ready."""
        idle = 0
        while idle < 3:
            await asyncio.sleep(0)
            moved = False
            while (ev := self.sess.take_ready()) is not None:
                await self.sess.dispatch(ev)
                moved = True
            idle = 0 if moved else idle + 1

    async def send(self, ev: Any) -> None:
        await self.sess.dispatch(ev)
        await self.pump()

    async def advance(self, ms: float) -> None:
        """Advance the fake clock by ``ms``, firing due timers in order."""
        target = self.clock.mono_ms() + ms
        await self.advance_to(target)

    async def wait(self, ms: float, ping_every_ms: float = 15_000.0) -> None:
        """Advance while the page keeps pinging (so T_RX_IDLE never fires)."""
        end = self.clock.mono_ms() + ms
        while self.clock.mono_ms() < end and not self.sess.is_ended():
            await self.advance_to(min(end, self.clock.mono_ms() + ping_every_ms))
            if not self.sess.is_ended():
                await self.ui_event("ping")

    async def advance_to(self, target: float) -> None:
        """Advance the clock to ``target``, firing due timers in deadline order. Voice rigs
        keep the microphone alive with one ``media`` frame per second (``auto_media``)."""
        while True:
            await self.pump()
            if self.sess.is_ended():
                break
            now = self.clock.mono_ms()
            step_end = min(target, self.next_media_at) if self.auto_media else target
            nxt = self.sess.timers.next_at()
            if nxt is not None and nxt <= step_end:
                if nxt > now:
                    self.clock.advance((nxt - now) / 1000.0)
                fired = self.sess.timers.pop_due(self.clock.mono_ms())
                if fired is not None:
                    await self.sess.dispatch(fired)
                continue
            if step_end > now:
                self.clock.advance((step_end - now) / 1000.0)
            if self.auto_media and self.clock.mono_ms() >= self.next_media_at:
                self.next_media_at += 1000.0
                await self.media_now()
                continue
            if self.clock.mono_ms() >= target:
                break
        await self.pump()

    async def media_now(self) -> None:
        now = self.clock.mono_ms()
        self.ts = max(self.ts + 20, int(now))
        await self.sess.dispatch(
            TMedia(
                ulaw=b"\xff" * 160,
                seq=self.ts,
                ts_ms=self.ts,
                received_mono_ms=now,
                capture_mono_ms=now,
            )
        )

    async def start(self) -> None:
        await self.sess.accept(self.t)  # type: ignore[arg-type]
        await self.send(
            TStart(
                stream_id="WB1",
                call_sid=None,
                custom={"session_id": self.sess.session_id, "token": self.token},
            )
        )

    async def ui_event(self, typ: str, **kw: Any) -> None:
        await self.send(TUi(event=BrowserUiEvent.model_validate({"type": typ, **kw})))

    async def finish_clip(self, extra_ms: float = 60.0) -> str | None:
        """Play out the outstanding clip and return its mark (None if nothing is playing)."""
        c = self.sess.player.outstanding
        if c is None or c.t_send is None:
            return None
        await self.advance_to(max(self.clock.mono_ms(), c.t_send + c.duration_ms + extra_ms))
        if self.sess.player.outstanding is not c:
            return None
        await self.send(TMark(name=c.mark, received_mono_ms=self.clock.mono_ms(), cleared=False))
        return c.mark

    async def to_listen(self, limit: int = 20) -> None:
        """Drive clips and the gate until the current presentation listens."""
        for _ in range(limit):
            p = self.sess.pres
            if self.sess.state.value not in ("TEST",) or p is None:
                if self.sess.player.outstanding is not None:
                    await self.finish_clip()
                    continue
                return
            if p.p is PresentationState.LISTEN:
                return
            if p.p is PresentationState.GATED:
                gate = self.sess.timers.armed("T_GATE")
                assert gate is not None
                await self.advance_to(gate)
                continue
            if self.sess.player.outstanding is not None:
                await self.finish_clip()
                continue
            await self.pump()
        raise AssertionError(f"never reached LISTEN: {self.sess.pres}")

    async def to_consent_done(self) -> None:
        await self.start()
        await self.finish_clip()  # consent prompt
        await self.ui_event("consent_given")

    async def through_instructions(self) -> None:
        assert self.sess.state.value == "INSTRUCTIONS", self.sess.state
        await self.finish_clip()  # instructions mark -> S42

    async def keypad_answer(self, digits: tuple[int, ...] | None = None) -> None:
        await self.to_listen()
        p = self.sess.pres
        assert p is not None
        ds = digits if digits is not None else p.spec.triplet
        for d in ds:
            await self.send(TDtmf(key=str(d), received_mono_ms=self.clock.mono_ms()))  # type: ignore[arg-type]
            await self.advance(150)

    async def voice_answer(self, text: str | None = None, *, latency: int = 300) -> None:
        """Speak ``text`` (default: the presented triplet) into the open window."""
        await self.to_listen()
        if text is None:
            p = self.sess.pres
            assert p is not None
            text = " ".join(str(d) for d in p.spec.triplet)
        tag = self.sess.current_tag
        now = self.clock.mono_ms()
        self.capture.speech.append((now, now + 900))
        await self.advance(1300)
        self.stt.current.push(
            SttTranscript(
                text=text,
                segment_id="g1",
                segment_index=1,
                audio_duration_ms=900,
                latency_ms=latency,
                detected_language_code="ta-IN",
                received_mono_ms=self.clock.mono_ms(),
                bound=tag,
            )
        )
        await self.pump()

    async def media(self, n: int = 1, step_ms: float = 20.0) -> None:
        for _ in range(n):
            await self.media_now()
            await self.advance(step_ms)

    async def drain_to_end(self, limit: int = 40) -> None:
        for _ in range(limit):
            if self.sess.is_ended():
                return
            if self.sess.player.outstanding is not None:
                await self.finish_clip()
            else:
                await self.advance(1000)
        raise AssertionError(f"session did not end: {self.sess.state}")

    async def hangup(self, clean: bool = True) -> None:
        await self.send(TClosed(code=1000 if clean else None, clean=clean))

    async def close(self) -> None:
        await self.sess.ended()
        await self.pump()

    def decision(self) -> ResultDecision:
        d = self.sess.rec.decision
        assert d is not None
        return d

    def playlist_marks(self) -> list[str]:
        return [m for m in self.t.marks() if m.startswith("z")]


def build(
    bank: BankStimulusProvider,
    *,
    mode: Mode = Mode.KEYPAD,
    earphones: bool = True,
    browser: str = "Chrome",
    evon: bool = True,
    explainer: FakeExplainer | None = None,
    prompt_ms: float = 400.0,
    **kw: Any,
) -> Rig:
    scorer, classifier, copy = shared_core()
    clock = FakeClock()
    settings = Settings(env="test")
    tm = RecordingTelemetry()
    store = MemoryStore()
    stt = FakeSttClient(clock)
    capture = ScriptedCapture()
    ex = explainer or FakeExplainer()
    sw = Switches.from_settings(settings)
    sw.evon = evon
    deps = OrchestratorDeps(
        settings=settings,
        clock=clock,
        telemetry=tm,  # type: ignore[arg-type]
        store=store,
        procedure=PROCEDURE,
        scorer=scorer,
        classifier=classifier,
        t8_decision=t8_decision,
        banks={Lang.TA: bank},
        prompts={
            Lang.TA: placeholder_prompts(Lang.TA, bank.manifest.digit_set, duration_ms=prompt_ms)
        },
        copy_text=copy.text,
        stt=stt,  # type: ignore[arg-type]
        explainer=ex,
        capture_factory=lambda: capture,
        uplink_factory=FakeUplink,
        switches=sw,
    )
    orch = Orchestrator(deps)
    req = CreateSessionRequest(
        lang=Lang.TA,
        mode=mode,  # type: ignore[arg-type]
        earphones=earphones,
        client=ClientHello(
            browser_family=browser,
            os_family="Android",
            ua="Mozilla/5.0 test",
            audio_worklet=True,
            sample_rate_hz=48000,
            page_version="abcdef012345",
        ),
    )
    permit = Permit()
    if mode is Mode.VOICE:
        kw.setdefault("key_id", KeyId.B)
    sess, token = orch.create_browser_session(
        req, permit=permit if mode is Mode.VOICE else None, **kw
    )
    rig = Rig(
        orch, sess, token, clock, MemoryTransport(), tm, store, stt, capture, ex, permit, deps
    )
    rig.auto_media = mode is Mode.VOICE
    rig.next_media_at = clock.mono_ms() + 1000.0
    return rig
