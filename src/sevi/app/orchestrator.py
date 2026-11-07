"""Browser session FSM + presentation FSM (contracts §6.1-§6.6, B spec §7).

One ``BrowserSession`` per test. Every event (transport, STT, timers, UI, internal results) goes
through one queue and is handled strictly in arrival order by ``dispatch``. Every transition
records its spec rule id (``S..`` / ``P..``) in ``rules`` and emits ``fsm.transition``.

The FSM is transport-agnostic: it talks to a ``CallTransport`` (contracts §5.2) and never to a
socket, so the phone IVR (#19) can reuse the presentation logic.

Seams that are not on ``main`` yet are reached through small Protocols defined here
(``CaptureLike`` = ``core.capture.CaptureTracker``, ``UplinkLike`` = ``transport.uplink.Uplink``,
``ExplainerLike`` = ``app.explain.Explainer``, ``HeldPermit`` = an admission slot). The wiring
passes the real implementations once they merge.

Timers are deterministic: ``Timers`` keeps absolute ``Clock.mono_ms()`` deadlines. ``run()``
waits on the queue until the next deadline (production). Tests drive ``dispatch`` and
``Timers.pop_due`` directly with a ``FakeClock`` (see ``SessionDriver`` in the tests).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import secrets
import statistics
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Final, Literal, Protocol

from sevi import __version__
from sevi.app.prompts import (
    PromptClip,
    PromptSet,
    digit_key,
    mark_for_feedback,
    mark_for_playlist,
    mark_for_prompt,
    mark_for_stimulus,
    silence,
    wording_key,
)
from sevi.app.settings import Settings, Switches
from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.adapters import (
    CallTransport,
    Clock,
    FrameTag,
    STTClient,
    SttClosed,
    SttCommitSent,
    SttErrorMsg,
    SttEvent,
    SttProcessing,
    STTSession,
    SttSessionConfig,
    SttSpeechStart,
    SttTranscript,
    TClosed,
    TDtmf,
    Telemetry,
    TMark,
    TMedia,
    TransportEvent,
    TStart,
    TStop,
    TUi,
    VadConfig,
)
from sevi.contracts.enums import (
    AnswerStatus,
    Channel,
    Condition,
    ConsentMethod,
    ConsentOutcome,
    ConsentScript,
    EndReason,
    EvonOutcome,
    FriendsVariant,
    KeyId,
    Lang,
    Mode,
    ParticipantType,
    PresentationKind,
    PresentationState,
    PromptKey,
    RetentionClass,
    ScoreAction,
    SessionFlag,
    SessionState,
    Surface,
    SystemEvent,
    Transducer,
    TruthRule,
    WindowCloseReason,
)
from sevi.contracts.errors import ErrorCode, SeviError
from sevi.contracts.languages import EXPERIMENTAL_LANGS
from sevi.contracts.ports import AdaptiveProcedure, AnswerScorer, ResultClassifier, StimulusProvider
from sevi.contracts.procedure import ProcedureConfig, ProcedureState, SrtResult
from sevi.contracts.result import ResultDecision, ResultInput
from sevi.contracts.session import ClientInfo, ConsentRecord, SessionRecord
from sevi.contracts.stimulus import RenderedTrial
from sevi.contracts.telemetry import EventName, SpanName
from sevi.contracts.trial import (
    AnswerEvidence,
    DtmfPress,
    PresentationSpec,
    ScoringResult,
    SttSegment,
    TrialRecord,
)
from sevi.contracts.wire import (
    CreateSessionRequest,
    ServerUiMessage,
    UiClosed,
    UiConsent,
    UiExplanation,
    UiFeedback,
    UiKeypad,
    UiMic,
    UiMode,
    UiNotice,
    UiPhase,
    UiPong,
    UiResult,
)

# ---------- constants (contracts §6.1) ----------
T_MIC_FIRST_MS: Final = 15_000.0
T_MEDIA_STALL_MS: Final = 6_000.0
T_FLOOR_MS: Final = 600.0
QUEUE_MAX: Final = 4_096
FEEDBACK_GAP_MS: Final = 200.0
LATENCY_DEFAULT_MS: Final = 400.0
SAFARI_SPEAKER_GUARD_EXTRA_MS: Final = 150
RENDER_TIMEOUT_S: Final = 2.0
STT_FRAME_MS: Final = 56.0  # backlog x 56 ms in the provisional T_COMMIT_MAX (§6.1)
SEGMENTS_MAX: Final = 20
SESSION_ACTIVE: Final = frozenset(
    {SessionState.CONSENT, SessionState.INSTRUCTIONS, SessionState.TEST}
)
LISTENING: Final = frozenset({PresentationState.LISTEN, PresentationState.COMMIT})


# ---------- internal events (B spec §7.1) ----------
@dataclass(frozen=True)
class SttEnvelope:
    gen: int
    ev: SttEvent


@dataclass(frozen=True)
class SttOpened:
    gen: int
    session: STTSession


@dataclass(frozen=True)
class SttOpenFailed:
    gen: int
    error: BaseException


@dataclass(frozen=True)
class SttReopened:
    gen: int
    session: STTSession


@dataclass(frozen=True)
class SttReopenFailed:
    gen: int
    error: BaseException


@dataclass(frozen=True)
class PlayerSent:
    mark: str
    t_send_mono_ms: float
    kind: str
    duration_ms: float
    deferred_ms: float


@dataclass(frozen=True)
class TimerFired:
    name: str
    token: int


@dataclass(frozen=True)
class BreakerOpened:
    key_id: KeyId


@dataclass(frozen=True)
class OperatorAbort:
    reason: str


@dataclass(frozen=True)
class ExplainFinished:
    text: str | None
    outcome: EvonOutcome


Event = (
    TransportEvent
    | SttEnvelope
    | SttOpened
    | SttOpenFailed
    | SttReopened
    | SttReopenFailed
    | PlayerSent
    | TimerFired
    | BreakerOpened
    | OperatorAbort
    | ExplainFinished
)


# ---------- small ports for pieces not on main yet ----------
class HeldPermit(Protocol):
    """An admission slot (a KeySemaphore 'stream' permit). ``release`` is idempotent."""

    def release(self) -> None: ...


class SpeechStatsLike(Protocol):
    @property
    def present(self) -> bool: ...
    @property
    def speech_ms(self) -> int: ...
    @property
    def last_mono_ms(self) -> float | None: ...


class CaptureLike(Protocol):
    """``core.capture.CaptureTracker`` (PR #40)."""

    def feed(self, frame_dbfs: float, mono_ms: float, gated: bool, playing: bool) -> None: ...
    def floor_dbfs(self) -> float: ...
    def speech_in(self, window: tuple[float, float]) -> SpeechStatsLike: ...
    def any_speech_after(self, t_ms: float, until_ms: float) -> bool: ...
    def early_onset(self, start_ms: float, end_ms: float) -> float | None: ...
    def add_echo_interval(self, start_ms: float, end_ms: float) -> None: ...
    def echo_coupling_db(self) -> float | None: ...
    def echo_mode(self) -> bool: ...


class CaptureFrameLike(Protocol):
    @property
    def dbfs(self) -> float: ...
    @property
    def mono_ms(self) -> float: ...


class PrismaFrameLike(Protocol):
    @property
    def pcm(self) -> bytes: ...
    @property
    def tag(self) -> FrameTag | None: ...


class UplinkLike(Protocol):
    """``transport.uplink.Uplink`` (PR #40)."""

    def on_media(
        self, m: TMedia
    ) -> tuple[Sequence[CaptureFrameLike], Sequence[PrismaFrameLike]]: ...
    def flush_tail(self) -> PrismaFrameLike | None: ...
    def drop_pending(self) -> None: ...


UplinkFactory = Callable[[Callable[[float], bool], Callable[[float], FrameTag | None]], UplinkLike]


class ExplainResultLike(Protocol):
    @property
    def evon(self) -> EvonOutcome: ...
    @property
    def text(self) -> str | None: ...


class ExplainerLike(Protocol):
    """``app.explain.Explainer`` (PR #36): guarded Evon text with template fallback."""

    def eligible(self, channel: Channel, decision: ResultDecision, evon_switch: bool) -> bool: ...
    async def explain(self, decision: ResultDecision, lang: Lang) -> ExplainResultLike: ...


class StorePort(Protocol):
    async def put_session(self, rec: SessionRecord) -> None: ...
    async def append_trial(self, trial: TrialRecord) -> None: ...


@dataclass(frozen=True)
class _NoSpeech:
    present: bool = False
    speech_ms: int = 0
    last_mono_ms: float | None = None


class NullCapture:
    """Keypad sessions: no microphone, never speech."""

    def feed(self, frame_dbfs: float, mono_ms: float, gated: bool, playing: bool) -> None:
        return None

    def floor_dbfs(self) -> float:
        return -60.0

    def speech_in(self, window: tuple[float, float]) -> SpeechStatsLike:
        return _NoSpeech()

    def any_speech_after(self, t_ms: float, until_ms: float) -> bool:
        return False

    def early_onset(self, start_ms: float, end_ms: float) -> float | None:
        return None

    def add_echo_interval(self, start_ms: float, end_ms: float) -> None:
        return None

    def echo_coupling_db(self) -> float | None:
        return None

    def echo_mode(self) -> bool:
        return False


# ---------- helpers ----------
def new_session_id() -> str:
    """UUIDv7 (RFC 9562): 48-bit Unix ms, version 7, variant 10, 74 random bits."""
    ms = time.time_ns() // 1_000_000
    rnd = int.from_bytes(secrets.token_bytes(10), "big") & ((1 << 74) - 1)
    rand_a, rand_b = rnd >> 62, rnd & ((1 << 62) - 1)
    v = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{v:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def sha256_hex(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def open_utterance(
    close_reason: WindowCloseReason,
    speech_present: bool,
    segments: Sequence[SttSegment],
    window_open_mono: float,
    close_mono: float,
    last_vad_mono: float | None,
    outstanding: int,
    any_speech_after: Callable[[float, float], bool],
) -> bool:
    """Contracts §6.3.1: may an utterance still be open at window close? (B spec §7.8)."""
    if close_reason is WindowCloseReason.DTMF_COMPLETE:
        return False
    nonempty = [s for s in segments if s.text.strip()]
    last = nonempty[-1] if nonempty else None
    a = close_reason is WindowCloseReason.TIMEOUT and speech_present
    b = False
    last_abs: float | None = None
    if last is not None:
        last_abs = window_open_mono + last.received_ms
        lat = last.latency_ms if last.latency_ms is not None else LATENCY_DEFAULT_MS
        b = any_speech_after(last_abs - lat + 50.0, close_mono)
    c = last is None and speech_present
    ref = last_abs if last_abs is not None else window_open_mono
    d = (last_vad_mono is not None and last_vad_mono > ref) or outstanding > 0
    return a or b or c or d


class Timers:
    """Named one-shot deadlines on the monotonic clock; re-arming replaces (B spec §7.3)."""

    def __init__(self) -> None:
        self._at: dict[str, tuple[float, int]] = {}
        self._tok = 0

    def arm(self, name: str, at_mono_ms: float) -> None:
        self._tok += 1
        self._at[name] = (at_mono_ms, self._tok)

    def cancel(self, name: str) -> None:
        self._at.pop(name, None)

    def cancel_all(self) -> None:
        self._at.clear()

    def armed(self, name: str) -> float | None:
        v = self._at.get(name)
        return None if v is None else v[0]

    def next_at(self) -> float | None:
        return min((v[0] for v in self._at.values()), default=None)

    def pop_due(self, now_mono_ms: float) -> TimerFired | None:
        due = [(at, tok, n) for n, (at, tok) in self._at.items() if at <= now_mono_ms]
        if not due:
            return None
        _at, tok, name = min(due)
        del self._at[name]
        return TimerFired(name, tok)


class Intervals:
    """Gate intervals ``[start, end]`` on the capture timeline (contracts §6.1)."""

    def __init__(self) -> None:
        self._iv: list[list[float]] = []

    def open(self, start: float) -> int:
        self._iv.append([start, float("inf")])
        return len(self._iv) - 1

    def set_start(self, h: int, start: float) -> None:
        self._iv[h][0] = start

    def set_end(self, h: int, end: float) -> None:
        self._iv[h][1] = end

    def end_of(self, h: int) -> float:
        return self._iv[h][1]

    def contains(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self._iv)


class WindowTags:
    def __init__(self) -> None:
        self._w: list[list[Any]] = []

    def open(self, tag: FrameTag, at: float) -> None:
        self._w.append([at, float("inf"), tag])

    def close(self, at: float) -> None:
        if self._w and self._w[-1][1] == float("inf"):
            self._w[-1][1] = at

    def tag_at(self, t: float) -> FrameTag | None:
        for a, b, tag in reversed(self._w):
            if a <= t <= b:
                return tag  # type: ignore[no-any-return]
        return None


# ---------- outbound player (B spec §7.5) ----------
ClipKind = Literal["prompt", "stim", "feedback", "playlist"]


@dataclass
class _Clip:
    ulaw: bytes
    mark: str
    kind: ClipKind
    duration_ms: float
    gate: int
    requested_mono: float
    key: PromptKey | None = None
    t_send: float | None = None


@dataclass(frozen=True)
class MarkResult:
    name: str
    kind: ClipKind
    key: PromptKey | None
    t_send: float
    rtt_ms: float
    duration_ms: float
    lost: bool = False


class OutboundPlayer:
    """One clip outstanding; every clip opens a gate interval; posts ``PlayerSent``."""

    def __init__(self, s: BrowserSession) -> None:
        self._s = s
        self.outstanding: _Clip | None = None
        self.deferred: _Clip | None = None

    @property
    def busy(self) -> bool:
        return self.outstanding is not None

    async def play(
        self,
        ulaw: bytes,
        mark: str,
        *,
        kind: ClipKind,
        duration_ms: float,
        key: PromptKey | None = None,
    ) -> int:
        s = self._s
        now = s.clock.mono_ms()
        clip = _Clip(ulaw, mark, kind, duration_ms, s.gates.open(now), now, key)
        if self.busy:
            self.deferred = clip
        else:
            await self._send(clip)
        return clip.gate

    async def _send(self, clip: _Clip) -> None:
        s = self._s
        t = s.transport
        clip.t_send = s.clock.mono_ms()
        s.gates.set_start(clip.gate, clip.t_send)
        self.outstanding = clip
        if t is not None and s.connected:
            try:
                await t.send_media(clip.ulaw)
                await t.send_mark(clip.mark)
            except SeviError:
                s.post(TClosed(code=None, clean=False))
        s.post_internal(
            PlayerSent(
                clip.mark,
                clip.t_send,
                clip.kind,
                clip.duration_ms,
                clip.t_send - clip.requested_mono,
            )
        )

    async def on_mark(self, m: TMark) -> MarkResult | None:
        s = self._s
        c = self.outstanding
        if m.cleared or c is None or m.name != c.mark or c.t_send is None:
            s.emit(EventName.MARK_RECEIVED, "DEBUG", mark=m.name, unexpected=True)
            return None
        rtt = m.received_mono_ms - c.t_send
        s.emit(EventName.MARK_RECEIVED, "DEBUG", mark=m.name, rtt_ms=round(rtt, 1))
        if c.kind != "stim":
            s.gates.set_end(c.gate, m.received_mono_ms + s.d_est_ms)
        s.capture_clip_done(c.t_send, m.received_mono_ms)
        self.outstanding = None
        res = MarkResult(c.mark, c.kind, c.key, c.t_send, rtt, c.duration_ms)
        await self._send_deferred()
        return res

    async def lost(self, name: str) -> MarkResult | None:
        c = self.outstanding
        if c is None or c.mark != name or c.t_send is None:
            return None
        s = self._s
        now = s.clock.mono_ms()
        if c.kind != "stim":
            s.gates.set_end(c.gate, now + s.d_est_ms)
        self.outstanding = None
        res = MarkResult(c.mark, c.kind, c.key, c.t_send, now - c.t_send, c.duration_ms, True)
        await self._send_deferred()
        return res

    async def _send_deferred(self) -> None:
        if self.deferred is not None:
            d, self.deferred = self.deferred, None
            await self._send(d)

    async def clear(self) -> None:
        s = self._s
        now = s.clock.mono_ms()
        for c in (self.outstanding, self.deferred):
            if c is not None and s.gates.end_of(c.gate) == float("inf"):
                s.gates.set_end(c.gate, now + s.d_est_ms)
        self.outstanding = None
        self.deferred = None
        s.timers.cancel("T_MARK_LOST")
        if s.transport is not None and s.connected:
            with contextlib.suppress(SeviError):
                await s.transport.send_clear()


# ---------- presentation context (B spec §7.7) ----------
@dataclass
class PresentationCtx:
    spec: PresentationSpec
    p: PresentationState = PresentationState.SEND
    rendered: RenderedTrial | None = None
    window_index: Literal[1, 2] = 1
    reask_used: bool = False
    replay_used: bool = False
    replay_n: int = 0
    segments: list[tuple[SttSegment, int]] = field(default_factory=list)
    dtmf: list[DtmfPress] = field(default_factory=list)
    window_open_mono: float = 0.0
    window_close_mono: float = 0.0
    window_mode: Mode = Mode.KEYPAD
    close_reason: WindowCloseReason | None = None
    system_events: set[SystemEvent] = field(default_factory=set)
    t_send: float = 0.0
    gate_end: float = 0.0
    gate_handle: int = -1
    t_commit: float | None = None
    last_vad_mono: float | None = None
    early_speech: bool = False
    early_answer: bool = False
    stale_segments: int = 0
    turn_ms: float | None = None
    post_commit_transcripts: int = 0
    arrival: int = 0
    instructions_restart: bool = False


# ---------- dependencies ----------
@dataclass
class OrchestratorDeps:
    settings: Settings
    clock: Clock
    telemetry: Telemetry
    store: StorePort
    procedure: AdaptiveProcedure
    scorer: AnswerScorer
    classifier: ResultClassifier
    t8_decision: Callable[[SrtResult | None], ResultDecision]
    banks: Mapping[Lang, StimulusProvider]
    prompts: Mapping[Lang, PromptSet]
    copy_text: Callable[[str, Lang], str]
    stt: STTClient | None = None
    explainer: ExplainerLike | None = None
    capture_factory: Callable[[], CaptureLike] = NullCapture
    uplink_factory: UplinkFactory | None = None
    switches: Switches | None = None
    git_sha: str = "0" * 40
    is_breaker_open: Callable[[KeyId], bool] = field(default=lambda _k: False)


# ---------- the session ----------
class BrowserSession:
    """Session runtime + both FSMs (B spec §7.2)."""

    def __init__(
        self,
        deps: OrchestratorDeps,
        rec: SessionRecord,
        *,
        ws_token: str,
        permit: HeldPermit | None,
        condition: Condition = Condition.NORMAL,
    ) -> None:
        self.deps = deps
        self.settings = deps.settings
        self.clock = deps.clock
        self.tm = deps.telemetry
        self.rec = rec
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_MAX)
        self._internal: deque[Event] = deque()
        self.overflowed = False
        self.transport: CallTransport | None = None
        self.connected = False
        self.ws_token_sha256 = sha256_hex(ws_token)
        self.token_used = False
        self.permit = permit
        self.switches = deps.switches or Switches.from_settings(deps.settings)
        self.mode: Mode = rec.mode_requested
        self.lang = rec.lang
        self.bank = deps.banks[rec.lang]
        self.prompts = deps.prompts[rec.lang]
        self.config_condition = condition
        # STT
        self.stt: STTSession | None = None
        self.stt_gen = 0
        self.stt_n = 0
        self.stt_opening = False
        self.stt_reopening = False
        self._stt_tasks: set[asyncio.Task[Any]] = set()
        # capture
        self.capture: CaptureLike = deps.capture_factory()
        self.gates = Intervals()
        self.windows = WindowTags()
        self.current_tag: FrameTag | None = None
        self.uplink: UplinkLike | None = (
            deps.uplink_factory(self.gates.contains, self.windows.tag_at)
            if deps.uplink_factory is not None and rec.mode_requested is Mode.VOICE
            else None
        )
        self.outstanding = 0
        self.player = OutboundPlayer(self)
        self.timers = Timers()
        self.first_media_mono: float | None = None
        self.mic_started = False
        self.last_rx_mono = self.clock.mono_ms()
        self.last_new_ts_mono: float | None = None
        self.last_ts: int | None = None
        self.d_est_samples: list[tuple[float, float]] = []
        self.d_est_ms: float = float(self.settings.capture.d_est_default_browser_ms)
        self.d_est_fixed = False
        self.t_guard_ms = rec.t_guard_ms
        self.echo_mode = False
        self.instr_repeats = 0
        self.instr_playing_mark: str | None = None
        self._echo_union: list[tuple[float, float]] = []
        self.proc: ProcedureState | None = None
        self.pres: PresentationCtx | None = None
        self.held_send: PresentationCtx | None = None
        self.n_presentations = 0
        self.last_answer_end_mono: float | None = None
        self.voice_scored = 0
        self.keypad_scored = 0
        self.flags: set[SessionFlag] = set(rec.flags)
        self.playlist: list[PromptKey] = []
        self.playlist_idx = 0
        self.result_entered = False
        self.drop_media = False
        self.explain_task: asyncio.Task[ExplainFinished] | None = None
        self.explain_text: str | None = None
        self.report_token: str | None = None
        self.prompt_seq = 0
        self.trials: list[TrialRecord] = []
        self.rules: list[str] = []
        self.ui_sent: list[ServerUiMessage] = []
        self._ended_done = False
        self.main_task: asyncio.Task[Any] | None = None
        self._pending_ui: list[ServerUiMessage] = []
        self._last_result: tuple[ScoringResult, SpeechStatsLike] | None = None
        self.consent_text = ""
        self.study_form_id: str | None = None

    # ----- small utilities -----
    @property
    def session_id(self) -> str:
        return self.rec.session_id

    @property
    def state(self) -> SessionState:
        return self.rec.state

    def emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        self.tm.emit(event, level, session_id=self.session_id, **fields)

    def rule(self, rule_id: str, to: SessionState | PresentationState | None = None) -> None:
        self.rules.append(rule_id)
        self.emit(
            EventName.FSM_TRANSITION,
            "DEBUG",
            rule_id=rule_id,
            state=self.state.value,
            to=None if to is None else to.value,
        )

    def goto(self, to: SessionState, rule_id: str) -> None:
        self.rule(rule_id, to)
        self._update(state=to)

    def _update(self, **kw: Any) -> None:
        self.rec = self.rec.model_copy(update=kw)

    def end(self, reason: EndReason) -> None:
        if self.rec.end_reason is None:
            self._update(end_reason=reason)

    def now(self) -> float:
        return self.clock.mono_ms()

    def post(self, ev: Event) -> None:
        """Producer side (transport reader, STT reader, tasks). Overflow -> crash (R-20)."""
        try:
            self.queue.put_nowait(ev)
        except asyncio.QueueFull:
            self.overflowed = True
            if self.main_task is not None:
                self.main_task.cancel()

    def post_internal(self, ev: Event) -> None:
        self._internal.append(ev)

    def take_ready(self) -> Event | None:
        if self._internal:
            return self._internal.popleft()
        if not self.queue.empty():
            return self.queue.get_nowait()
        return None

    async def ui(self, msg: ServerUiMessage) -> None:
        if self.transport is not None and self.connected:
            self.ui_sent.append(msg)
            try:
                await self.transport.send_ui(msg)
            except SeviError:
                self.post(TClosed(code=None, clean=False))

    async def persist(self) -> None:
        self._update(flags=tuple(f for f in SessionFlag if f in self.flags))
        await self.deps.store.put_session(self.rec)

    def capture_clip_done(self, t_send: float, t_mark: float) -> None:
        out = self.player.outstanding
        if out is not None and out.mark == self.instr_playing_mark:
            self._echo_union.append((t_send, t_mark))

    # ----- lifecycle -----
    async def accept(self, transport: CallTransport) -> None:
        """S13: the WS upgrade was accepted for this session (AWAIT_START)."""
        self.transport = transport
        self.connected = True
        now = self.now()
        self.last_rx_mono = now
        self.timers.arm("T_START", now + self.settings.session.t_start_s * 1000.0)
        self.timers.arm("T_RX_IDLE", now + self.settings.session.t_rx_idle_s * 1000.0)
        self.emit(EventName.WS_ACCEPTED)
        self.rule("S13", SessionState.AWAIT_START)

    async def run(self) -> None:
        """session_main (production): wait for events or the next timer deadline."""
        self.main_task = asyncio.current_task()
        try:
            while self.state is not SessionState.ENDED:
                ev = await self._next_event()
                await self.dispatch(ev)
        except asyncio.CancelledError:
            if not self.overflowed:
                raise
            await self._crash(SeviError(ErrorCode.INTERNAL, "session queue overflow"))
        finally:
            await self.ended()

    async def _next_event(self) -> Event:
        while True:
            ev = self.take_ready()
            if ev is not None:
                return ev
            fired = self.timers.pop_due(self.now())
            if fired is not None:
                return fired
            nxt = self.timers.next_at()
            timeout = None if nxt is None else max(0.0, (nxt - self.now()) / 1000.0)
            try:
                return await asyncio.wait_for(self.queue.get(), timeout)
            except TimeoutError:
                continue

    async def dispatch(self, ev: Event) -> None:
        """Handle one event; exceptions become S18 / S55 / S64."""
        if self.state is SessionState.ENDED:
            return
        try:
            await self._handle(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._crash(e)
        if self.is_ended():
            await self.ended()

    def is_ended(self) -> bool:
        return self.rec.state is SessionState.ENDED

    # ----- routing -----
    async def _handle(self, ev: Event) -> None:
        if isinstance(ev, TimerFired):
            await self._on_timer(ev.name)
            return
        if isinstance(ev, TMedia | TMark | TDtmf | TUi | TStart | TStop | TClosed):
            self.last_rx_mono = self.now()
        st = self.state
        if isinstance(ev, TClosed):
            await self._on_closed(ev)
        elif isinstance(ev, TStart):
            await self._on_start(ev)
        elif isinstance(ev, TStop):
            await self._on_closed(TClosed(code=1000, clean=True))
        elif isinstance(ev, TMedia):
            await self._on_media(ev)
        elif isinstance(ev, TUi):
            await self._on_ui(ev)
        elif isinstance(ev, TMark):
            await self._on_mark(ev)
        elif isinstance(ev, TDtmf):
            await self._on_dtmf(ev)
        elif isinstance(ev, PlayerSent):
            await self._on_player_sent(ev)
        elif isinstance(ev, SttEnvelope):
            if ev.gen == self.stt_gen and self.stt is not None:
                await self._on_stt(ev.ev)
        elif isinstance(ev, SttOpened | SttReopened):
            await self._on_stt_opened(ev)
        elif isinstance(ev, SttOpenFailed | SttReopenFailed):
            await self._on_stt_open_failed(ev)
        elif isinstance(ev, BreakerOpened):
            if st is SessionState.TEST and self.mode is Mode.VOICE:
                await self._switch_to_keypad("S51", "breaker_open", SystemEvent.BREAKER_TRIP)
        elif isinstance(ev, OperatorAbort):
            await self._on_operator_abort(ev.reason)
        elif isinstance(ev, ExplainFinished):
            self.explain_text = ev.text
            self._update(evon=ev.outcome)
            if self.state is SessionState.CLOSING:
                self.explain_task = None
                await self._finish_closing()

    def unexpected(self, what: str) -> None:
        self.emit(EventName.FSM_UNEXPECTED_EVENT, "WARNING", state=self.state.value, ev=what)

    # ----- S11 / S12 -----
    async def _on_start(self, ev: TStart) -> None:
        if self.state is not SessionState.AWAIT_START:
            self.unexpected("start")
            return
        token = ev.custom.get("token", "")
        sid = ev.custom.get("session_id", self.session_id)
        if self.token_used or sid != self.session_id or sha256_hex(token) != self.ws_token_sha256:
            self.rule("S12")
            self.emit(EventName.WS_REJECTED, reason="token")
            if self.transport is not None:
                await self.transport.close(4401, "SESSION_TOKEN_INVALID")
            return
        self.token_used = True
        self.timers.cancel("T_START")
        self.emit(EventName.SESSION_STARTED)
        self.goto(SessionState.CONSENT, "S11")
        await self._enter_consent()

    # ----- S20 -----
    def _consent_script(self) -> ConsentScript:
        study = self.rec.participant_type is ParticipantType.STUDY
        return ConsentScript.STUDY_V1 if study else ConsentScript.PUBLIC_V1

    async def _enter_consent(self) -> None:
        self.rule("S20", SessionState.CONSENT)
        script = self._consent_script()
        page_key = (
            "consent.study.page" if script is ConsentScript.STUDY_V1 else "consent.public.page"
        )
        self.consent_text = self.deps.copy_text(page_key, self.lang)
        await self.ui(UiPhase(state=SessionState.CONSENT, kind=None))
        await self.ui(UiConsent(script=script, text=self.consent_text))
        key = (
            PromptKey.CONSENT_STUDY
            if script is ConsentScript.STUDY_V1
            else PromptKey.CONSENT_PUBLIC
        )
        await self.play_prompt(key)
        self.timers.arm(
            "T_CONSENT_BROWSER", self.now() + self.settings.session.t_consent_browser_s * 1000.0
        )
        self.emit(EventName.CONSENT_PROMPTED)

    async def play_prompt(self, key: PromptKey) -> str:
        self.prompt_seq += 1
        mark = mark_for_prompt(self.prompt_seq)
        clip = self.prompts.get(key)
        await self.player.play(
            clip.ulaw, mark, kind="prompt", duration_ms=clip.duration_ms, key=key
        )
        return mark

    def _consent_record(self, outcome: ConsentOutcome, method: ConsentMethod | None) -> None:
        script = self._consent_script()
        self._update(
            consent=ConsentRecord(
                session_id=self.session_id,
                script=script,
                script_sha256=sha256_hex(self.consent_text),
                lang=self.lang,
                outcome=outcome,
                method=method,
                repeats=0,
                at=self.clock.now_utc(),
                study_form_id=self.study_form_id if script is ConsentScript.STUDY_V1 else None,
            )
        )

    # ----- UI events -----
    async def _on_ui(self, ev: TUi) -> None:
        t = ev.event.type
        st = self.state
        if t == "ping":
            await self.ui(UiPong())
            return
        if t == "mic_settings":
            if st in SESSION_ACTIVE and self.rec.mode_requested is Mode.VOICE:
                e = ev.event
                if self.rec.client is not None:
                    client = self.rec.client.model_copy(
                        update={"mic_ec": e.ec, "mic_ns": e.ns, "mic_agc": e.agc}
                    )
                    self._update(client=client)
                if e.ns or e.agc:
                    self.flags.add(SessionFlag.MIC_PROCESSING)
                self.emit(EventName.MIC_SETTINGS, ec=e.ec, ns=e.ns, agc=e.agc)
                self.rule("S29")
            return
        if t == "consent_given" and st is SessionState.CONSENT and self.rec.consent is None:
            await self._on_consent_given()
        elif t == "consent_declined" and st is SessionState.CONSENT and self.rec.consent is None:
            self.timers.cancel("T_CONSENT_BROWSER")
            await self.player.clear()
            self._consent_record(ConsentOutcome.DECLINED, ConsentMethod.CLICK_DECLINE)
            self.emit(EventName.CONSENT_DECLINED)
            self.end(EndReason.CONSENT_DECLINED)
            self.goto(SessionState.RESULT, "S23")
            await self._enter_result()
        elif t == "cant_hear_confirmed" and st in SESSION_ACTIVE:
            await self.player.clear()
            self._abort_presentation()
            self.end(EndReason.CANT_HEAR)
            self.goto(SessionState.RESULT, "S52")
            await self._enter_result()
        elif t in ("mic_denied", "mic_lost"):
            await self._on_mic_problem("mic_denied" if t == "mic_denied" else "mic_lost")
        else:
            self.unexpected(f"ui:{t}")

    async def _on_consent_given(self) -> None:
        self.timers.cancel("T_CONSENT_BROWSER")
        if self.player.busy:
            await self.player.clear()  # IT-ORC-03: the consent sample is discarded
        self._consent_record(ConsentOutcome.GIVEN, ConsentMethod.CLICK)
        self._update(started_at=self.clock.now_utc())
        self.emit(EventName.CONSENT_GIVEN)
        self.timers.arm("T_SESSION_MAX", self.now() + self.settings.session.max_s * 1000.0)
        if self.mode is Mode.VOICE:
            self.rule("S22", SessionState.CONSENT)
            self.mic_started = True
            await self.ui(UiMic(start=True))
            self.timers.arm("T_MIC_FIRST", self.now() + T_MIC_FIRST_MS)
            self._open_stt(reopen=False)
        else:
            self.goto(SessionState.INSTRUCTIONS, "S22")
            await self._enter_instructions()

    async def _on_mic_problem(self, reason: Literal["mic_denied", "mic_lost"]) -> None:
        st = self.state
        if self.mode is not Mode.VOICE or not self.mic_started:
            self.unexpected(f"ui:{reason}")
            return
        if st is SessionState.TEST:
            await self._switch_to_keypad("S58", "mic_lost", SystemEvent.MIC_LOST)
        elif st in (SessionState.CONSENT, SessionState.INSTRUCTIONS):
            await self._switch_to_keypad("S28", reason, None)
            if st is SessionState.CONSENT:
                self.goto(SessionState.INSTRUCTIONS, "S27")
                await self._enter_instructions()

    # ----- S40 / S41 / S42 -----
    async def _enter_instructions(self) -> None:
        self.rule("S40", SessionState.INSTRUCTIONS)
        await self.ui(UiPhase(state=SessionState.INSTRUCTIONS, kind=None))
        self.instr_playing_mark = await self.play_prompt(PromptKey.INSTRUCTIONS)

    async def _after_instructions(self) -> None:
        """S42: the instructions mark returned (not cleared)."""
        self.instr_playing_mark = None
        if self.stt_opening:  # IT-ORC-29: open still pending at the instructions mark
            await self._switch_to_keypad("S28", "start_timeout", None, start_failure=True)
        m = self.bank.manifest
        skip = self.config_condition is not Condition.NORMAL
        self.proc = self.deps.procedure.start(
            ProcedureConfig(
                seed_scope=self.session_id,
                design=m.design,
                digit_set=m.digit_set,
                condition=self.config_condition,
                skip_volume_and_practice=skip,
            )
        )
        self.goto(SessionState.TEST, "S42")
        if self.proc.pending is None:  # pragma: no cover - start() always sets pending
            self.goto(SessionState.RESULT, "S50")
            await self._enter_result()
            return
        await self._begin(self.proc.pending)

    # ----- DTMF -----
    async def _on_dtmf(self, ev: TDtmf) -> None:
        st = self.state
        self.emit(EventName.DTMF_RECEIVED, "DEBUG", key=ev.key)
        if st is SessionState.INSTRUCTIONS:
            playing = self.player.outstanding
            if (
                ev.key == "0"
                and playing is not None
                and playing.key is PromptKey.INSTRUCTIONS
                and self.instr_repeats < self.settings.session.instruction_repeats_max
            ):
                await self.player.clear()
                self.instr_repeats += 1
                self.rule("S41", SessionState.INSTRUCTIONS)
                self.instr_playing_mark = await self.play_prompt(PromptKey.INSTRUCTIONS)
            else:
                self.emit(EventName.DTMF_IGNORED, "DEBUG", key=ev.key)
            return
        pres = self.pres
        if st is not SessionState.TEST or pres is None:
            self.emit(EventName.DTMF_IGNORED, "DEBUG", key=ev.key)
            return
        if pres.p is not PresentationState.LISTEN:
            if pres.p is PresentationState.GATED:
                self.rule("P06")
            self.emit(EventName.DTMF_IGNORED, "DEBUG", key=ev.key, p_state=pres.p.value)
            return
        k = ev.key
        if k in "123456789":
            at = max(0.0, ev.received_mono_ms - pres.window_open_mono)
            pres.dtmf.append(DtmfPress(key=k, at_ms=at))
            self.rule("P12")
            if len(pres.dtmf) >= 3:
                await self._close(WindowCloseReason.DTMF_COMPLETE)
        elif k == "*" and not pres.replay_used:
            self.rule("P13", PresentationState.SEND)
            self._cancel_window_timers()
            self.windows.close(self.now())
            pres.replay_used = True
            pres.replay_n = 1
            pres.segments.clear()
            pres.dtmf.clear()
            self.emit(EventName.ANSWER_REPLAY, source="dtmf")
            await self._send(pres)
        elif (
            k == "0"
            and pres.spec.kind is PresentationKind.PRACTICE
            and self.instr_repeats < self.settings.session.instruction_repeats_max
        ):
            self.rule("P14", PresentationState.SEND)
            self._cancel_window_timers()
            self.windows.close(self.now())
            self.instr_repeats += 1
            pres.instructions_restart = True
            pres.p = PresentationState.PRE_PROMPT
            await self.play_prompt(PromptKey.INSTRUCTIONS)
        else:
            self.rule("P15")
            self.emit(EventName.DTMF_IGNORED, "DEBUG", key=k)

    # ----- marks -----
    async def _on_mark(self, ev: TMark) -> None:
        if self.state is SessionState.ENDED:
            return
        res = await self.player.on_mark(ev)
        if res is None:
            return
        self.timers.cancel("T_MARK_LOST")
        await self._mark_done(res)

    async def _mark_done(self, res: MarkResult) -> None:
        st = self.state
        if res.kind == "prompt" and not res.lost and not self.d_est_fixed:
            sample = res.rtt_ms - res.duration_ms
            voice_ok = self.rec.mode_requested is not Mode.VOICE or (
                self.first_media_mono is not None and res.t_send > self.first_media_mono
            )
            if voice_ok:
                self.d_est_samples.append((res.t_send, sample))
        if st is SessionState.CONSENT:
            return  # consent prompt finished; wait for the click
        if st is SessionState.INSTRUCTIONS:
            if res.key is PromptKey.INSTRUCTIONS and res.name == self.instr_playing_mark:
                await self._after_instructions()
            return
        if st is SessionState.TEST:
            await self._pres_mark(res)
            return
        if st is SessionState.RESULT and res.kind == "playlist":
            await self._next_playlist_item()

    # ----- player sent -----
    async def _on_player_sent(self, ev: PlayerSent) -> None:
        self.emit(EventName.PLAYER_SENT, "DEBUG", mark=ev.mark, deferred_ms=round(ev.deferred_ms))
        self.timers.arm(
            "T_MARK_LOST",
            ev.t_send_mono_ms + ev.duration_ms + self.settings.capture.mark_timeout_extra_ms,
        )
        if self.last_answer_end_mono is not None:
            turn = ev.t_send_mono_ms - self.last_answer_end_mono
            self.tm.mark_span(
                SpanName.TURN_TOTAL,
                self.last_answer_end_mono,
                ev.t_send_mono_ms,
                session_id=self.session_id,
                kind=ev.kind,
            )
            if self.pres is not None and self.pres.turn_ms is None:
                self.pres.turn_ms = max(0.0, turn)
            self.last_answer_end_mono = None
        pres = self.pres
        if (
            ev.kind == "stim"
            and pres is not None
            and pres.p is PresentationState.SEND
            and pres.rendered is not None
        ):
            self.rule("P04b", PresentationState.GATED)
            pres.t_send = ev.t_send_mono_ms
            pres.gate_end = (
                pres.t_send
                + self.d_est_ms
                + pres.rendered.timing.last_digit_offset_ms
                + self.t_guard_ms
            )
            self.gates.set_end(pres.gate_handle, pres.gate_end)
            self.timers.arm("T_GATE", pres.gate_end)
            pres.p = PresentationState.GATED
            self.emit(
                EventName.PRESENTATION_SENT,
                presentation_id=pres.spec.presentation_id,
                gate_end_in_ms=round(pres.gate_end - pres.t_send),
            )

    # ----- media -----
    async def _on_media(self, ev: TMedia) -> None:
        st = self.state
        if self.drop_media or st not in SESSION_ACTIVE or self.mode is not Mode.VOICE:
            return
        cap = ev.capture_mono_ms if ev.capture_mono_ms is not None else ev.received_mono_ms
        if ev.ts_ms is not None and (self.last_ts is None or ev.ts_ms > self.last_ts):
            self.last_ts = ev.ts_ms
            self.last_new_ts_mono = self.now()
        if self.first_media_mono is None and self.mic_started:
            self.first_media_mono = cap
            self.timers.cancel("T_MIC_FIRST")
            self.timers.arm("T_MEDIA_STALL", self.now() + T_MEDIA_STALL_MS)
            if st is SessionState.CONSENT:
                self.timers.arm("T_FLOOR", self.now() + T_FLOOR_MS)
        gated = self.pres is not None and self.pres.p is PresentationState.GATED
        if gated and self.rules[-1:] != ["P05"]:
            self.rule("P05")
        if self.uplink is None:
            return
        frames, pframes = self.uplink.on_media(ev)
        playing = self.player.outstanding is not None
        for f in frames:
            self.capture.feed(f.dbfs, f.mono_ms, self.gates.contains(f.mono_ms), playing)
        if self.stt is not None and not self.stt_reopening:
            for p in pframes:
                try:
                    self.stt.send_frame(p.pcm, p.tag)
                except SeviError:
                    await self._on_stt(SttClosed(code=None, reason="send_failed"))
                    return

    # ----- timers -----
    async def _on_timer(self, name: str) -> None:
        st = self.state
        now = self.now()
        if name == "T_START" and st is SessionState.AWAIT_START:
            self.rule("S15")
            if self.transport is not None:
                await self.transport.close(4401, "start timeout")
            self.connected = False
            self._release_permit()
            self.end(EndReason.QUEUE_ABANDONED)
            self.goto(SessionState.ENDED, "S15")
        elif name == "T_RX_IDLE":
            due = self.last_rx_mono + self.settings.session.t_rx_idle_s * 1000.0
            if due > now:
                self.timers.arm("T_RX_IDLE", due)
                return
            self.emit(EventName.WS_RX_IDLE, "WARNING")
            if st in SESSION_ACTIVE:
                self.rule("S17")
                await self._on_closed(TClosed(code=None, clean=False))
            elif st in (SessionState.RESULT, SessionState.CLOSING):
                self.rule("S65")
                await self._on_closed(TClosed(code=None, clean=False))
            elif st is SessionState.AWAIT_START:
                await self._on_closed(TClosed(code=None, clean=False))
        elif name == "T_CONSENT_BROWSER" and st is SessionState.CONSENT:
            await self.player.clear()
            self._consent_record(ConsentOutcome.NO_REPLY, None)
            self.emit(EventName.CONSENT_NO_REPLY)
            self.end(EndReason.CONSENT_NO_REPLY)
            self.goto(SessionState.RESULT, "S26")
            await self._enter_result()
        elif name == "T_SESSION_MAX" and st in SESSION_ACTIVE:
            await self.player.clear()
            self._abort_presentation()
            self.end(EndReason.SESSION_TIMEOUT)
            self.goto(SessionState.RESULT, "S56")
            await self._enter_result()
        elif name == "T_FLOOR" and st is SessionState.CONSENT and self.rec.consent is not None:
            self.goto(SessionState.INSTRUCTIONS, "S27")
            await self._enter_instructions()
        elif name == "T_MIC_FIRST" and self.mode is Mode.VOICE:
            await self._on_mic_problem("mic_lost")
        elif name == "T_MEDIA_STALL" and self.mode is Mode.VOICE:
            ref = self.last_new_ts_mono if self.last_new_ts_mono is not None else now
            due = ref + T_MEDIA_STALL_MS
            if due > now:
                self.timers.arm("T_MEDIA_STALL", due)
                return
            await self._on_mic_problem("mic_lost")
        elif name == "T_MARK_LOST":
            c = self.player.outstanding
            if c is None:
                return
            self.flags.add(SessionFlag.MARK_LOST)
            self.emit(EventName.MARK_LOST, "WARNING", mark=c.mark)
            res = await self.player.lost(c.mark)
            if res is not None:
                await self._mark_done(res)
        elif name == "T_EXPLAIN":
            if self.explain_task is not None and not self.explain_task.done():
                self.explain_task.cancel()
                self._update(evon=EvonOutcome.TEMPLATE)
                self.emit(EventName.EVON_TEMPLATE, cause="deadline")
            self.explain_task = None
            if st is SessionState.CLOSING:
                await self._finish_closing()
        elif st is SessionState.TEST and self.pres is not None:
            await self._pres_timer(name)

    # ----- transport closed / abort -----
    async def _on_closed(self, ev: TClosed) -> None:
        st = self.state
        self.connected = False
        self.emit(EventName.WS_CLOSED, code=ev.code, clean=ev.clean)
        if st is SessionState.AWAIT_START:
            self._release_permit()
            self.end(EndReason.QUEUE_ABANDONED)
            self.goto(SessionState.ENDED, "S16")
        elif st in SESSION_ACTIVE:
            self._abort_presentation()
            if ev.clean:
                self.end(EndReason.CALLER_HANGUP)
                self.goto(SessionState.RESULT, "S53")
            else:
                self.end(EndReason.SOCKET_LOST)
                self.goto(SessionState.RESULT, "S54")
            await self._enter_result()
        elif st in (SessionState.RESULT, SessionState.CLOSING):
            self.playlist = []
            if self.explain_task is not None:
                self.explain_task.cancel()
                self.explain_task = None
            self.goto(SessionState.ENDED, "S63")

    async def _on_operator_abort(self, reason: str) -> None:
        st = self.state
        if st in (SessionState.RESULT, SessionState.CLOSING):
            self.end(EndReason.OPERATOR_ABORT)
            if st is SessionState.RESULT:
                self.playlist = []
                self.goto(SessionState.CLOSING, "S57")
                await self._enter_closing()
            return
        if st is SessionState.ENDED:
            return
        await self.player.clear()
        self._abort_presentation()
        self.end(EndReason.OPERATOR_ABORT)
        if st in (SessionState.AWAIT_START, SessionState.QUEUED):
            self._release_permit()
            self.goto(SessionState.ENDED, "S57")
            return
        self.goto(SessionState.RESULT, "S57")
        await self._enter_result()

    def _abort_presentation(self) -> None:
        self._cancel_window_timers()
        for n in ("T_GATE", "T_COMMIT_MAX", "T_GRACE"):
            self.timers.cancel(n)
        self.windows.close(self.now())
        self.pres = None
        self.held_send = None

    def _release_permit(self) -> None:
        if self.permit is not None:
            p, self.permit = self.permit, None
            with contextlib.suppress(Exception):
                p.release()

    # ----- crash (S18 / S55 / S64) -----
    async def _crash(self, e: BaseException) -> None:
        st = self.state
        code = e.code.value if isinstance(e, SeviError) else ErrorCode.INTERNAL.value
        self.emit(
            EventName.SESSION_CRASH, "ERROR", error_code=code, exc_type=type(e).__name__, state=st
        )
        if st in (SessionState.RESULT, SessionState.CLOSING):
            self.rule("S64", SessionState.ENDED)
            if self.explain_task is not None:
                self.explain_task.cancel()
            self._update(state=SessionState.ENDED)
            return
        if st in (SessionState.QUEUED, SessionState.AWAIT_START, SessionState.CREATED):
            self._release_permit()
            self.end(EndReason.SERVER_ERROR)
            self._update(decision=self.deps.t8_decision(None))
            self.goto(SessionState.ENDED, "S18")
            return
        if st in SESSION_ACTIVE:
            self._abort_presentation()
            self.end(EndReason.SERVER_ERROR)
            srt = self.deps.procedure.summarize(self.proc) if self.proc is not None else None
            self._update(srt=srt, decision=self.deps.t8_decision(srt))
            self.goto(SessionState.RESULT, "S55")
            try:
                await self._enter_result()
            except Exception as e2:  # crash inside RESULT entry -> S64
                await self._crash(e2)

    # ----- STT open / reopen / close -----
    def _stt_config(self) -> SttSessionConfig:
        p = self.settings.prisma
        mv = self.bank.manifest.model_version
        key = KeyId.A if self.rec.key_id is KeyId.A else KeyId.B
        self.stt_n += 1
        return SttSessionConfig(
            key_id=key,
            lang=self.lang,
            model_version=mv,
            format=p.format,
            keyterms=self.deps.scorer.keyterms(self.lang, mv) if p.keyterms_enabled else (),
            sample_rate_hz=8000 if p.sample_rate_hz == 8000 else 16000,
            vad=VadConfig(
                threshold=p.vad_threshold,
                min_silence_ms=p.min_silence_ms,
                min_speech_ms=p.min_speech_ms,
                min_segment_ms=p.min_segment_ms,
            ),
            idle_timeout_s=p.idle_timeout_s,
            request_id=f"{self.session_id}-s{self.stt_n}",
        )

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        t: asyncio.Task[Any] = asyncio.create_task(coro)
        self._stt_tasks.add(t)
        t.add_done_callback(self._stt_tasks.discard)
        return t

    def _open_stt(self, *, reopen: bool) -> None:
        client = self.deps.stt
        self.stt_gen += 1
        gen = self.stt_gen
        if client is None:
            self.post_internal(
                (SttReopenFailed if reopen else SttOpenFailed)(
                    gen, SeviError(ErrorCode.STT_DISCONNECTED, "no STT client")
                )
            )
            return
        cfg = self._stt_config()
        if reopen:
            self.stt_reopening = True
        else:
            self.stt_opening = True
        self.emit(EventName.STT_OPEN, request_id=cfg.request_id, reopen=reopen)

        async def opener() -> None:
            try:
                sess = await client.open(cfg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # SttError, BreakerOpenError, BudgetExceeded, timeouts
                self.post((SttReopenFailed if reopen else SttOpenFailed)(gen, e))
                return
            self.post((SttReopened if reopen else SttOpened)(gen, sess))

        self._spawn(opener())

    def _start_reader(self, sess: STTSession, gen: int) -> None:
        async def reader() -> None:
            try:
                async for ev in sess.events():
                    self.post(SttEnvelope(gen, ev))
            except asyncio.CancelledError:
                raise
            except Exception:
                self.post(SttEnvelope(gen, SttClosed(code=None, reason="reader_error")))

        self._spawn(reader())

    async def _on_stt_opened(self, ev: SttOpened | SttReopened) -> None:
        if (
            ev.gen != self.stt_gen
            or self.mode is not Mode.VOICE
            or self.state not in SESSION_ACTIVE
        ):
            await _close_quietly(ev.session)
            return
        self.stt = ev.session
        self.stt_opening = False
        self.emit(EventName.STT_CONNECTED, request_id=ev.session.request_id)
        self._start_reader(ev.session, ev.gen)
        if isinstance(ev, SttReopened):
            self.stt_reopening = False
            await self._resume_held()

    async def _on_stt_open_failed(self, ev: SttOpenFailed | SttReopenFailed) -> None:
        if ev.gen != self.stt_gen:
            return
        self.emit(EventName.STT_ERROR, "WARNING", error=type(ev.error).__name__)
        if isinstance(ev, SttOpenFailed):
            self.stt_opening = False
            if self.mode is Mode.VOICE and self.state in SESSION_ACTIVE:
                await self._switch_to_keypad("S22", "start_failed", None, start_failure=True)
            return
        self.stt_reopening = False
        if self.mode is Mode.VOICE:
            await self._switch_to_keypad("S51", "reconnect_failed", SystemEvent.BREAKER_TRIP)
        await self._resume_held()

    async def _close_stt(self) -> None:
        sess, self.stt = self.stt, None
        self.stt_gen += 1  # every later envelope of the old socket is stale
        self.stt_opening = False
        self.stt_reopening = False
        for t in list(self._stt_tasks):
            t.cancel()
        if sess is not None:
            await _close_quietly(sess)

    async def _switch_to_keypad(
        self,
        rule_id: str,
        reason: str,
        system_event: SystemEvent | None,
        *,
        start_failure: bool = False,
    ) -> None:
        """S28 / S51 / S58 / §6.2.1 step 2: voice -> keypad for the rest of the session."""
        if self.mode is not Mode.VOICE:
            return
        self.rule(rule_id)
        await self._close_stt()
        if self.uplink is not None:
            self.uplink.drop_pending()
        self._release_permit()
        self.mode = Mode.KEYPAD
        for n in ("T_MIC_FIRST", "T_MEDIA_STALL", "T_FLOOR"):
            self.timers.cancel(n)
        if reason in ("mic_denied", "mic_lost"):
            self.flags.add(SessionFlag.MIC_LOST)
        if system_event is SystemEvent.BREAKER_TRIP:
            self.flags.add(SessionFlag.BREAKER_TRIPPED)
        if not start_failure and self.voice_scored >= 1:
            self.flags.add(SessionFlag.MIXED_MODE)
        self.emit(EventName.MODE_SWITCHED, reason=reason, rule_id=rule_id)
        await self.ui(UiMode(mode=Mode.KEYPAD))
        pres = self.pres
        if pres is not None and pres.p in LISTENING and system_event is not None:
            await self._break_window(pres, system_event)
        if self.held_send is not None:
            await self._resume_held()
        if start_failure and self.state is SessionState.CONSENT and self.rec.consent is not None:
            self.goto(SessionState.INSTRUCTIONS, "S27")
            await self._enter_instructions()

    async def _break_window(self, pres: PresentationCtx, ev: SystemEvent) -> None:
        """P18: the answer window ends with a capture failure -> CAPTURE_MISS."""
        pres.system_events.add(ev)
        if pres.p is PresentationState.LISTEN:
            await self._close(WindowCloseReason.BREAKER)
        elif pres.p is PresentationState.COMMIT:
            await self._resolve_commit()

    async def _resume_held(self) -> None:
        pres, self.held_send = self.held_send, None
        if pres is not None and pres is self.pres:
            await self._send(pres)

    # ----- STT events (P07-P22) -----
    async def _on_stt(self, ev: SttEvent) -> None:
        pres = self.pres
        p = pres.p if pres is not None else None
        if isinstance(ev, SttClosed):
            await self._on_stt_closed(ev)
            return
        if isinstance(ev, SttErrorMsg):
            self.emit(EventName.STT_ERROR, "WARNING", message=ev.message[:200])
            if pres is not None and p in LISTENING:
                pres.system_events.add(SystemEvent.STT_ERROR)
                self.rule("P18b")
            return
        if isinstance(ev, SttSpeechStart):
            if pres is not None:
                pres.last_vad_mono = self.now()
            return
        if isinstance(ev, SttProcessing):
            self.outstanding += 1
            if pres is not None:
                pres.last_vad_mono = self.now()
                if p is PresentationState.LISTEN:
                    self.rule("P09")
                elif p is PresentationState.COMMIT:
                    self.rule("P21b")
            return
        if isinstance(ev, SttCommitSent):
            if pres is not None and p is PresentationState.COMMIT:
                self.rule("P20b")
                pres.t_commit = ev.t_mono_ms
                self.timers.arm("T_COMMIT_MAX", ev.t_mono_ms + self.settings.prisma.t_commit_max_ms)
            return
        if isinstance(ev, SttTranscript):
            await self._on_transcript(ev)

    async def _on_transcript(self, ev: SttTranscript) -> None:
        self.outstanding = max(0, self.outstanding - 1)
        pres = self.pres
        p = pres.p if pres is not None else None
        bound = (
            pres is not None
            and p in LISTENING
            and ev.bound is not None
            and ev.bound == self.current_tag
        )
        if pres is None or not bound:
            if pres is not None:
                pres.stale_segments += 1
                self.rule("P07" if p is PresentationState.GATED else "P10b")
            self.emit(
                EventName.STT_STALE_SEGMENT,
                "DEBUG",
                p_state=None if p is None else p.value,
                bound_pid=None if ev.bound is None else ev.bound.presentation_id,
            )
            return
        after_commit = p is PresentationState.COMMIT
        seg = SttSegment(
            text=ev.text[:500],
            segment_id=ev.segment_id,
            segment_index=ev.segment_index,
            received_ms=max(0.0, ev.received_mono_ms - pres.window_open_mono),
            audio_duration_ms=ev.audio_duration_ms,
            latency_ms=ev.latency_ms,
            after_commit=after_commit,
        )
        pres.arrival += 1
        pres.segments.append((seg, pres.arrival))
        self.emit(
            EventName.STT_TRANSCRIPT, "DEBUG", n_chars=len(ev.text), after_commit=after_commit
        )
        if after_commit:
            self.rule("P21")
            pres.post_commit_transcripts += 1
            if self.settings.prisma.commit_emits_processing:
                if self.outstanding == 0:
                    await self._resolve_commit()
            elif self.timers.armed("T_GRACE") is None:
                self.timers.arm("T_GRACE", self.now() + self.settings.prisma.late_grace_ms)
            return
        if not ev.text.strip():
            self.rule("P11")
            return
        self.rule("P10")
        texts = [s.text for s, _ in pres.segments]
        if self.deps.scorer.digits_so_far(self.lang, texts) >= 3:
            await self._close(WindowCloseReason.THREE_DIGITS)
        elif self.deps.scorer.intent_so_far(self.lang, texts) == "DONT_KNOW":
            await self._close(WindowCloseReason.INTENT)
        else:
            self.timers.arm("T_SETTLE", self.now() + self.settings.capture.t_settle_ms)

    async def _on_stt_closed(self, ev: SttClosed) -> None:
        """P18 / S51 / §8.2.6. A budget stop never reconnects (credit safety)."""
        self.emit(EventName.STT_CLOSED, "WARNING", code=ev.code, reason=ev.reason)
        if self.mode is not Mode.VOICE:
            return
        st = self.state
        budget = ev.reason == "budget"
        key = KeyId.A if self.rec.key_id is KeyId.A else KeyId.B
        if st is not SessionState.TEST:
            await self._switch_to_keypad("S22", "stt_closed", None, start_failure=True)
            return
        if budget or self.deps.is_breaker_open(key):
            await self._switch_to_keypad(
                "S51", "budget" if budget else "breaker_open", SystemEvent.BREAKER_TRIP
            )
            return
        # reconnect (CCR-v3-40): old socket closed first; the next SEND waits for the outcome
        self.rule("P18", PresentationState.SCORE)
        self.emit(EventName.STT_RECONNECT)
        await self._close_stt()
        self._open_stt(reopen=True)
        pres = self.pres
        if pres is not None and pres.p in LISTENING:
            await self._break_window(pres, SystemEvent.RECONNECT)

    # ----- presentation FSM -----
    async def _begin(self, spec: PresentationSpec) -> None:
        self.n_presentations += 1
        pres = PresentationCtx(spec=spec)
        self.pres = pres
        self.emit(
            EventName.PRESENTATION_BEGIN,
            presentation_id=spec.presentation_id,
            kind=spec.kind.value,
            scored_index=spec.scored_index,
        )
        await self.ui(
            UiPhase(state=SessionState.TEST, kind=spec.kind, scored_index=spec.scored_index)
        )
        if spec.pre_prompt is not None:
            self.rule("P01", PresentationState.PRE_PROMPT)
            pres.p = PresentationState.PRE_PROMPT
            mark = await self.play_prompt(spec.pre_prompt)
            if not self.d_est_fixed:
                self.instr_playing_mark = mark  # echo union: INSTRUCTIONS + first pre_prompt
        else:
            self.rule("P02", PresentationState.SEND)
            await self._send(pres)

    def _fix_d_est(self) -> None:
        if self.d_est_fixed:
            return
        self.d_est_fixed = True
        cfg = self.settings.capture
        samples = [s for _, s in self.d_est_samples]
        if len(samples) >= 2:
            self.d_est_ms = min(max(statistics.median(samples), 0.0), float(cfg.d_est_max_ms))
        else:
            self.d_est_ms = float(cfg.d_est_default_browser_ms)
            self.flags.add(SessionFlag.D_EST_DEFAULTED)
        if self.mode is Mode.VOICE:
            for a, b in self._echo_union:
                self.capture.add_echo_interval(a, b)
            self.echo_mode = self.capture.echo_mode()
            self._update(echo_coupling_db=self.capture.echo_coupling_db())
            self.emit(EventName.CAPTURE_FLOOR, floor_dbfs=round(self.capture.floor_dbfs(), 1))
            self.emit(EventName.CAPTURE_ECHO_MODE, echo_mode=self.echo_mode)
            if self.echo_mode:
                self.flags.add(SessionFlag.ECHO_MODE)
        self.instr_playing_mark = None
        self._update(d_est_ms=self.d_est_ms)

    async def _pres_mark(self, res: MarkResult) -> None:
        pres = self.pres
        if pres is None:
            return
        if pres.p is PresentationState.PRE_PROMPT and res.kind == "prompt":
            if pres.instructions_restart:  # P14 instructions replay finished
                pres.instructions_restart = False
                pres.segments.clear()
                pres.dtmf.clear()
                await self._send(pres)
                return
            self.rule("P03", PresentationState.SEND)
            first = not self.d_est_fixed
            self._fix_d_est()
            if first and self.echo_mode:
                await self.ui(UiNotice(key="speaker_echo"))
            await self._send(pres)
        elif pres.p is PresentationState.REASK_PROMPT and res.kind == "prompt":
            self.rule("P36", PresentationState.LISTEN)
            pres.window_index = 2
            pres.segments.clear()
            pres.dtmf.clear()
            pres.system_events.clear()
            pres.early_speech = False
            await self._open_window(pres, self.now())
        elif pres.p is PresentationState.FEEDBACK and res.kind == "feedback":
            self.rule("P41")
            await self._outcome()

    async def _send(self, pres: PresentationCtx) -> None:
        """P04a. Waits while a mid-session STT reconnect is in flight (CCR-v3-40)."""
        pres.p = PresentationState.SEND
        if self.stt_reopening:
            self.held_send = pres
            return
        if not self.d_est_fixed:
            self._fix_d_est()
        self.rule("P04a", PresentationState.SEND)
        if pres.rendered is None:
            with self.tm.span(SpanName.STIM_RENDER, session_id=self.session_id):
                pres.rendered = await asyncio.wait_for(
                    asyncio.to_thread(self.bank.render_trial, pres.spec), RENDER_TIMEOUT_S
                )
            if pres.rendered.level_limited:
                self.flags.add(SessionFlag.LEVEL_LIMITED)
        r = pres.rendered
        pres.gate_handle = await self.player.play(
            r.ulaw,
            mark_for_stimulus(pres.spec.presentation_id, pres.replay_n),
            kind="stim",
            duration_ms=r.timing.clip_ms,
        )
        await self.ui(UiKeypad(enabled=False))

    async def _open_window(self, pres: PresentationCtx, at: float) -> None:
        pres.p = PresentationState.LISTEN
        pres.window_open_mono = at
        pres.window_mode = Mode.VOICE if self.mode is Mode.VOICE else Mode.KEYPAD
        pres.last_vad_mono = None
        pres.post_commit_transcripts = 0
        pres.t_commit = None
        self.current_tag = FrameTag(
            presentation_id=pres.spec.presentation_id,
            window_index=pres.window_index,
            replay=pres.replay_n,
        )
        self.windows.open(self.current_tag, at)
        self.timers.arm("T_ANSWER", self.now() + self.settings.capture.t_answer_ms)
        self.emit(
            EventName.ANSWER_WINDOW_OPEN,
            presentation_id=pres.spec.presentation_id,
            window_index=pres.window_index,
        )
        await self.ui(UiKeypad(enabled=True))

    async def _pres_timer(self, name: str) -> None:
        pres = self.pres
        if pres is None:
            return
        if name == "T_GATE" and pres.p is PresentationState.GATED:
            self.rule("P08", PresentationState.LISTEN)
            if self.mode is Mode.VOICE and pres.window_index == 1 and pres.rendered is not None:
                onset = self.capture.early_onset(
                    pres.t_send + self.d_est_ms + pres.rendered.timing.digits[0].onset_ms,
                    pres.gate_end,
                )
                if onset is not None and not self.echo_mode:
                    pres.early_speech = True
                    self.emit(EventName.CAPTURE_EARLY_ANSWER, onset_ms=round(onset))
            await self._open_window(pres, pres.gate_end)
        elif name == "T_SETTLE" and pres.p is PresentationState.LISTEN:
            self.rule("P16")
            await self._close(WindowCloseReason.SETTLE)
        elif name == "T_ANSWER" and pres.p is PresentationState.LISTEN:
            self.rule("P17")
            await self._close(WindowCloseReason.TIMEOUT)
        elif name in ("T_COMMIT_MAX", "T_GRACE") and pres.p is PresentationState.COMMIT:
            self.rule("P22" if name == "T_COMMIT_MAX" else "P21")
            await self._resolve_commit()

    def _cancel_window_timers(self) -> None:
        self.timers.cancel("T_ANSWER")
        self.timers.cancel("T_SETTLE")

    async def _close(self, reason: WindowCloseReason) -> None:
        pres = self.pres
        if pres is None or pres.p is not PresentationState.LISTEN:
            return
        self._cancel_window_timers()
        now = self.now()
        pres.close_reason = reason
        pres.window_close_mono = now
        self.windows.close(now)
        self.emit(EventName.ANSWER_WINDOW_CLOSED, close_reason=reason.value)
        await self.ui(UiKeypad(enabled=False))
        if (
            reason is WindowCloseReason.BREAKER
            or self.mode is not Mode.VOICE
            or pres.window_mode is not Mode.VOICE
        ):
            self.rule("P19", PresentationState.SCORE)
            await self._score()
            return
        speech = self.capture.speech_in((pres.window_open_mono, now))
        segs = [s for s, _ in pres.segments]
        if not open_utterance(
            reason,
            speech.present,
            segs,
            pres.window_open_mono,
            now,
            pres.last_vad_mono,
            self.outstanding,
            self.capture.any_speech_after,
        ):
            self.rule("P20", PresentationState.SCORE)
            await self._score()
            return
        self.rule("P20", PresentationState.COMMIT)
        pres.p = PresentationState.COMMIT
        try:
            if self.uplink is not None and self.stt is not None:
                tail = self.uplink.flush_tail()
                if tail is not None:
                    self.stt.send_frame(tail.pcm, tail.tag)
            if self.stt is None:
                raise SeviError(ErrorCode.STT_DISCONNECTED, "no stt")
            self.stt.commit()
        except SeviError:
            await self._score()
            return
        backlog = float(getattr(self.stt, "backlog", 0) or 0)
        self.timers.arm(
            "T_COMMIT_MAX", now + backlog * STT_FRAME_MS + self.settings.prisma.t_commit_max_ms
        )

    async def _resolve_commit(self) -> None:
        self.timers.cancel("T_COMMIT_MAX")
        self.timers.cancel("T_GRACE")
        await self._score()

    def _evidence(self, pres: PresentationCtx) -> tuple[AnswerEvidence, SpeechStatsLike]:
        voice = pres.window_mode is Mode.VOICE
        speech: SpeechStatsLike = (
            self.capture.speech_in((pres.window_open_mono, pres.window_close_mono))
            if voice
            else _NoSpeech()
        )
        segs = sorted(
            pres.segments,
            key=lambda t: (t[0].segment_index if t[0].segment_index is not None else 10**9, t[1]),
        )
        ev = AnswerEvidence(
            presentation_id=pres.spec.presentation_id,
            lang=self.lang,
            mode=Mode.VOICE if voice else Mode.KEYPAD,
            window_index=pres.window_index,
            window_open_mono_ms=max(0.0, pres.window_open_mono),
            window_close_mono_ms=max(pres.window_open_mono, pres.window_close_mono, 0.0),
            close_reason=pres.close_reason or WindowCloseReason.SESSION_END,
            segments=tuple(s for s, _ in segs[:SEGMENTS_MAX]),
            dtmf=tuple(pres.dtmf[:SEGMENTS_MAX]),
            speech_present=speech.present,
            speech_ms=speech.speech_ms,
            early_speech=pres.early_speech and pres.window_index == 1,
            echo_mode=self.echo_mode,
            system_events=tuple(sorted(pres.system_events)),
            reask_used=pres.reask_used,
            replay_used=pres.replay_used,
        )
        return ev, speech

    async def _score(self) -> None:
        """P30-P34."""
        pres = self.pres
        if pres is None:
            return
        pres.p = PresentationState.SCORE
        self.rule("P30", PresentationState.SCORE)
        ev, speech = self._evidence(pres)
        with self.tm.span(SpanName.SCORE_NORMALISE, session_id=self.session_id):
            result = self.deps.scorer.score(pres.spec, ev)
        self.emit(
            EventName.SCORE_RESULT,
            status=result.status.value,
            action=result.action.value,
            source=result.source.value,
        )
        if result.status is AnswerStatus.EARLY_ANSWER:
            pres.early_answer = True
        if result.status is AnswerStatus.CAPTURE_MISS:
            self.emit(EventName.CAPTURE_MISS, presentation_id=pres.spec.presentation_id)
        if result.action is ScoreAction.REASK and result.reask_prompt is not None:
            self.rule("P31", PresentationState.REASK_PROMPT)
            pres.reask_used = True
            pres.p = PresentationState.REASK_PROMPT
            self.emit(EventName.ANSWER_REASK, prompt=result.reask_prompt.value)
            self.rule("P35", PresentationState.REASK_PROMPT)
            await self.play_prompt(result.reask_prompt)
            return
        if result.action is ScoreAction.REPLAY:
            self.rule("P32", PresentationState.SEND)
            pres.replay_used = True
            pres.replay_n = 1
            pres.segments.clear()
            pres.dtmf.clear()
            self.emit(EventName.ANSWER_REPLAY, source="speech")
            await self._send(pres)
            return
        self._last_result = (result, speech)
        if pres.dtmf:
            self.last_answer_end_mono = pres.window_open_mono + pres.dtmf[-1].at_ms
        elif pres.window_mode is Mode.VOICE and speech.last_mono_ms is not None:
            self.last_answer_end_mono = speech.last_mono_ms
        else:
            self.last_answer_end_mono = pres.window_close_mono
        if pres.spec.feedback:
            self.rule("P33", PresentationState.FEEDBACK)
            pres.p = PresentationState.FEEDBACK
            self.rule("P40", PresentationState.FEEDBACK)
            await self._play_feedback(pres, result)
            return
        self.rule("P34")
        await self._outcome()

    async def _play_feedback(self, pres: PresentationCtx, result: ScoringResult) -> None:
        parts: list[PromptClip] = []
        if result.triplet_correct:
            parts.append(self.prompts.get(PromptKey.PRACTICE_CORRECT))
            audio = parts[0].ulaw
            dur = parts[0].duration_ms
        else:
            wrong = self.prompts.get(PromptKey.PRACTICE_WRONG)
            gap = silence(FEEDBACK_GAP_MS)
            audio, dur = wrong.ulaw, wrong.duration_ms
            for d in pres.spec.triplet:
                c = self.prompts.get(digit_key(d))
                audio += gap + c.ulaw
                dur += FEEDBACK_GAP_MS + c.duration_ms
        await self.ui(UiFeedback(correct=result.triplet_correct))
        await self.player.play(
            audio, mark_for_feedback(pres.spec.presentation_id), kind="feedback", duration_ms=dur
        )

    async def _outcome(self) -> None:
        """S50: persist the TrialRecord, advance the procedure."""
        pres = self.pres
        if pres is None or self.proc is None or self._last_result is None:
            return
        result, speech = self._last_result
        r = pres.rendered
        spec = pres.spec
        rec = TrialRecord(
            session_id=self.session_id,
            spec=spec,
            result=result,
            system_flag=spec.attempt == 2 and result.status is AnswerStatus.CAPTURE_MISS,
            reasks=int(pres.reask_used),
            replays=int(pres.replay_used),
            early_answer=pres.early_answer,
            echo_mode=self.echo_mode,
            rx_floor_dbfs=self.capture.floor_dbfs() if pres.window_mode is Mode.VOICE else None,
            speech_ms=speech.speech_ms,
            t_send_mono_ms=max(0.0, pres.t_send),
            gate_end_mono_ms=max(0.0, pres.gate_end),
            d_est_ms=self.d_est_ms,
            t_guard_ms=self.t_guard_ms,
            last_digit_offset_ms=r.timing.last_digit_offset_ms if r is not None else 0.0,
            transcript=(" ".join(s.text for s, _ in pres.segments)[:1000] or None),
            stale_segments=pres.stale_segments,
            turn_ms=pres.turn_ms,
            level_limited=bool(r is not None and r.level_limited),
        )
        self.trials.append(rec)
        await self.deps.store.append_trial(rec)
        if spec.kind is PresentationKind.SCORED and result.status is not AnswerStatus.CAPTURE_MISS:
            if result.mode is Mode.VOICE:
                self.voice_scored += 1
            else:
                self.keypad_scored += 1
        with self.tm.span(SpanName.PROCEDURE_STEP, session_id=self.session_id):
            self.proc = self.deps.procedure.record(self.proc, result)
        self.emit(EventName.PROCEDURE_RECORD, presentation_id=spec.presentation_id)
        self.rule("S50")
        self.pres = None
        self.current_tag = None
        if self.proc.pending is not None:
            await self._begin(self.proc.pending)
        else:
            self.emit(EventName.PROCEDURE_TERMINAL, terminal=str(self.proc.terminal))
            self.goto(SessionState.RESULT, "S50")
            await self._enter_result()

    # ----- RESULT / CLOSING / ENDED (S60-S73) -----
    def _mode_effective(self) -> Mode:
        v, k = self.voice_scored, self.keypad_scored
        if v == 0 and k == 0:
            return Mode.KEYPAD if self.mode is Mode.KEYPAD else Mode.VOICE
        if v == 0:
            return Mode.KEYPAD
        if k == 0:
            return Mode.VOICE
        return Mode.MIXED

    async def _enter_result(self) -> None:
        if self.result_entered:
            return
        self.result_entered = True
        self.rule("S60", SessionState.RESULT)
        self._abort_presentation()
        for n in ("T_SESSION_MAX", "T_CONSENT_BROWSER", "T_MIC_FIRST", "T_MEDIA_STALL", "T_FLOOR"):
            self.timers.cancel(n)
        if self.mic_started and self.connected:
            await self.ui(UiMic(start=False))
        self.drop_media = True
        if self.rec.decision is None:
            await self._close_stt()
            self._release_permit()
            srt = self.deps.procedure.summarize(self.proc) if self.proc is not None else None
            self.end(EndReason.COMPLETED)
            reason = self.rec.end_reason or EndReason.COMPLETED
            decision = self.deps.classifier.classify(
                ResultInput(
                    session_id=self.session_id,
                    lang=self.lang,
                    channel=Channel.BROWSER,
                    mode=self._mode_effective(),
                    mode_requested=Mode.VOICE
                    if self.rec.mode_requested is Mode.VOICE
                    else Mode.KEYPAD,
                    transducer=self.rec.transducer,
                    participant_type=self.rec.participant_type,
                    end_reason=reason,
                    srt=srt,
                    friends=_friends(self.switches),
                    experimental_language=self.lang in EXPERIMENTAL_LANGS,
                    bank_hash=self.bank.manifest.bank_hash,
                    norms=None,
                    hhie_total=None,
                )
            )
            self._update(srt=srt, decision=decision)
            self.emit(EventName.RESULT_DECIDED, rule=decision.rule.value, band=decision.band.value)
            if decision.rule is TruthRule.T8 and reason not in (
                EndReason.SERVER_ERROR,
                EndReason.SOCKET_LOST,
            ):
                self.emit(EventName.ERROR, "ERROR", code=ErrorCode.PROCEDURE_PROTOCOL.value)
        else:
            await self._close_stt()
            self._release_permit()
        final = self.rec.decision
        assert final is not None
        decision = final
        if self.rec.report_token_sha256 is None and Surface.REPORT in decision.surfaces:
            self.report_token = secrets.token_urlsafe(16)
            self._update(report_token_sha256=sha256_hex(self.report_token))
        self._update(mode_effective=self._mode_effective())
        await self.persist()
        self.playlist = self._build_playlist(decision) if self.connected else []
        self.playlist_idx = 0
        self._start_explain(decision)
        if self.connected:
            page_text = (
                self.deps.classifier.wording(decision, self.lang, Surface.PAGE).text
                if Surface.PAGE in decision.surfaces
                else ""
            )
            report_url = (
                f"{str(self.settings.pages_origin).rstrip('/')}/sevi/r.html#{self.report_token}"
                if Surface.REPORT in decision.surfaces and self.report_token is not None
                else ""
            )
            await self.ui(UiPhase(state=SessionState.RESULT, kind=None))
            await self.ui(
                UiResult(
                    band=decision.band,
                    wording_class=decision.wording_class,
                    page_text=page_text,
                    srt_display=decision.srt_display,
                    report_url=report_url,
                )
            )
        await self._next_playlist_item(first=True)

    def _build_playlist(self, d: ResultDecision) -> list[PromptKey]:
        """Contracts §6.5 / B spec §7.11 step 3."""
        out: list[PromptKey] = []
        wk = wording_key(d.wording_class.value)
        if Surface.SPOKEN in d.surfaces and wk is not None and self.prompts.has(wk):
            out.append(wk)
        if d.rule is not TruthRule.T10:
            out.append(PromptKey.DISCLAIMER)
        out.append(PromptKey.GOODBYE_EARLY if d.rule is TruthRule.T7 else PromptKey.GOODBYE)
        return out

    def _start_explain(self, decision: ResultDecision) -> None:
        ex = self.deps.explainer
        if (
            ex is None
            or not self.connected
            or not ex.eligible(Channel.BROWSER, decision, self.switches.evon)
        ):
            self._update(evon=EvonOutcome.SKIPPED)
            return

        async def runner() -> ExplainFinished:
            try:
                res = await ex.explain(decision, self.lang)
                out = ExplainFinished(res.text, res.evon)
            except asyncio.CancelledError:
                raise
            except Exception:
                out = ExplainFinished(None, EvonOutcome.TEMPLATE)
            self.post(out)
            return out

        self.explain_task = asyncio.create_task(runner())
        self.timers.arm("T_EXPLAIN", self.now() + self.settings.evon.browser_deadline_s * 1000.0)

    async def _next_playlist_item(self, *, first: bool = False) -> None:
        if not first:
            self.playlist_idx += 1
        if self.connected and self.playlist_idx < len(self.playlist):
            key = self.playlist[self.playlist_idx]
            clip = self.prompts.get(key)
            self.rule("S61", SessionState.RESULT)
            await self.player.play(
                clip.ulaw,
                mark_for_playlist(self.playlist_idx),
                kind="playlist",
                duration_ms=clip.duration_ms,
                key=key,
            )
            return
        self.goto(SessionState.CLOSING, "S62")
        await self._enter_closing()

    async def _enter_closing(self) -> None:
        self.rule("S70", SessionState.CLOSING)
        await self._finish_closing()

    async def _finish_closing(self) -> None:
        if self.explain_task is not None and not self.explain_task.done():
            return  # wait for ExplainFinished or T_EXPLAIN
        task, self.explain_task = self.explain_task, None
        if task is not None and not task.cancelled() and task.exception() is None:
            res = task.result()
            self.explain_text = res.text
            self._update(evon=res.outcome)
        if self.connected:
            if self.explain_text:
                await self.ui(UiExplanation(text=self.explain_text[:300]))
            await self.ui(UiClosed())
            if self.transport is not None:
                await self.transport.close(1000, "")
            self.connected = False
        self.timers.cancel("T_EXPLAIN")
        self.goto(SessionState.ENDED, "S70")

    async def ended(self) -> None:
        """S73 (idempotent)."""
        if self._ended_done:
            return
        self._ended_done = True
        self.rule("S73")
        self.timers.cancel_all()
        if self.explain_task is not None:
            self.explain_task.cancel()
        await self._close_stt()
        self._release_permit()
        if self.transport is not None and self.connected:
            with contextlib.suppress(Exception):
                await self.transport.close(1000, "")
            self.connected = False
        self._update(
            state=SessionState.ENDED,
            ended_at=self.clock.now_utc(),
            mode_effective=self._mode_effective(),
        )
        with contextlib.suppress(Exception):
            await self.persist()
        self.emit(
            EventName.SESSION_ENDED,
            end_reason=None if self.rec.end_reason is None else self.rec.end_reason.value,
        )


def _friends(sw: Switches) -> FriendsVariant:
    return FriendsVariant.ALLOWED if sw.friends else FriendsVariant.DISALLOWED


async def _close_quietly(sess: STTSession) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(sess.close(), 1.0)


# ---------- orchestrator (B spec §7.1) ----------
class Orchestrator:
    """Creates sessions, keeps the registry, aborts, recovers orphans."""

    def __init__(self, deps: OrchestratorDeps) -> None:
        self.deps = deps
        self.sessions: dict[str, BrowserSession] = {}

    def create_browser_session(
        self,
        req: CreateSessionRequest,
        *,
        permit: HeldPermit | None,
        key_id: KeyId = KeyId.NONE,
        condition: Condition = Condition.NORMAL,
        participant_type: ParticipantType = ParticipantType.PUBLIC,
        study_form_id: str | None = None,
    ) -> tuple[BrowserSession, str]:
        """S01 (keypad) / S02 (voice, permit already held) / S08-lite (station condition)."""
        d = self.deps
        s = d.settings
        now = d.clock.now_utc()
        study = participant_type is ParticipantType.STUDY
        retention = RetentionClass.STUDY_KEEP if study else RetentionClass.DELETE_72H
        transducer = Transducer.EARPHONES if (req.earphones or study) else Transducer.SPEAKER
        safari_extra = (
            SAFARI_SPEAKER_GUARD_EXTRA_MS
            if transducer is Transducer.SPEAKER and req.client.browser_family == "Safari"
            else 0
        )
        bank = d.banks[req.lang]
        prompts = d.prompts[req.lang]
        sid = new_session_id()
        rec = SessionRecord(
            session_id=sid,
            contracts_version=CONTRACTS_VERSION,
            app_version=__version__,
            git_sha=d.git_sha,
            channel=Channel.BROWSER,
            mode_requested=req.mode,
            mode_effective=req.mode,
            transducer=transducer,
            lang=req.lang,
            participant_type=participant_type,
            participant_id=None,
            pseudonym=None,
            age_band=None,
            key_id=KeyId.NONE if req.mode is Mode.KEYPAD else key_id,
            state=SessionState.AWAIT_START,
            created_at=now,
            started_at=None,
            ended_at=None,
            end_reason=None,
            consent=None,
            d_est_ms=None,
            t_guard_ms=s.capture.t_guard_browser_ms + safari_extra,
            echo_coupling_db=None,
            bank_hash=bank.manifest.bank_hash,
            bank_version=bank.manifest.bank_version,
            prompts_hash=prompts.prompts_hash,
            retention=retention,
            delete_after=None if study else now + timedelta(hours=s.retention.hours),
            client=ClientInfo(
                browser_family=req.client.browser_family,
                os_family=req.client.os_family,
                ua_sha256=sha256_hex(req.client.ua),
                audio_worklet=req.client.audio_worklet,
                sample_rate_hz=req.client.sample_rate_hz,
            ),
        )
        token = secrets.token_urlsafe(16)
        sess = BrowserSession(d, rec, ws_token=token, permit=permit, condition=condition)
        sess.study_form_id = study_form_id
        self.sessions[sid] = sess
        d.telemetry.emit(
            EventName.SESSION_CREATED,
            session_id=sid,
            mode=req.mode.value,
            latency_source="none" if req.client.browser_family == "Safari" else "outputLatency",
        )
        return sess, token

    async def persist_created(self, sess: BrowserSession) -> None:
        await self.deps.store.put_session(sess.rec)

    async def abort(self, session_id: str, reason: str) -> bool:
        sess = self.sessions.get(session_id)
        if sess is None:
            return False
        sess.post(OperatorAbort(reason))
        return True

    def get(self, session_id: str) -> BrowserSession | None:
        return self.sessions.get(session_id)


__all__ = [
    "BreakerOpened",
    "BrowserSession",
    "CaptureLike",
    "ExplainFinished",
    "ExplainerLike",
    "HeldPermit",
    "NullCapture",
    "Orchestrator",
    "OrchestratorDeps",
    "OutboundPlayer",
    "PlayerSent",
    "PresentationCtx",
    "SttEnvelope",
    "SttOpenFailed",
    "SttOpened",
    "SttReopenFailed",
    "SttReopened",
    "TimerFired",
    "Timers",
    "UplinkLike",
    "new_session_id",
    "open_utterance",
]
