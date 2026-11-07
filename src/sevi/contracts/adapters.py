"""I/O adapter ports owned by B (contracts §5.2).

Scope (BUILD-BRIEF reduced scope), omitted: `SessionStore.panel_put_response` and
`SessionStore.panel_list_responses` (human panel cut), `Telemetry.subscribe` (SSE timeline cut).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import Field

from sevi.contracts.enums import Channel, KeyId, Lang, ModelVersion
from sevi.contracts.session import AllowlistEntry, SessionRecord
from sevi.contracts.telemetry import EventName, SpanName
from sevi.contracts.types import SessionId, SeviModel, Sha256Hex, UtcDatetime
from sevi.contracts.wire import BrowserUiEvent, ServerUiMessage

if TYPE_CHECKING:  # Person A's module (contracts §2.1)
    from sevi.contracts.trial import TrialRecord


# ---------- clock ----------
class Clock(Protocol):
    def now_utc(self) -> datetime: ...  # tz-aware UTC
    def mono_ms(self) -> float: ...  # time.monotonic_ns() / 1e6
    async def sleep_until(self, mono_ms: float) -> None: ...  # FakeClock advances instantly


# ---------- STT (Prisma streaming) ----------
class VadConfig(SeviModel):
    threshold: float = 0.7
    min_silence_ms: int = 800  # 0.2.1: live Prisma v3 split triplets at 300 ms
    min_speech_ms: int = 100
    min_segment_ms: int = 100


class SttSessionConfig(SeviModel):
    key_id: Literal[KeyId.A, KeyId.B]
    lang: Lang
    model_version: ModelVersion
    format: Literal["transcribe", "codemix", "verbatim"]
    keyterms: tuple[str, ...]  # empty = keyterms off
    sample_rate_hz: Literal[8000, 16000]
    vad: VadConfig
    idle_timeout_s: int
    request_id: str = Field(max_length=80)  # f"{session_id}-s{n}"


class FrameTag(SeviModel):
    """Which answer window a sent audio frame belongs to."""

    presentation_id: int = Field(ge=1)
    window_index: Literal[1, 2]
    replay: int = Field(0, ge=0, le=1)


class SttConnected(SeviModel):
    kind: Literal["connected"] = "connected"
    request_id: str | None
    config_echo: dict[str, Any]  # verbatim `config` object


class SttSpeechStart(SeviModel):
    kind: Literal["speech_start"] = "speech_start"
    utterance_id: str | None = None


class SttProcessing(SeviModel):
    kind: Literal["processing"] = "processing"


class SttTranscript(SeviModel):
    kind: Literal["transcript"] = "transcript"
    text: str
    segment_id: str | None
    segment_index: int | None
    audio_duration_ms: int | None
    latency_ms: int | None
    detected_language_code: str | None
    received_mono_ms: float
    stream_mid_ms: float | None = None  # estimated segment midpoint on the sent-stream timeline
    bound: FrameTag | None = None  # tag of the frame sent at stream_mid_ms; None = gated/untagged


class SttCommitSent(SeviModel):
    kind: Literal["commit_sent"] = "commit_sent"
    t_mono_ms: float


class SttErrorMsg(SeviModel):
    kind: Literal["error"] = "error"
    message: str


class SttClosed(SeviModel):
    kind: Literal["closed"] = "closed"
    code: int | None
    reason: str | None


SttEvent = (
    SttConnected
    | SttSpeechStart
    | SttProcessing
    | SttTranscript
    | SttCommitSent
    | SttErrorMsg
    | SttClosed
)


class STTSession(Protocol):
    @property
    def request_id(self) -> str: ...
    @property
    def config_echo(self) -> dict[str, Any]: ...
    def send_frame(self, pcm: bytes, tag: FrameTag | None) -> None:
        """Enqueue exactly PRISMA_FRAME_BYTES of s16le on the single pacer queue (§8.2.2).
        Raises ValueError on wrong length, SttError(STT_DISCONNECTED) if closed. Non-blocking."""
        ...

    def commit(self) -> None:
        """Enqueue {"type":"commit"} on the SAME pacer queue; the writer then emits SttCommitSent.
        Non-blocking. Raises SttError(STT_DISCONNECTED)."""
        ...

    def events(self) -> AsyncIterator[SttEvent]:
        """Single consumer. Unknown message types are logged and skipped. Ends after SttClosed."""
        ...

    async def close(self) -> None:
        """Idempotent; <= 1 s. Socket writes bounded by SEVI_SESSION__SEND_TIMEOUT_S."""
        ...


class STTClient(Protocol):
    async def open(self, cfg: SttSessionConfig) -> STTSession:
        """Connect + wait for `connected` within SEVI_PRISMA__CONNECT_TIMEOUT_S;
        validates config_echo (§8.2.3).
        Raises SttError (STT_AUTH, STT_CREDITS, STT_RATE_LIMITED, STT_CONNECT_TIMEOUT,
        STT_HANDSHAKE_REJECTED, STT_CONFIG_MISMATCH), BreakerOpenError(STT_BREAKER_OPEN).
        Caller MUST hold a KeySemaphore 'stream' permit for cfg.key_id."""
        ...


# ---------- STT REST (calibration, benchmark) ----------
class SttRestRequest(SeviModel):
    key_id: Literal[KeyId.A] = KeyId.A
    wav_8k_pcm16: bytes  # decoded from RenderedTrial.ulaw (or upsampled per chain)
    lang: Lang
    model_version: ModelVersion
    format: Literal["transcribe", "codemix", "verbatim"]
    keyterms: tuple[str, ...]
    repeat_index: int = Field(0, ge=0, le=4)  # part of the cache key


class SttRestResult(SeviModel):
    transcript: str
    request_id: str | None
    model: str | None
    processing_time_s: float | None
    cached: bool
    cache_key: Sha256Hex


class STTRestClient(Protocol):
    async def transcribe(self, req: SttRestRequest) -> SttRestResult:
        """Content-addressed cache first. Raises SttError(STT_REST_FAILED|STT_CREDITS|STT_AUTH)."""
        ...


# ---------- TTS (Timbre) ----------
class TTSClient(Protocol):
    async def synthesize_ulaw(
        self, text: str, lang: Lang, voice: str, model: str, key_id: KeyId
    ) -> bytes:
        """Prompt pre-render only. 8 kHz headerless mu-law. Raises TtsError(TTS_TEXT_TOO_LONG) if
        len(text) > 300 before any request; TTS_TIMEOUT, TTS_FAILED, TTS_BAD_REQUEST, TTS_CREDITS.
        Caller MUST hold a KeySemaphore 'tts' permit."""
        ...

    async def synthesize_wav(
        self, text: str, lang: Lang, voice: str, model: str, sample_rate_hz: int = 48000
    ) -> bytes:
        """Offline digit/prompt rendering (Key A). WAV linear_pcm. Timeout 20 s."""
        ...


# ---------- LLM (Evon via llama-server) ----------
class LlmRequest(SeviModel):
    purpose: Literal["EXPLAIN_SPOKEN", "REPORT_PROSE", "TRANSLATION_DRAFT"]
    lang: Lang
    system: str
    user: str
    json_schema: dict[str, Any]
    max_tokens: int
    temperature: float


class LlmResponse(SeviModel):
    raw_text: str
    parsed: dict[str, Any]
    tokens_out: int | None
    latency_ms: float
    finish_reason: str | None


class LLMClient(Protocol):
    async def complete_json(self, req: LlmRequest) -> LlmResponse:
        """POST /v1/chat/completions (§8.5). Total timeout SEVI_EVON__TIMEOUT_S. Raises
        LlmError(LLM_TIMEOUT|LLM_UNAVAILABLE|LLM_SCHEMA_INVALID),
        BreakerOpenError(LLM_BREAKER_OPEN).
        Never retries internally."""
        ...

    async def health(self) -> bool:
        """GET {base}/../health within 2 s; True iff HTTP 200 and body status 'ok'."""
        ...


# ---------- transport (browser) ----------
class TStart(SeviModel):
    kind: Literal["start"] = "start"
    stream_id: str
    call_sid: str | None
    custom: dict[str, str]


class TMedia(SeviModel):
    kind: Literal["media"] = "media"
    ulaw: bytes
    seq: int
    ts_ms: int | None
    received_mono_ms: float
    capture_mono_ms: float | None = None  # ts_ms + offset (min over last 50 frames of rx - ts)


class TMark(SeviModel):
    kind: Literal["mark"] = "mark"
    name: str
    received_mono_ms: float
    cleared: bool  # True if a clear was sent before this mark returned


class TDtmf(SeviModel):
    kind: Literal["dtmf"] = "dtmf"
    key: Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "#"]
    received_mono_ms: float


class TUi(SeviModel):
    kind: Literal["ui"] = "ui"
    event: BrowserUiEvent


class TStop(SeviModel):
    kind: Literal["stop"] = "stop"


class TClosed(SeviModel):
    kind: Literal["closed"] = "closed"
    code: int | None
    clean: bool  # False -> SOCKET_LOST


TransportEvent = TStart | TMedia | TMark | TDtmf | TUi | TStop | TClosed


class CallTransport(Protocol):
    """Every send_* only enqueues on the session's single outbound writer queue (maxsize 64).
    A send timeout or a full queue enqueues TClosed(clean=False) on events()."""

    @property
    def channel(self) -> Channel: ...
    async def send_media(self, ulaw: bytes) -> None:
        """One `media` message per clip. Enqueue only, <= 5 ms. Raises TransportError."""
        ...

    async def send_mark(self, name: str) -> None: ...  # name pattern §6.4
    async def send_clear(self) -> None: ...  # only on mode switch / hangup
    async def send_ui(self, msg: ServerUiMessage) -> None:
        """Browser: sends {"event":"kj","kj":msg}."""
        ...

    def events(self) -> AsyncIterator[TransportEvent]: ...  # single consumer
    async def close(self, code: int = 1000, reason: str = "") -> None: ...  # idempotent


# ---------- storage ----------
class LedgerRow(SeviModel):
    key_id: KeyId
    stt_s: float
    tts_chars: int
    stt_rest_calls: int
    credits_used_est: float
    credits_total: float
    updated_at: UtcDatetime


class LedgerEvent(SeviModel):
    """One row of the single credit ledger (§14.2)."""

    event_id: str = Field(pattern=r"^[0-9a-f]{32}$")  # uuid4 hex, idempotency key
    key_id: Literal[KeyId.A, KeyId.B]
    bucket: Literal["CAL", "OPS", "RESERVE", "DEV", "PUBLIC"]
    purpose: Literal[
        "calibration",
        "bench",
        "vt",
        "g5",
        "prompt_render",
        "validation",
        "video",
        "dev",
        "public_voice",
        "phone_record",
        "interview",
        "channel_check",
    ]
    window_id: str = Field(max_length=32)
    origin: Literal["server", "manual"]
    stt_s: float = 0
    tts_chars: int = 0
    stt_rest_calls: int = 0
    credits_est: float = 0
    dashboard_used: float | None = None  # required iff origin == "manual"
    at: UtcDatetime


class SessionStore(Protocol):
    """SQLite (WAL) via aiosqlite. Raises SeviError(STORE_FAILED). Records stored as JSON TEXT."""

    async def put_session(self, rec: SessionRecord) -> None: ...  # upsert by session_id
    async def get_session(self, session_id: SessionId) -> SessionRecord | None: ...
    async def append_trial(self, trial: TrialRecord) -> None: ...
    async def get_by_report_token(self, token_sha256: Sha256Hex) -> SessionRecord | None: ...
    async def list_due_for_retention(self, now: datetime) -> list[SessionId]: ...
    async def delete_session(self, session_id: SessionId) -> None: ...  # cascades
    async def strip_study_identifiers(self, session_id: SessionId) -> None: ...
    async def twilio_status_upsert(self, call_sid: str, status: str, seq: int) -> bool: ...
    async def idempotency_get(self, scope: str, key: str) -> str | None: ...
    async def idempotency_put(self, scope: str, key: str, value: str, ttl_s: int) -> None: ...
    async def allowlist_get(self, participant_id: str) -> AllowlistEntry | None: ...
    async def allowlist_put(self, entry: AllowlistEntry) -> None: ...
    async def allowlist_list(self) -> list[AllowlistEntry]: ...
    async def ledger_add(
        self, key_id: KeyId, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0
    ) -> LedgerRow: ...
    async def ledger_get(self, key_id: KeyId) -> LedgerRow: ...
    async def ip_bucket_hit(self, ip_hmac: str, window_s: int) -> int: ...  # count incl. this hit
    async def list_study_sessions(self) -> list[SessionRecord]: ...
    async def trial_minutes_add(self, call_sid: str, minutes: int) -> float: ...
    async def trial_minutes_total(self) -> float: ...
    async def ledger_import(self, events: list[LedgerEvent]) -> int: ...  # idempotent on event_id
    async def ledger_bucket_used(self, key_id: KeyId, bucket: str) -> float: ...
    async def compact(self) -> None: ...  # wal_checkpoint(TRUNCATE) + VACUUM


# ---------- key semaphore ----------
PermitPurpose = Literal["stream", "rest", "tts"]


class Permit(Protocol):
    key_id: KeyId
    purpose: PermitPurpose


class KeySemaphore(Protocol):
    def acquire(
        self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float
    ) -> AbstractAsyncContextManager[Permit]:
        """A voice slot IS a 'stream' permit (R-14). 'tts' is never granted while a 'stream'
        permit is held on the same key. Raises SeviError(KEY_SEMAPHORE_TIMEOUT)."""
        ...


# ---------- telemetry ----------
class Telemetry(Protocol):
    def emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        """Non-blocking, never raises, applies redaction (§10.5)."""
        ...

    def span(self, name: SpanName, **fields: Any) -> AbstractContextManager[None]:
        """Emits span.start / span.end with dur_ms from Clock.mono_ms()."""
        ...

    def mark_span(
        self, name: SpanName, start_mono_ms: float, end_mono_ms: float, **fields: Any
    ) -> None:
        """Emit a span whose endpoints were observed asynchronously (e.g. turn.total)."""
        ...
