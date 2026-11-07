"""Telemetry event model, event-name and span-name catalogues (contracts §4.9, §10.2, §10.3).

Scope (BUILD-BRIEF reduced scope): `norms.reloaded` is omitted (norms are cut).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from sevi.contracts.enums import Channel, Mode
from sevi.contracts.types import NonNegMs, SessionId, SeviModel, UtcDatetime


class EventName(StrEnum):
    """Exactly the §10.2 catalogue."""

    SESSION_CREATED = "session.created"
    SESSION_QUEUED = "session.queued"
    SESSION_ADMITTED = "session.admitted"
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    SESSION_CRASH = "session.crash"
    FSM_TRANSITION = "fsm.transition"
    FSM_UNEXPECTED_EVENT = "fsm.unexpected_event"
    CONSENT_PROMPTED = "consent.prompted"
    CONSENT_GIVEN = "consent.given"
    CONSENT_DECLINED = "consent.declined"
    CONSENT_NO_REPLY = "consent.no_reply"
    WS_ACCEPTED = "ws.accepted"
    WS_REJECTED = "ws.rejected"
    WS_CLOSED = "ws.closed"
    WIRE_UNKNOWN_MESSAGE = "wire.unknown_message"
    TWILIO_DIAL = "twilio.dial"
    TWILIO_STATUS = "twilio.status"
    TWILIO_DELETED = "twilio.deleted"
    TWILIO_STEP = "twilio.step"
    AUDIO_SERVED = "audio.served"
    PROMPT_SENT = "prompt.sent"
    MARK_RECEIVED = "mark.received"
    MARK_LOST = "mark.lost"
    PRESENTATION_BEGIN = "presentation.begin"
    PRESENTATION_SENT = "presentation.sent"
    ANSWER_WINDOW_OPEN = "answer.window_open"
    ANSWER_WINDOW_CLOSED = "answer.window_closed"
    ANSWER_REASK = "answer.reask"
    ANSWER_REPLAY = "answer.replay"
    CAPTURE_FLOOR = "capture.floor"
    CAPTURE_SPEECH = "capture.speech"
    CAPTURE_EARLY_ANSWER = "capture.early_answer"
    CAPTURE_MISS = "capture.miss"
    CAPTURE_ECHO_MODE = "capture.echo_mode"
    STT_OPEN = "stt.open"
    STT_CONNECTED = "stt.connected"
    STT_CONFIG_UNVERIFIED = "stt.config_unverified"
    STT_CONFIG_MISMATCH = "stt.config_mismatch"
    STT_SPEECH_START = "stt.speech_start"
    STT_PROCESSING = "stt.processing"
    STT_TRANSCRIPT = "stt.transcript"
    STT_COMMIT = "stt.commit"
    STT_STALE_SEGMENT = "stt.stale_segment"
    STT_ERROR = "stt.error"
    STT_CLOSED = "stt.closed"
    STT_RECONNECT = "stt.reconnect"
    DTMF_RECEIVED = "dtmf.received"
    DTMF_IGNORED = "dtmf.ignored"
    SCORE_RESULT = "score.result"
    PROCEDURE_RECORD = "procedure.record"
    PROCEDURE_LOUD_CHECK = "procedure.loud_check"
    PROCEDURE_TERMINAL = "procedure.terminal"
    MODE_SWITCHED = "mode.switched"
    RESULT_DECIDED = "result.decided"
    EVON_REQUEST = "evon.request"
    EVON_RESPONSE = "evon.response"
    EVON_GUARD_REJECTED = "evon.guard_rejected"
    EVON_TEMPLATE = "evon.template"
    EVON_HEALTH = "evon.health"
    TIMBRE_REQUEST = "timbre.request"
    TIMBRE_RESPONSE = "timbre.response"
    GNANI_REST_CALL = "gnani.rest_call"
    BREAKER_OPEN = "breaker.open"
    BREAKER_CLOSED = "breaker.closed"
    SEMAPHORE_WAIT = "semaphore.wait"
    SEMAPHORE_TIMEOUT = "semaphore.timeout"
    ADMISSION_LEVEL = "admission.level"
    ADMISSION_REJECTED = "admission.rejected"
    LEDGER_UPDATE = "ledger.update"
    LEDGER_STOP = "ledger.stop"
    LEDGER_BUDGET_STOP = "ledger.budget_stop"  # 0.2.0 (#52): a session lost voice to the budget
    LEDGER_RECONCILED = "ledger.reconciled"
    STARTUP_STUB_FACTORY = "startup.stub_factory"
    ADMISSION_NO_CLIENT_IP = "admission.no_client_ip"
    WS_RX_IDLE = "ws.rx_idle"
    WS_SEND_TIMEOUT = "ws.send_timeout"
    PLAYER_SENT = "player.sent"
    MIC_SETTINGS = "mic.settings"
    UPLINK_DROP_GATED = "uplink.drop_gated"
    RETENTION_COMPACT = "retention.compact"
    RETENTION_DELETED = "retention.deleted"
    RETENTION_STRIPPED = "retention.stripped"
    READYZ_STATUS = "readyz.status"
    SWITCH_CHANGED = "switch.changed"
    UPLINK_GAP = "uplink.gap"
    UPLINK_BACKLOG = "uplink.backlog"
    SPAN_START = "span.start"
    SPAN_END = "span.end"
    ERROR = "error"


class SpanName(StrEnum):
    """Exactly the §10.3 latency hops."""

    TURN_TOTAL = "turn.total"
    STT_ENDPOINT = "stt.endpoint"
    STT_COMMIT = "stt.commit"
    SCORE_NORMALISE = "score.normalise"
    PROCEDURE_STEP = "procedure.step"
    STIM_RENDER = "stim.render"
    TX_FIRST_BYTE = "tx.first_byte"
    MARK_RTT = "mark.rtt"
    GATE_DURATION = "gate.duration"
    PRISMA_OPEN = "prisma.open"
    EVON_TURN = "evon.turn"
    TIMBRE_TTS = "timbre.tts"
    HTTP_REQUEST = "http.request"


FieldValue = str | int | float | bool | None


class TelemetryEvent(SeviModel):
    ts: UtcDatetime
    t_mono_ms: NonNegMs
    event: EventName
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    service: Literal["app", "cal", "bench", "cli"] = "app"
    session_id: SessionId | None = None
    request_id: str | None = None
    channel: Channel | None = None
    mode: Mode | None = None
    presentation_id: int | None = Field(None, ge=1)
    scored_index: int | None = Field(None, ge=1, le=24)
    span: str | None = None  # SpanName value when event is span.start / span.end
    dur_ms: float | None = Field(None, ge=0)
    error_code: str | None = None  # ErrorCode value
    fields: dict[str, FieldValue] = Field(default_factory=dict)  # flat, redacted (§10.5)
