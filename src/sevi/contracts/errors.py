"""Error codes, error envelope and exception hierarchy (contracts §4.12, §7).

Scope (BUILD-BRIEF reduced scope): `NORMS_BANK_MISMATCH` is omitted (norms are cut).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from sevi.contracts.types import SeviModel


class ErrorCode(StrEnum):
    """Exactly the §7 catalogue; value == name."""

    BAD_REQUEST = "BAD_REQUEST"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN_NETWORK = "FORBIDDEN_NETWORK"
    NOT_FOUND = "NOT_FOUND"
    REPORT_EXPIRED = "REPORT_EXPIRED"
    RATE_LIMITED_IP = "RATE_LIMITED_IP"
    ADMISSION_WAIT = "ADMISSION_WAIT"
    VOICE_UNAVAILABLE = "VOICE_UNAVAILABLE"
    LANG_UNAVAILABLE = "LANG_UNAVAILABLE"
    SESSION_STATE_CONFLICT = "SESSION_STATE_CONFLICT"
    SESSION_TOKEN_INVALID = "SESSION_TOKEN_INVALID"
    TRANSPORT_PROTOCOL = "TRANSPORT_PROTOCOL"
    TRANSPORT_CLOSED = "TRANSPORT_CLOSED"
    MARK_TIMEOUT = "MARK_TIMEOUT"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"
    SERVICE_DEGRADED = "SERVICE_DEGRADED"
    TWILIO_SIGNATURE_INVALID = "TWILIO_SIGNATURE_INVALID"
    TWILIO_DIAL_FAILED = "TWILIO_DIAL_FAILED"
    TWILIO_TRIAL_MINUTES_STOP = "TWILIO_TRIAL_MINUTES_STOP"
    TWILIO_REST_FAILED = "TWILIO_REST_FAILED"
    PHONE_DISABLED = "PHONE_DISABLED"
    ALLOWLIST_DENIED = "ALLOWLIST_DENIED"
    VOLUNTEER_CALLS_DISABLED = "VOLUNTEER_CALLS_DISABLED"
    TWILIO_TOKEN_INVALID = "TWILIO_TOKEN_INVALID"
    STT_AUTH = "STT_AUTH"
    STT_CREDITS = "STT_CREDITS"
    STT_RATE_LIMITED = "STT_RATE_LIMITED"
    STT_CONNECT_TIMEOUT = "STT_CONNECT_TIMEOUT"
    STT_HANDSHAKE_REJECTED = "STT_HANDSHAKE_REJECTED"
    STT_CONFIG_MISMATCH = "STT_CONFIG_MISMATCH"
    STT_DISCONNECTED = "STT_DISCONNECTED"
    STT_SESSION_LIMIT = "STT_SESSION_LIMIT"
    STT_UTTERANCE_ERROR = "STT_UTTERANCE_ERROR"
    STT_SCHEMA = "STT_SCHEMA"
    STT_BREAKER_OPEN = "STT_BREAKER_OPEN"
    STT_REST_FAILED = "STT_REST_FAILED"
    TTS_BAD_REQUEST = "TTS_BAD_REQUEST"
    TTS_TEXT_TOO_LONG = "TTS_TEXT_TOO_LONG"
    TTS_TIMEOUT = "TTS_TIMEOUT"
    TTS_FAILED = "TTS_FAILED"
    TTS_CREDITS = "TTS_CREDITS"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_SCHEMA_INVALID = "LLM_SCHEMA_INVALID"
    LLM_GUARD_REJECTED = "LLM_GUARD_REJECTED"
    LLM_BREAKER_OPEN = "LLM_BREAKER_OPEN"
    KEY_SEMAPHORE_TIMEOUT = "KEY_SEMAPHORE_TIMEOUT"
    CREDIT_STOP = "CREDIT_STOP"
    BANK_INTEGRITY = "BANK_INTEGRITY"
    BANK_HASH_REJECTED = "BANK_HASH_REJECTED"
    BANK_CONFIG_MISMATCH = "BANK_CONFIG_MISMATCH"
    PROCEDURE_PROTOCOL = "PROCEDURE_PROTOCOL"
    STORE_FAILED = "STORE_FAILED"
    CONFIG_INVALID = "CONFIG_INVALID"
    INTERNAL = "INTERNAL"


# HTTP status per §7 (codes without an HTTP status are internal-only).
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.VALIDATION_FAILED: 422,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN_NETWORK: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.REPORT_EXPIRED: 410,
    ErrorCode.RATE_LIMITED_IP: 429,
    ErrorCode.ADMISSION_WAIT: 503,
    ErrorCode.VOICE_UNAVAILABLE: 409,
    ErrorCode.LANG_UNAVAILABLE: 409,
    ErrorCode.SESSION_STATE_CONFLICT: 409,
    ErrorCode.SERVICE_DEGRADED: 503,
    ErrorCode.TWILIO_SIGNATURE_INVALID: 403,
    ErrorCode.TWILIO_DIAL_FAILED: 502,
    ErrorCode.TWILIO_TRIAL_MINUTES_STOP: 409,
    ErrorCode.PHONE_DISABLED: 409,
    ErrorCode.ALLOWLIST_DENIED: 403,
    ErrorCode.VOLUNTEER_CALLS_DISABLED: 403,
    ErrorCode.TWILIO_TOKEN_INVALID: 403,
    ErrorCode.CREDIT_STOP: 409,
    ErrorCode.STORE_FAILED: 500,
    ErrorCode.INTERNAL: 500,
}

# Browser WebSocket close codes (§9.4).
WS_CLOSE_CODE: dict[ErrorCode, int] = {
    ErrorCode.TRANSPORT_PROTOCOL: 4400,
    ErrorCode.SESSION_TOKEN_INVALID: 4401,
    ErrorCode.SESSION_STATE_CONFLICT: 4409,
    ErrorCode.RATE_LIMITED_IP: 4429,
    ErrorCode.INTERNAL: 4500,
}

# Codes whose §7 retry policy allows a retry; the default for SeviError.retryable.
RETRYABLE_CODES: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.RATE_LIMITED_IP,
        ErrorCode.ADMISSION_WAIT,
        ErrorCode.TWILIO_REST_FAILED,
        ErrorCode.STT_RATE_LIMITED,
        ErrorCode.STT_CONNECT_TIMEOUT,
        ErrorCode.STT_DISCONNECTED,
        ErrorCode.STT_SESSION_LIMIT,
        ErrorCode.STT_REST_FAILED,
        ErrorCode.TTS_TIMEOUT,
        ErrorCode.TTS_FAILED,
        ErrorCode.LLM_SCHEMA_INVALID,
        ErrorCode.LLM_GUARD_REJECTED,
    }
)

DetailValue = str | int | float | bool | None


class ErrorBody(SeviModel):
    code: ErrorCode
    message: str = Field(max_length=300)  # English, no PII, no secrets
    retryable: bool
    retry_after_s: int | None = None
    request_id: str | None = None
    details: dict[str, DetailValue] = Field(default_factory=dict)


class ErrorEnvelope(SeviModel):
    error: ErrorBody


class SeviError(Exception):
    """Base of every Sevi exception. Raise a subclass, never a bare SeviError, from adapters."""

    code: ErrorCode
    retryable: bool
    details: dict[str, Any]

    def __init__(
        self,
        code: ErrorCode,
        message: str = "",
        *,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or code.value)
        self.code = code
        self.message = message
        self.retryable = code in RETRYABLE_CODES if retryable is None else retryable
        self.details = dict(details or {})

    def to_body(self, request_id: str | None = None, retry_after_s: int | None = None) -> ErrorBody:
        """Envelope body. Only scalar details survive (the envelope is flat, §4.12)."""
        flat = {k: v for k, v in self.details.items() if isinstance(v, str | int | float | bool)}
        return ErrorBody(
            code=self.code,
            message=(self.message or self.code.value)[:300],
            retryable=self.retryable,
            retry_after_s=retry_after_s,
            request_id=request_id,
            details=flat,
        )


KJError = SeviError  # spec name (contracts §4.12)


class UpstreamError(SeviError):
    """Base for Gnani / Evon / Twilio errors."""


class SttError(UpstreamError): ...


class TtsError(UpstreamError): ...


class LlmError(UpstreamError): ...


class TwilioError(UpstreamError): ...


class BreakerOpenError(UpstreamError): ...


class BudgetExceeded(UpstreamError):
    """Raised by adapters/ledger.py BEFORE a Gnani call once a key's credit budget is reached
    (BUILD-BRIEF credit safety). Always code CREDIT_STOP, never retryable."""

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.CREDIT_STOP, message, retryable=False, details=details)


class TransportError(SeviError): ...


class BankError(SeviError):
    """BANK_HASH_REJECTED, BANK_INTEGRITY, BANK_CONFIG_MISMATCH."""


class ProcedureProtocolError(SeviError):
    """Programming error in the A<->B seam (PROCEDURE_PROTOCOL)."""


class AdmissionError(SeviError): ...
