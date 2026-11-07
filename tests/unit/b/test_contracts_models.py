"""Validators and behaviour in B-lead contracts files (contracts §4.8, §4.9, §4.12, §5.2, §9)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from sevi.contracts import ALL_MODELS, PENDING_MODELS
from sevi.contracts.adapters import FrameTag, LedgerEvent, SttTranscript, TransportEvent
from sevi.contracts.enums import ConsentMethod, ConsentOutcome, ConsentScript, Lang, ParticipantType
from sevi.contracts.errors import (
    HTTP_STATUS,
    BudgetExceeded,
    ErrorCode,
    ErrorEnvelope,
    SeviError,
    SttError,
    UpstreamError,
)
from sevi.contracts.session import ConsentRecord
from sevi.contracts.telemetry import EventName, SpanName, TelemetryEvent
from sevi.contracts.wire import (
    CreateSessionResponse,
    DialRequest,
    ServerUiMessage,
    SwitchRequest,
    UiPhase,
)

SID = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b"
NOW = datetime(2026, 11, 7, 7, 30, tzinfo=UTC)
H64 = "a" * 64


def _consent(**kw: Any) -> ConsentRecord:
    base: dict[str, Any] = {
        "session_id": SID,
        "script": ConsentScript.PUBLIC_V1,
        "script_sha256": H64,
        "lang": Lang.TA,
        "outcome": ConsentOutcome.GIVEN,
        "method": ConsentMethod.CLICK,
        "repeats": 0,
        "at": NOW,
    }
    base.update(kw)
    return ConsentRecord(**base)


def test_consent_study_form_id_iff_study_script() -> None:
    _consent()
    _consent(script=ConsentScript.STUDY_V1, study_form_id="SF-001")
    with pytest.raises(ValidationError):
        _consent(script=ConsentScript.STUDY_V1)
    with pytest.raises(ValidationError):
        _consent(study_form_id="SF-001")


def test_consent_repeats_bounded() -> None:
    with pytest.raises(ValidationError):
        _consent(repeats=2)


def test_dial_request_study_fields_rule() -> None:
    DialRequest(participant_id="P-01", participant_type=ParticipantType.PILOT)
    DialRequest(
        participant_id="P-01",
        participant_type=ParticipantType.STUDY,
        pseudonym="S-001",
        age_band="26-35",
        study_form_id="SF-001",
    )
    with pytest.raises(ValidationError):
        DialRequest(participant_id="P-01", participant_type=ParticipantType.STUDY)
    with pytest.raises(ValidationError):
        DialRequest(
            participant_id="P-01", participant_type=ParticipantType.PILOT, pseudonym="S-001"
        )
    with pytest.raises(ValidationError):
        DialRequest(participant_id="P-01", participant_type=ParticipantType.PUBLIC)


def test_ws_token_pattern() -> None:
    ok = {
        "session_id": SID,
        "status": "ADMITTED",
        "ws_url": f"wss://x/web/stream?sid={SID}",
        "ws_token": "A" * 22,
        "expires_in_s": 30,
    }
    CreateSessionResponse(**ok)
    with pytest.raises(ValidationError):
        CreateSessionResponse(**{**ok, "ws_token": "A" * 21})


def test_switch_reload_norms_is_out_of_scope() -> None:
    SwitchRequest(name="friends", value="ALLOWED")
    with pytest.raises(ValidationError):
        SwitchRequest(name="reload_norms", value=True)


def test_server_ui_message_discriminates_on_type() -> None:
    ta: TypeAdapter[Any] = TypeAdapter(ServerUiMessage)
    msg = ta.validate_python(
        {"type": "phase", "state": "TEST", "kind": "SCORED", "scored_index": 4}
    )
    assert isinstance(msg, UiPhase)
    assert msg.total == 24
    with pytest.raises(ValidationError):
        ta.validate_python({"type": "phase", "state": "TEST", "kind": None, "scored_index": 25})
    with pytest.raises(ValidationError):
        ta.validate_python({"type": "unknown"})


def test_transport_event_union_and_bytes_base64_roundtrip() -> None:
    ta: TypeAdapter[Any] = TypeAdapter(TransportEvent)
    ev = ta.validate_python({"kind": "dtmf", "key": "4", "received_mono_ms": 1.0})
    assert ev.key == "4"
    media = ta.validate_json(
        '{"kind":"media","ulaw":"/w==","seq":1,"ts_ms":20,"received_mono_ms":5.0}'
    )
    assert media.ulaw == b"\xff"


def test_stt_transcript_binds_frame_tag() -> None:
    t = SttTranscript(
        text="ondru",
        segment_id=None,
        segment_index=0,
        audio_duration_ms=300,
        latency_ms=200,
        detected_language_code="ta-IN",
        received_mono_ms=10.0,
        bound=FrameTag(presentation_id=3, window_index=1),
    )
    assert t.bound is not None
    assert t.bound.replay == 0
    with pytest.raises(ValidationError):
        FrameTag(presentation_id=0, window_index=1)


def test_ledger_event_id_is_uuid4_hex() -> None:
    LedgerEvent(
        event_id="0" * 32, key_id="A", bucket="OPS", purpose="dev", window_id="W1",
        origin="server", at=NOW,
    )  # fmt: skip
    with pytest.raises(ValidationError):
        LedgerEvent(
            event_id="xyz", key_id="A", bucket="OPS", purpose="dev", window_id="W1",
            origin="server", at=NOW,
        )  # fmt: skip


def test_telemetry_event_catalogue_and_example_line() -> None:
    assert EventName.SESSION_CREATED.value == "session.created"
    assert EventName.ERROR.value == "error"
    assert SpanName.TURN_TOTAL.value == "turn.total"
    assert "norms.reloaded" not in {e.value for e in EventName}
    line = (
        '{"ts":"2026-11-06T07:31:12.402Z","t_mono_ms":91234.5,"event":"span.end","level":"INFO",'
        '"service":"app","session_id":"01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b","request_id":null,'
        '"channel":"browser","mode":"voice","presentation_id":9,"scored_index":4,'
        '"span":"turn.total","dur_ms":1184.0,"error_code":null,'
        '"fields":{"close_reason":"THREE_DIGITS"}}'
    )
    ev = TelemetryEvent.model_validate_json(line)
    assert ev.event is EventName.SPAN_END
    with pytest.raises(ValidationError):
        TelemetryEvent.model_validate_json(line.replace('"scored_index":4', '"scored_index":25'))
    with pytest.raises(ValidationError):
        TelemetryEvent.model_validate_json(line.replace('"close_reason":"THREE_DIGITS"', '"x":[1]'))


def test_error_codes_value_equals_name() -> None:
    assert all(c.name == c.value for c in ErrorCode)
    assert "NORMS_BANK_MISMATCH" not in ErrorCode.__members__
    assert HTTP_STATUS[ErrorCode.ADMISSION_WAIT] == 503


def test_error_envelope_example() -> None:
    raw = (
        '{"error":{"code":"ADMISSION_WAIT","message":"Waiting room: capacity reached",'
        '"retryable":true,"retry_after_s":60,"request_id":"6f1c","details":{"ahead":12}}}'
    )
    env = ErrorEnvelope.model_validate_json(raw)
    assert env.error.details == {"ahead": 12}


def test_sevi_error_defaults_and_body() -> None:
    err = SttError(ErrorCode.STT_RATE_LIMITED, "busy", details={"n": 1, "obj": object()})
    assert isinstance(err, UpstreamError)
    assert err.retryable is True
    body = err.to_body(request_id="r1")
    assert body.details == {"n": 1}
    assert SeviError(ErrorCode.STT_AUTH).retryable is False


def test_budget_exceeded_is_credit_stop_never_retryable() -> None:
    err = BudgetExceeded("key A budget reached")
    assert err.code is ErrorCode.CREDIT_STOP
    assert err.retryable is False


def test_session_record_pending_until_a_models_land() -> None:
    names = {f"{m.__module__}.{m.__qualname__}" for m in ALL_MODELS}
    for pending in PENDING_MODELS:
        assert pending not in names
    assert "sevi.contracts.session.ConsentRecord" in names
