"""Twilio-shaped message parsing and encoding shared by the browser transport (B spec §6.1,
contracts §9.4).

Inbound: `parse_client_message` turns one JSON text frame into a `TransportEvent`, the literals
"connected" / "ping", or None (unknown event, logged by the caller as `wire.unknown_message`).
A frame with bad JSON or a bad shape raises `WireParseError`.

Outbound: every message is `json.dumps(..., separators=(",", ":"))`.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Final, Literal

from pydantic import ValidationError

from sevi.contracts.adapters import TDtmf, TMark, TMedia, TransportEvent, TStart, TStop, TUi
from sevi.contracts.errors import ErrorCode, TransportError
from sevi.contracts.types import SessionId
from sevi.contracts.wire import BrowserUiEvent, ServerUiMessage

KNOWN_EVENTS: Final = frozenset({"connected", "start", "media", "mark", "dtmf", "kj", "stop"})
MEDIA_PAYLOAD_MIN_BYTES: Final = 1
MEDIA_PAYLOAD_MAX_BYTES: Final = 1600
DTMF_DIGITS: Final = "0123456789*#"

ParseResult = TransportEvent | Literal["connected", "ping"] | None


class WireParseError(TransportError):
    """A client frame that is not valid JSON or does not have the §9.4 shape.

    Transport-local exception (not a wire field): code TRANSPORT_PROTOCOL."""

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.TRANSPORT_PROTOCOL, message, retryable=False)


def stream_sid_for(session_id: SessionId | str) -> str:
    """'WB' + the first 12 hex chars of the session id without dashes."""
    return "WB" + str(session_id).replace("-", "")[:12]


def _dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _obj(v: Any, name: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise WireParseError(f"{name}: object required")
    return v


def _digits(v: Any, name: str, *, required: bool) -> int | None:
    """Twilio sends numeric fields as strings of decimal digits; anything else is rejected."""
    if v is None:
        if required:
            raise WireParseError(f"{name}: required")
        return None
    if not isinstance(v, str) or not v.isascii() or not v.isdigit():
        raise WireParseError(f"{name}: string of decimal digits required")
    return int(v)


def parse_client_message(raw: str, received_mono_ms: float) -> ParseResult:
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise WireParseError(f"bad json: {e}") from None
    msg = _obj(msg, "message")
    event = msg.get("event")
    if not isinstance(event, str):
        raise WireParseError("event: string required")
    if event not in KNOWN_EVENTS:
        return None
    if event == "connected":
        return "connected"
    seq = _digits(msg.get("sequenceNumber"), "sequenceNumber", required=True)
    assert seq is not None

    if event == "start":
        start = _obj(msg.get("start"), "start")
        stream_id = start.get("streamSid")
        call_sid = start.get("callSid")
        custom = _obj(start.get("customParameters", {}), "customParameters")
        if not isinstance(stream_id, str) or not stream_id:
            raise WireParseError("start.streamSid: string required")
        if call_sid is not None and not isinstance(call_sid, str):
            raise WireParseError("start.callSid: string or null required")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in custom.items()):
            raise WireParseError("start.customParameters: string values required")
        return TStart(stream_id=stream_id, call_sid=call_sid, custom=dict(custom))

    if event == "media":
        media = _obj(msg.get("media"), "media")
        _digits(media.get("chunk"), "media.chunk", required=False)
        ts = _digits(media.get("timestamp"), "media.timestamp", required=False)
        payload = media.get("payload")
        if not isinstance(payload, str):
            raise WireParseError("media.payload: string required")
        try:
            ulaw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise WireParseError("media.payload: bad base64") from None
        if not MEDIA_PAYLOAD_MIN_BYTES <= len(ulaw) <= MEDIA_PAYLOAD_MAX_BYTES:
            raise WireParseError(f"media.payload: {len(ulaw)} bytes outside 1..1600")
        return TMedia(ulaw=ulaw, seq=seq, ts_ms=ts, received_mono_ms=received_mono_ms)

    if event == "mark":
        mark = _obj(msg.get("mark"), "mark")
        name = mark.get("name")
        if not isinstance(name, str) or not name:
            raise WireParseError("mark.name: string required")
        return TMark(name=name, received_mono_ms=received_mono_ms, cleared=False)

    if event == "dtmf":
        dtmf = _obj(msg.get("dtmf"), "dtmf")
        digit = dtmf.get("digit")
        if not isinstance(digit, str) or len(digit) != 1 or digit not in DTMF_DIGITS:
            raise WireParseError("dtmf.digit: one of 0-9 * #")
        return TDtmf.model_validate({"key": digit, "received_mono_ms": received_mono_ms})

    if event == "kj":
        try:
            ui = BrowserUiEvent.model_validate(_obj(msg.get("kj"), "kj"))
        except ValidationError as e:
            raise WireParseError(f"kj: {e.error_count()} validation errors") from None
        if ui.type == "ping":
            return "ping"
        return TUi(event=ui)

    return TStop()  # event == "stop"


def encode_media(stream_sid: str, ulaw: bytes) -> str:
    payload = base64.b64encode(ulaw).decode("ascii")
    return _dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": payload}})


def encode_mark(stream_sid: str, name: str) -> str:
    return _dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


def encode_clear(stream_sid: str) -> str:
    return _dumps({"event": "clear", "streamSid": stream_sid})


def encode_ui(stream_sid: str, msg: ServerUiMessage) -> str:
    return _dumps({"event": "kj", "streamSid": stream_sid, "kj": msg.model_dump(mode="json")})
