"""U-TR-01 parsing round trips and outbound encoding (B spec §6.1, contracts §9.4)."""

from __future__ import annotations

import base64
import json

import pytest

from sevi.contracts.adapters import TDtmf, TMark, TMedia, TStart, TStop, TUi
from sevi.contracts.enums import SessionState
from sevi.contracts.wire import UiPhase
from sevi.transport.base import (
    WireParseError,
    encode_clear,
    encode_mark,
    encode_media,
    encode_ui,
    parse_client_message,
    stream_sid_for,
)

SID = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b"
P160 = base64.b64encode(bytes(range(160))).decode()


def test_stream_sid_for() -> None:
    assert stream_sid_for(SID) == "WB01933b6e8f1a"


def test_U_TR_01_contract_examples_parse() -> None:
    assert parse_client_message(
        '{"event":"connected","protocol":"KJWeb","version":"1.0.0"}', 1
    ) == ("connected")
    start = parse_client_message(
        json.dumps(
            {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "streamSid": "WB01933b6e8f1a",
                    "callSid": None,
                    "customParameters": {"session_id": SID, "token": "t" * 22},
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                },
            }
        ),
        1.0,
    )
    assert isinstance(start, TStart)
    assert start.custom["session_id"] == SID
    assert start.call_sid is None
    media = parse_client_message(
        json.dumps(
            {
                "event": "media",
                "sequenceNumber": "42",
                "media": {"track": "inbound", "chunk": "41", "timestamp": "820", "payload": P160},
            }
        ),
        900.0,
    )
    assert isinstance(media, TMedia)
    assert (media.seq, media.ts_ms, media.ulaw, media.received_mono_ms) == (
        42,
        820,
        bytes(range(160)),
        900.0,
    )
    mark = parse_client_message(
        '{"event":"mark","sequenceNumber":"43","mark":{"name":"s009r0"}}', 5
    )
    assert isinstance(mark, TMark)
    assert mark.name == "s009r0"
    assert not mark.cleared
    dtmf = parse_client_message(
        '{"event":"dtmf","sequenceNumber":"44","dtmf":{"track":"inbound","digit":"4"}}', 5
    )
    assert isinstance(dtmf, TDtmf)
    assert dtmf.key == "4"
    ui = parse_client_message(
        '{"event":"kj","sequenceNumber":"45","kj":{"type":"consent_given"}}', 5
    )
    assert isinstance(ui, TUi)
    assert ui.event.type == "consent_given"
    assert parse_client_message('{"event":"kj","sequenceNumber":"46","kj":{"type":"ping"}}', 5) == (
        "ping"
    )
    assert isinstance(parse_client_message('{"event":"stop","sequenceNumber":"46"}', 5), TStop)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"event":"media","sequenceNumber":42,"media":{"payload":"AA=="}}',  # numeric, not string
        '{"event":"media","sequenceNumber":"4a","media":{"payload":"AA=="}}',
        '{"event":"media","sequenceNumber":"1","media":{"timestamp":20,"payload":"AA=="}}',
        '{"event":"media","sequenceNumber":"1","media":{"chunk":"-1","payload":"AA=="}}',
        '{"event":"media","sequenceNumber":"1","media":{"payload":""}}',
        '{"event":"media","sequenceNumber":"1","media":{"payload":"!!"}}',
        '{"event":"dtmf","sequenceNumber":"1","dtmf":{"digit":"A"}}',
        '{"event":"kj","sequenceNumber":"1","kj":{"type":"hack"}}',
        '{"event":"mark","sequenceNumber":"1","mark":{}}',
        '{"event":"stop"}',
    ],
)
def test_U_TR_01_bad_shapes_raise(raw: str) -> None:
    with pytest.raises(WireParseError):
        parse_client_message(raw, 0.0)


def test_U_TR_01_payload_size_bounds() -> None:
    ok = base64.b64encode(b"\xff" * 1600).decode()
    big = base64.b64encode(b"\xff" * 1601).decode()
    msg = '{"event":"media","sequenceNumber":"1","media":{"payload":"%s"}}'
    assert isinstance(parse_client_message(msg % ok, 0), TMedia)
    with pytest.raises(WireParseError):
        parse_client_message(msg % big, 0)


def test_U_TR_03_unknown_event_is_none() -> None:
    assert parse_client_message('{"event":"bogus"}', 0) is None


def test_encode_compact_round_trip() -> None:
    s = stream_sid_for(SID)
    m = encode_media(s, b"\x01\x02")
    assert " " not in m
    assert json.loads(m) == {"event": "media", "streamSid": s, "media": {"payload": "AQI="}}
    assert json.loads(encode_mark(s, "p0003")) == {
        "event": "mark",
        "streamSid": s,
        "mark": {"name": "p0003"},
    }
    assert json.loads(encode_clear(s)) == {"event": "clear", "streamSid": s}
    ui = json.loads(encode_ui(s, UiPhase(state=SessionState.TEST, kind=None, scored_index=4)))
    assert ui["event"] == "kj"
    assert ui["kj"]["type"] == "phase"
    assert ui["kj"]["total"] == 24
