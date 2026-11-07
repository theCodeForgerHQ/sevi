"""fake_prisma over a real socket (contracts §11.8): scripted endpointing, commit, knobs."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from sevi.fakes.fake_prisma import FakePrisma, Rule, frame_dbfs, silent_frame, voiced_frame

HDRS = [("x-sample-rate", "8000"), ("language_code", "ta-IN,en-IN"), ("format", "transcribe")]


async def _recv_json(ws: ClientConnection, timeout: float = 2.0) -> dict[str, Any]:
    raw = await asyncio.wait_for(ws.recv(), timeout)
    assert isinstance(raw, str)
    return dict(json.loads(raw))


async def _until(ws: ClientConnection, kind: str) -> dict[str, Any]:
    while True:
        m = await _recv_json(ws)
        if m.get("type") == kind:
            return m


def test_voiced_detection_threshold() -> None:
    assert frame_dbfs(voiced_frame()) > -40.0
    assert frame_dbfs(silent_frame()) == -200.0


@pytest.mark.asyncio
async def test_connected_echoes_request_and_vad_endpoint_emits_transcript() -> None:
    async with FakePrisma([Rule("4 7 2", latency_ms=10)]) as fp:
        url = await fp.start()
        async with connect(url, additional_headers=HDRS) as ws:
            hello = await _recv_json(ws)
            assert hello["type"] == "connected"
            assert hello["config"]["sample_rate"] == 8000
            assert hello["config"]["language_candidates"] == ["ta", "en"]
            for _ in range(8):
                await ws.send(voiced_frame())
            for _ in range(5):  # 320 ms >= 300 ms of silence
                await ws.send(silent_frame())
            assert (await _until(ws, "processing"))["type"] == "processing"
            tr = await _until(ws, "transcript")
            assert tr["text"] == "4 7 2"
            assert tr["segment_index"] == 1
            assert tr["audio_duration_ms"] == 8 * 64
        conn = fp.connections[0]
        assert conn.headers["x-sample-rate"] == "8000"
        assert [o for o, _ in conn.frames[:3]] == [0.0, 64.0, 128.0]


@pytest.mark.asyncio
async def test_commit_with_pending_voice_emits_and_split_segments() -> None:
    fp = FakePrisma([Rule("4 7 2", latency_ms=5)], split_segments=True)
    url = await fp.start()
    try:
        async with connect(url, additional_headers=HDRS) as ws:
            await _recv_json(ws)
            for _ in range(8):
                await ws.send(voiced_frame())
            for _ in range(5):
                await ws.send(silent_frame())
            first = await _until(ws, "transcript")
            await ws.send(json.dumps({"type": "commit"}))
            second = await _until(ws, "transcript")
            assert (first["text"], second["text"]) == ("4", "7 2")
            assert (first["segment_index"], second["segment_index"]) == (1, 2)
            # commit with fresh voice and commit_emits_processing -> processing + transcript
            for _ in range(3):
                await ws.send(voiced_frame())
            await ws.send(json.dumps({"type": "commit"}))
            await _until(ws, "processing")
            assert (await _until(ws, "transcript"))["segment_index"] == 3
        assert len(fp.connections[0].commits) == 2
    finally:
        await fp.stop()


@pytest.mark.asyncio
async def test_http_status_rejects_upgrade() -> None:
    async with FakePrisma(http_status=429) as fp:
        url = await fp.start()
        with pytest.raises(InvalidStatus) as e:
            async with connect(url, additional_headers=HDRS):
                pass
        assert e.value.response.status_code == 429
        assert fp.rejected
        assert fp.rejected[0]["format"] == "transcribe"


@pytest.mark.asyncio
async def test_handshake_error_then_close_1008() -> None:
    async with FakePrisma(handshake_error="policy") as fp:
        url = await fp.start()
        async with connect(url, additional_headers=HDRS) as ws:
            assert (await _recv_json(ws))["type"] == "error"
            await asyncio.wait_for(ws.wait_closed(), 2)
            assert ws.close_code == 1008


@pytest.mark.asyncio
async def test_drop_after_frames_closes_with_code_and_garbage_json() -> None:
    async with FakePrisma(drop_after_frames=3, close_code=1011, garbage_json=True) as fp:
        url = await fp.start()
        async with connect(url, additional_headers=HDRS) as ws:
            await _recv_json(ws)  # connected
            assert await asyncio.wait_for(ws.recv(), 2) == "{not json"
            assert (await _recv_json(ws))["type"] == "vad_end"
            for _ in range(3):
                await ws.send(silent_frame())
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(ws.recv(), 2)
            assert ws.close_code == 1011


@pytest.mark.asyncio
async def test_omit_config_and_override() -> None:
    for fp, check in (
        (FakePrisma(omit_config=True), lambda m: "config" not in m),
        (
            FakePrisma(config_echo={"sample_rate": 16000}),
            lambda m: m["config"]["sample_rate"] == 16000,
        ),
    ):
        try:
            async with connect(await fp.start(), additional_headers=HDRS) as ws:
                assert check(await _recv_json(ws))
        finally:
            await fp.stop()
