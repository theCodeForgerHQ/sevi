"""VT-GN-01 credit metering (#64, contracts §00.1 (g), §14). NEVER run by CI or builder agents.

Gnani exposes no balance API, so a human reads the balance at app.gnani.ai/voice before and
after each test. Run ONE test at a time, Key B only, under infisical:

    SEVI_ALLOW_LIVE=1 infisical run --projectId <id> --env=dev --silent -- \
        uv run pytest -m live tests/live/b/test_vt_gn_01.py -k stream -s

Then read R1, run `-k rest`, read R2, run `-k tts`, read R3. Rates:
    stream STT credits/s = (R0 - R1) / 10, REST STT credits/s = (R1 - R2) / 10,
    TTS credits/char = (R2 - R3) / 100.

Self-contained on purpose (httpx + websockets, no adapters): the ledger refuses every call
while SEVI_LEDGER__CREDITS_PER_* are unset. Hard local caps, independent of the server:
- stream: exactly 156 frames x 64 ms (8 kHz, 1,024 B) = 9.98 s of audio, socket force-closed
  at 12 s wall clock;
- REST: one 10.0 s WAV, a single attempt (no retries);
- TTS: exactly 100 characters, a single attempt.
Any 402 fails the test immediately. Audio is a synthetic tone. Prints status codes and server
message types/shape only, never headers or keys.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import wave

import httpx
import numpy as np
import pytest
import websockets

WS_URL = "wss://api.vachana.ai/stt/v3/stream"
REST_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"

RATE = 8000
FRAME_BYTES = 1024  # 512 samples s16le
FRAME_S = 512 / RATE  # 64 ms
STREAM_FRAMES = 156  # 9.984 s
STREAM_WALL_CAP_S = 12.0
REST_SECONDS = 10.0
TTS_TEXT = ("ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது பத்து " * 3)[:100]


def _key_b() -> str:
    k = os.environ.get("SEVI_GNANI__KEY_B", "")
    if not k:
        pytest.skip("SEVI_GNANI__KEY_B not set (run under infisical)")
    return k


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (3000 * np.sin(2 * np.pi * 300 * t)).astype("<i2")


def _shape(obj: object) -> object:
    """Keys and types only, so nothing echoed back is printed verbatim."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shape(obj[0])] if obj else []
    return type(obj).__name__


def _check_402(status: int) -> None:
    if status == 402:
        pytest.fail("402 Insufficient credit: STOP all live work")


@pytest.mark.live
@pytest.mark.max_credits(30)
@pytest.mark.asyncio
async def test_VT_GN_01_stream_10s() -> None:
    pcm = _tone(STREAM_FRAMES * FRAME_S).tobytes()
    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    assert len(frames) == STREAM_FRAMES
    assert all(len(f) == FRAME_BYTES for f in frames)
    headers = {
        "x-api-key-id": _key_b(),
        "language_code": "ta-IN",
        "model": "prisma-v2.5",
        "format": "transcribe",
        "x-sample-rate": str(RATE),
    }
    t0 = time.monotonic()
    seen: list[object] = []
    close: tuple[int | None, str | None] = (None, None)

    async def run() -> None:
        nonlocal close
        try:
            async with websockets.connect(WS_URL, additional_headers=headers) as ws:
                seen.append(_shape(json.loads(await ws.recv())))

                async def reader() -> None:
                    async for m in ws:
                        if isinstance(m, str):
                            seen.append(_shape(json.loads(m)))

                rd = asyncio.create_task(reader())
                for f in frames:
                    await ws.send(f)
                    await asyncio.sleep(FRAME_S)
                await asyncio.sleep(0.5)
                rd.cancel()
                close = (ws.close_code, ws.close_reason)
        except websockets.InvalidStatus as e:
            _check_402(e.response.status_code)
            raise

    try:
        await asyncio.wait_for(run(), timeout=STREAM_WALL_CAP_S)
    finally:
        print(f"stream wall={time.monotonic() - t0:.2f}s close={close} messages={seen}")


@pytest.mark.live
@pytest.mark.max_credits(30)
@pytest.mark.asyncio
async def test_VT_GN_01_rest_10s() -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(_tone(REST_SECONDS).tobytes())
    data = {"language_code": "ta-IN", "model": "prisma-v2.5", "format": "transcribe"}
    async with httpx.AsyncClient(trust_env=False, timeout=30.0) as http:
        r = await http.post(
            REST_URL,
            headers={"X-API-Key-ID": _key_b()},
            data=data,
            files={"audio_file": ("clip.wav", buf.getvalue(), "audio/wav")},
        )
    _check_402(r.status_code)
    body = r.json() if "json" in r.headers.get("content-type", "") else None
    print(f"rest status={r.status_code} shape={_shape(body)}")
    assert r.status_code == 200


@pytest.mark.live
@pytest.mark.max_credits(30)
@pytest.mark.asyncio
async def test_VT_GN_01_tts_100_chars() -> None:
    assert len(TTS_TEXT) == 100
    body = {
        "text": TTS_TEXT,
        "voice": "Chetana",
        "model": "timbre-v2.5",
        "language": "ta-IN",
        "speed": 1.0,
        "audio_config": {
            "sample_rate": 8000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    async with httpx.AsyncClient(trust_env=False, timeout=30.0) as http:
        r = await http.post(TTS_URL, headers={"X-API-Key-ID": _key_b()}, json=body)
    _check_402(r.status_code)
    ct = r.headers.get("content-type", "")
    riff = r.content[:4] == b"RIFF"
    print(f"tts status={r.status_code} content-type={ct} bytes={len(r.content)} riff={riff}")
    assert r.status_code == 200
