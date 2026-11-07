"""Fake Timbre TTS endpoint (contracts §11.8, B spec §12.1).

An `httpx.MockTransport` handler for `POST .../api/v1/tts/inference`: plug it into the
`httpx.AsyncClient` the adapter receives (`httpx.AsyncClient(transport=fake.transport())`).
Returns a 440 Hz tone shaped by the request's `audio_config` (raw 8 kHz mu-law for
`container: mulaw` or `raw`, a RIFF WAV otherwise) or a fixture.

Knobs: `latency_ms`, `status` (one int, or a list consumed one per request, the last one
repeating), `riff` (answer a mu-law request with a WAV, VT-TMB-01 outcome b), `empty` (200 with
an empty body), `fixture` (exact bytes returned on 200). Records every request body.
"""

from __future__ import annotations

import asyncio
import io
import json
import wave
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np

from sevi.core.dsp import ulaw_encode

TTS_PATH = "/api/v1/tts/inference"


def tone_pcm16(
    sample_rate_hz: int, ms: int = 500, freq_hz: float = 440.0
) -> np.ndarray[Any, np.dtype[np.int16]]:
    n = int(sample_rate_hz * ms / 1000)
    t = np.arange(n) / sample_rate_hz
    return (0.3 * 32767 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)


def wav_bytes(pcm: np.ndarray[Any, np.dtype[np.int16]], sample_rate_hz: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


class FakeTimbre:
    def __init__(
        self,
        *,
        latency_ms: int = 0,
        status: int | Sequence[int] = 200,
        riff: bool = False,
        empty: bool = False,
        fixture: bytes | None = None,
    ) -> None:
        self.latency_ms = latency_ms
        self._status = [status] if isinstance(status, int) else list(status)
        self.riff = riff
        self.empty = empty
        self.fixture = fixture
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def _next_status(self) -> int:
        return self._status.pop(0) if len(self._status) > 1 else self._status[0]

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith(TTS_PATH):
            return httpx.Response(404, json={"detail": "not found"})
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000.0)
        status = self._next_status()
        if status != 200:
            return httpx.Response(
                status,
                json={"success": False, "message": f"fake timbre {status}", "status_code": status},
            )
        if self.empty:
            return httpx.Response(200, content=b"", headers={"content-type": "audio/basic"})
        if self.fixture is not None:
            return httpx.Response(200, content=self.fixture)
        cfg = body.get("audio_config") or {}
        mulaw = cfg.get("container") in ("mulaw", "raw") or cfg.get("encoding") == "pcm_mulaw"
        if mulaw and not self.riff:
            data = ulaw_encode(tone_pcm16(8000))
            return httpx.Response(200, content=data, headers={"content-type": "audio/basic"})
        rate = 8000 if mulaw else int(cfg.get("sample_rate", 48000))
        return httpx.Response(
            200, content=wav_bytes(tone_pcm16(rate), rate), headers={"content-type": "audio/wav"}
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
