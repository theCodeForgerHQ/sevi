"""Live verification of the Gnani adapters (B spec §13.12 VT-*). NEVER run by CI or builder
agents: `@pytest.mark.live` is skipped unless SEVI_ALLOW_LIVE=1, and keys come only from
`infisical run` (SEVI_GNANI__KEY_A). Each test is tiny and below the G5 caps (60 s STT,
200 TTS chars) and still goes through a budgeted ledger, so an overrun is refused, not spent.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from sevi.adapters.breaker import BreakerRegistry
from sevi.adapters.deps import SerialSemaphore
from sevi.adapters.prisma_rest import PrismaRestClient
from sevi.adapters.prisma_stream import PrismaOptions, PrismaStreamClient
from sevi.adapters.timbre import TimbreClient
from sevi.contracts.adapters import SttRestRequest, SttSessionConfig, SttTranscript, VadConfig
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.contracts.telemetry import EventName
from sevi.fakes.clock import ScaledClock
from sevi.fakes.ledger_kit import open_ledger
from sevi.fakes.null_telemetry import RecordingTelemetry


def _key() -> str:
    k = os.environ.get("SEVI_GNANI__KEY_A", "")
    if not k:
        pytest.skip("SEVI_GNANI__KEY_A not set (run under infisical)")
    return k


def _pcm(seconds: float) -> bytes:
    t = np.arange(int(8000 * seconds)) / 8000.0
    return (3000 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()


@pytest.mark.live
@pytest.mark.max_credits(5)
@pytest.mark.asyncio
async def test_VT_TMB_01_live_ulaw_form(tmp_path: Path) -> None:
    """One 6-char Timbre request: the mulaw form returns 8 kHz mu-law (or RIFF, outcome b)."""
    kit = await open_ledger(tmp_path / "l.sqlite", budget=5.0, credits_per_tts_char=0.25)
    led = kit.ledger
    async with httpx.AsyncClient(trust_env=False) as http:
        c = TimbreClient(
            {KeyId.A: _key()}, SerialSemaphore(), led, http, ScaledClock(1.0),
            RecordingTelemetry(), purpose="vt",
        )  # fmt: skip
        ulaw = await c.synthesize_ulaw("ஒன்று", Lang.TA, "Chetana", "timbre-v3.0", KeyId.A)
    assert len(ulaw) > 800
    assert sum(e.tts_chars for e in await kit.events()) <= 200


@pytest.mark.live
@pytest.mark.max_credits(5)
@pytest.mark.asyncio
async def test_VT_PRS_REST_01_live_two_seconds(tmp_path: Path) -> None:
    """One 2 s REST transcription (C§8.3 multipart); a success or an empty transcript passes."""
    kit = await open_ledger(tmp_path / "l.sqlite", budget=5.0, credits_per_stt_s=1.0)
    led = kit.ledger
    async with httpx.AsyncClient(trust_env=False) as http:
        c = PrismaRestClient(
            {KeyId.A: _key()}, SerialSemaphore(), led, http, ScaledClock(1.0),
            RecordingTelemetry(), cache_dir=None, purpose="vt",
        )  # fmt: skip
        r = await c.transcribe(
            SttRestRequest(
                wav_8k_pcm16=_pcm(2.0),
                lang=Lang.TA,
                model_version=ModelVersion.V3,
                format="transcribe",
                keyterms=(),
            ),
            retries=0,
        )
    assert isinstance(r.transcript, str)
    assert sum(e.stt_s for e in await kit.events()) <= 60


def _stream_pcm() -> bytes:
    """8 kHz s16le speech from SEVI_VT_AUDIO (raw PCM, e.g. Tamil digits through the phone
    chain) if set, else 3 s of tone."""
    path = os.environ.get("SEVI_VT_AUDIO", "")
    return Path(path).read_bytes() if path else _pcm(3.008)


@pytest.mark.live
@pytest.mark.max_credits(15)
@pytest.mark.asyncio
async def test_VT_PRS_01_live_stream_connected_echo(tmp_path: Path) -> None:
    """Open one v3 stream on Key A with ONE locale (live #64: "ta-IN,en-IN" is rejected),
    stream 8 kHz audio paced in real time (the adapter upsamples to 16 kHz, the only rate v3
    runs at), commit, and wait for a transcript. Checks the `connected` echo (C§8.2.3).
    Budget = one 60 s reservation at <= 0.25 credits/s."""
    kit = await open_ledger(tmp_path / "l.sqlite", budget=15.0, credits_per_stt_s=0.25)
    led = kit.ledger
    tel = RecordingTelemetry()
    c = PrismaStreamClient(
        PrismaOptions(),
        {KeyId.A: _key()},
        BreakerRegistry(ScaledClock(1.0), tel, alias_b=True),
        led,
        ScaledClock(1.0),
        tel,
        purpose="vt",
    )
    cfg = SttSessionConfig(
        key_id=KeyId.A,
        lang=Lang.TA,
        model_version=ModelVersion.V3,
        format="transcribe",
        keyterms=("ஒன்று", "onru", "one"),
        sample_rate_hz=8000,
        vad=VadConfig(),
        idle_timeout_s=30,
        request_id="vt-prs-01-s1",
    )
    s = await c.open(cfg)
    assert s.config_echo.get("sample_rate") == 16000
    transcripts: list[SttTranscript] = []

    async def read() -> None:
        async for ev in s.events():
            if isinstance(ev, SttTranscript):
                transcripts.append(ev)
                return

    reader = asyncio.create_task(read())
    try:
        pcm = _stream_pcm()[: 8000 * 2 * 20]  # hard cap: 20 s of audio
        for i in range(0, len(pcm) - 1023, 1024):
            s.send_frame(pcm[i : i + 1024], None)
            await asyncio.sleep(0.064)  # real time: one 8 kHz frame per 64 ms
        s.commit()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(reader), 10.0)
    finally:
        reader.cancel()
        await s.close()
    assert transcripts, "no transcript within 10 s of commit"
    assert sum(e.stt_s for e in await kit.events()) <= 60
    assert tel.of(EventName.STT_CONNECTED)
