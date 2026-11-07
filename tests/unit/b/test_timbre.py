"""Timbre adapter (B spec §5.5, §13.5 U-TMB-01..04) and the offline factory for A's render."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from sevi.adapters.timbre import TimbreClient, offline_client
from sevi.contracts.enums import KeyId, Lang
from sevi.contracts.errors import BudgetExceeded, ErrorCode, TtsError
from sevi.core.dsp import dbfs, pcm16_to_float, ulaw_decode
from sevi.fakes.fake_clock import FakeClock
from sevi.fakes.fake_timbre import FakeTimbre
from sevi.fakes.memory_ledger import MemoryLedger, MemorySemaphore
from sevi.fakes.null_telemetry import NullTelemetry

TEXT = "வணக்கம். இது ஒரு கேட்கும் திறன் சோதனை."


def _client(
    fake: FakeTimbre, *, ledger: MemoryLedger | None = None, cache: Path | None = None
) -> tuple[TimbreClient, MemoryLedger, MemorySemaphore]:
    led, sem = ledger or MemoryLedger(), MemorySemaphore()
    c = TimbreClient(
        {KeyId.A: "KA", KeyId.B: "KB"},
        sem,
        led,
        httpx.AsyncClient(transport=fake.transport()),
        FakeClock(),
        NullTelemetry(),
        cache_dir=cache,
        rng=random.Random(0),  # noqa: S311
    )
    return c, led, sem


@pytest.mark.asyncio
async def test_U_TMB_01_wav_body_exact() -> None:
    fake = FakeTimbre()
    c, led, sem = _client(fake)
    wav = await c.synthesize_wav(TEXT, Lang.TA, "Chetana", "timbre-v3.0")
    assert wav[:4] == b"RIFF"
    assert fake.requests[0] == {
        "text": TEXT,
        "voice": "Chetana",
        "model": "timbre-v3.0",
        "language": "ta-IN",
        "speed": 1.0,
        "audio_config": {
            "sample_rate": 48000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    assert fake.headers[0]["x-api-key-id"] == "KA"
    assert sem.taken == [(KeyId.A, "tts")]
    assert led.events[0].tts_chars == len(TEXT)
    await c.synthesize_wav("five", Lang.EN, "Chetana", "timbre-v3.0")
    assert fake.requests[1]["language"] == "en-IN"


@pytest.mark.parametrize("riff", [False, True])
@pytest.mark.asyncio
async def test_U_TMB_02_ulaw_riff_or_raw_then_normalised(riff: bool) -> None:
    fake = FakeTimbre(riff=riff)
    c, _, _ = _client(fake)
    ulaw = await c.synthesize_ulaw(TEXT, Lang.TA, "Chetana", "timbre-v3.0", KeyId.A)
    assert fake.requests[0]["audio_config"] == {"container": "mulaw"}
    x = pcm16_to_float(ulaw_decode(ulaw))
    assert abs(dbfs(x) - -20.0) <= 0.1
    assert len(ulaw) == pytest.approx(4000, abs=8)  # 500 ms at 8 kHz either way


@pytest.mark.asyncio
async def test_U_TMB_03_too_long_zero_http() -> None:
    fake = FakeTimbre()
    c, led, _ = _client(fake)
    with pytest.raises(TtsError) as e:
        await c.synthesize_ulaw("a" * 301, Lang.TA, "v", "m", KeyId.A)
    assert e.value.code is ErrorCode.TTS_TEXT_TOO_LONG
    assert fake.requests == []
    assert led.events == []


@pytest.mark.parametrize(
    ("status", "code", "attempts"),
    [
        (400, ErrorCode.TTS_BAD_REQUEST, 1),
        (402, ErrorCode.TTS_CREDITS, 1),
        (500, ErrorCode.TTS_FAILED, 6),
    ],
)
@pytest.mark.asyncio
async def test_U_TMB_04_error_mapping(status: int, code: ErrorCode, attempts: int) -> None:
    fake = FakeTimbre(status=status)
    c, led, _ = _client(fake)
    with pytest.raises(TtsError) as e:
        await c.synthesize_wav(TEXT, Lang.TA, "v", "m")
    assert e.value.code is code
    assert len(fake.requests) == attempts
    if status == 402:
        assert KeyId.A in led.exhausted


@pytest.mark.asyncio
async def test_U_TMB_04_empty_body_failed() -> None:
    c, _, _ = _client(FakeTimbre(empty=True))
    with pytest.raises(TtsError) as e:
        await c.synthesize_ulaw(TEXT, Lang.TA, "v", "m", KeyId.A)
    assert e.value.code is ErrorCode.TTS_FAILED


@pytest.mark.asyncio
async def test_budget_refused_before_request_and_cache_hit_unbilled(tmp_path: Path) -> None:
    fake = FakeTimbre()
    c, led, _ = _client(fake, ledger=MemoryLedger(budget=0.1), cache=tmp_path)  # 10 chars max
    with pytest.raises(BudgetExceeded):
        await c.synthesize_wav(TEXT, Lang.TA, "v", "m")
    assert fake.requests == []
    led.budget = 100.0
    a = await c.synthesize_wav(TEXT, Lang.TA, "v", "m")
    b = await c.synthesize_wav(TEXT, Lang.TA, "v", "m")
    assert a == b
    assert len(fake.requests) == 1
    assert len(led.events) == 1


@pytest.mark.asyncio
async def test_offline_client_for_calibration_render_is_ledger_guarded() -> None:
    fake = FakeTimbre()
    settings = SimpleNamespace(
        gnani=SimpleNamespace(key_a="KA", rest_base="https://api.vachana.ai")
    )
    led = MemoryLedger(budget=1.0)  # 100 chars
    client = offline_client(
        settings=settings,
        ledger=led,
        http=httpx.AsyncClient(transport=fake.transport()),
        clock=FakeClock(),
    )
    wav = await client.synthesize_wav("ஒன்று", Lang.TA, "Chetana", "timbre-v3.0")
    assert wav[:4] == b"RIFF"
    assert fake.headers[0]["x-api-key-id"] == "KA"
    assert led.events[0].purpose == "calibration"
    assert led.events[0].key_id is KeyId.A
    with pytest.raises(BudgetExceeded):
        await client.synthesize_wav("x" * 200, Lang.TA, "Chetana", "timbre-v3.0")
    assert len(fake.requests) == 1
    missing = offline_client(settings=SimpleNamespace(gnani=SimpleNamespace(key_a=None)))
    with pytest.raises(TtsError, match="KEY_A"):
        await missing.synthesize_wav("x", Lang.TA, "v", "m")
    assert math.isfinite(float(np.frombuffer(wav[44:], "<i2").std()))
    assert json.dumps(fake.requests[0], ensure_ascii=False)
