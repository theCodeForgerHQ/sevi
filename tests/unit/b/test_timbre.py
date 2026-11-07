"""Timbre adapter (B spec §5.5, §13.5 U-TMB-01..04) and the offline factory for A's render.
The ledger is the real one (SQLite on tmp_path)."""

from __future__ import annotations

import random
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from sevi.adapters.timbre import TimbreClient, offline_client
from sevi.app.settings import Settings
from sevi.contracts.enums import KeyId, Lang
from sevi.contracts.errors import BudgetExceeded, ErrorCode, TtsError
from sevi.core.dsp import dbfs, pcm16_to_float, ulaw_decode
from sevi.fakes.clock import FakeClock
from sevi.fakes.fake_timbre import FakeTimbre
from sevi.fakes.ledger_kit import LedgerKit, open_ledger
from sevi.fakes.null_telemetry import NullTelemetry
from sevi.fakes.recording_deps import RecordingSemaphore

TEXT = "வணக்கம். இது ஒரு கேட்கும் திறன் சோதனை."


@pytest_asyncio.fixture
async def kit(tmp_path: Path) -> AsyncIterator[LedgerKit]:
    k = await open_ledger(tmp_path / "ledger.sqlite")
    yield k
    await k.close()


def _client(
    fake: FakeTimbre, kit: LedgerKit, *, cache: Path | None = None
) -> tuple[TimbreClient, LedgerKit, RecordingSemaphore]:
    sem = RecordingSemaphore()
    c = TimbreClient(
        {KeyId.A: "KA", KeyId.B: "KB"},
        sem,
        kit.ledger,
        httpx.AsyncClient(transport=fake.transport()),
        FakeClock(),
        NullTelemetry(),
        cache_dir=cache,
        rng=random.Random(0),  # noqa: S311
    )
    return c, kit, sem


@pytest.mark.asyncio
async def test_U_TMB_01_wav_body_exact(kit: LedgerKit) -> None:
    fake = FakeTimbre()
    c, led, sem = _client(fake, kit)
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
    assert (await led.events())[0].tts_chars == len(TEXT)
    await c.synthesize_wav("five", Lang.EN, "Chetana", "timbre-v3.0")
    assert fake.requests[1]["language"] == "en-IN"


@pytest.mark.parametrize("riff", [False, True])
@pytest.mark.asyncio
async def test_U_TMB_02_ulaw_riff_or_raw_then_normalised(riff: bool, kit: LedgerKit) -> None:
    fake = FakeTimbre(riff=riff)
    c, _, _ = _client(fake, kit)
    ulaw = await c.synthesize_ulaw(TEXT, Lang.TA, "Chetana", "timbre-v3.0", KeyId.A)
    assert fake.requests[0]["audio_config"] == {"container": "mulaw"}
    x = pcm16_to_float(ulaw_decode(ulaw))
    assert abs(dbfs(x) - -20.0) <= 0.1
    assert len(ulaw) == pytest.approx(4000, abs=8)  # 500 ms at 8 kHz either way


@pytest.mark.asyncio
async def test_U_TMB_03_too_long_zero_http(kit: LedgerKit) -> None:
    fake = FakeTimbre()
    c, led, _ = _client(fake, kit)
    with pytest.raises(TtsError) as e:
        await c.synthesize_ulaw("a" * 301, Lang.TA, "v", "m", KeyId.A)
    assert e.value.code is ErrorCode.TTS_TEXT_TOO_LONG
    assert fake.requests == []
    assert await led.events() == []


@pytest.mark.parametrize(
    ("status", "code", "attempts"),
    [
        (400, ErrorCode.TTS_BAD_REQUEST, 1),
        (402, ErrorCode.TTS_CREDITS, 1),
        (500, ErrorCode.TTS_FAILED, 6),
    ],
)
@pytest.mark.asyncio
async def test_U_TMB_04_error_mapping(
    status: int, code: ErrorCode, attempts: int, kit: LedgerKit
) -> None:
    fake = FakeTimbre(status=status)
    c, led, _ = _client(fake, kit)
    with pytest.raises(TtsError) as e:
        await c.synthesize_wav(TEXT, Lang.TA, "v", "m")
    assert e.value.code is code
    assert len(fake.requests) == attempts
    if status == 402:
        assert await led.exhausted(KeyId.A)  # 402 persisted (awaited)


@pytest.mark.asyncio
async def test_U_TMB_04_empty_body_failed(kit: LedgerKit) -> None:
    c, _, _ = _client(FakeTimbre(empty=True), kit)
    with pytest.raises(TtsError) as e:
        await c.synthesize_ulaw(TEXT, Lang.TA, "v", "m", KeyId.A)
    assert e.value.code is ErrorCode.TTS_FAILED


@pytest.mark.asyncio
async def test_budget_refused_before_request_and_cache_hit_unbilled(tmp_path: Path) -> None:
    fake = FakeTimbre()
    small = await open_ledger(tmp_path / "small.sqlite", budget=0.1)  # 10 chars max
    c, _, _ = _client(fake, small, cache=tmp_path)
    with pytest.raises(BudgetExceeded):
        await c.synthesize_wav(TEXT, Lang.TA, "v", "m")
    assert fake.requests == []
    await small.close()
    big = await open_ledger(tmp_path / "big.sqlite", budget=100.0)
    c2, _, _ = _client(fake, big, cache=tmp_path)
    a = await c2.synthesize_wav(TEXT, Lang.TA, "v", "m")
    b = await c2.synthesize_wav(TEXT, Lang.TA, "v", "m")
    assert a == b
    assert len(fake.requests) == 1
    assert len(await big.events()) == 1
    await big.close()


def _offline_env(monkeypatch: pytest.MonkeyPatch, data_dir: Path, budget: float) -> None:
    monkeypatch.setenv("SEVI_GNANI__KEY_A", "dummy-key-a")
    monkeypatch.setenv("SEVI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SEVI_LEDGER__BUDGET_CREDITS_PER_KEY", str(budget))
    monkeypatch.setenv("SEVI_LEDGER__CREDITS_PER_TTS_CHAR", "0.25")


def _rows(db: Path, table: str) -> int:
    with sqlite3.connect(db) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608


@pytest.mark.asyncio
async def test_offline_client_real_sqlite_ledger_reserves_before_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """offline_client() with nothing injected: Settings from env, the real SqliteStore + Ledger
    at settings.sqlite_path. The reservation row exists when the HTTP request is made; the
    ledger_events row exists after it."""
    _offline_env(monkeypatch, tmp_path, budget=100.0)
    db = Settings().sqlite_path
    fake = FakeTimbre()
    seen: list[tuple[int, int]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((_rows(db, "ledger_reservations"), _rows(db, "ledger_events")))
        return await fake.handler(request)

    client = offline_client(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        wav = await client.synthesize_wav("ஒன்று", Lang.TA, "Chetana", "timbre-v3.0")
    finally:
        await client.aclose()
    assert wav[:4] == b"RIFF"
    assert fake.headers[0]["x-api-key-id"] == "dummy-key-a"
    assert seen == [(1, 0)]  # reserved before the call, nothing committed yet
    assert _rows(db, "ledger_reservations") == 0
    assert _rows(db, "ledger_events") == 1
    with sqlite3.connect(db) as con:
        key, purpose = con.execute("SELECT key_id, purpose FROM ledger_events").fetchone()
    assert (key, purpose) == ("A", "calibration")


@pytest.mark.asyncio
async def test_offline_client_budget_zero_makes_no_http_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline_env(monkeypatch, tmp_path, budget=0.0)
    fake = FakeTimbre()
    client = offline_client(http=httpx.AsyncClient(transport=fake.transport()))
    try:
        with pytest.raises(BudgetExceeded):
            await client.synthesize_wav("ஒன்று", Lang.TA, "Chetana", "timbre-v3.0")
    finally:
        await client.aclose()
    assert fake.requests == []
    assert _rows(Settings().sqlite_path, "ledger_reservations") == 0


@pytest.mark.asyncio
async def test_offline_client_without_key_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _offline_env(monkeypatch, tmp_path, budget=10.0)
    monkeypatch.setenv("SEVI_GNANI__KEY_A", "")
    client = offline_client(http=httpx.AsyncClient(transport=FakeTimbre().transport()))
    with pytest.raises(TtsError, match="KEY_A"):
        await client.synthesize_wav("x", Lang.TA, "v", "m")
    await client.aclose()
