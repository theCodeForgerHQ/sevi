"""Ledger hard budget under concurrency and across restarts (BUILD-BRIEF credit safety)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sevi.adapters.ledger import Ledger
from sevi.app.settings import LedgerSettings
from sevi.app.store import SqliteStore
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import BudgetExceeded
from sevi.contracts.telemetry import EventName, SpanName

PARAMS = LedgerSettings(
    budget_credits_per_key=100.0, credits_per_stt_s=1.0, credits_per_tts_char=0.01
)


class _Clock:
    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def mono_ms(self) -> float:
        return 0.0

    async def sleep_until(self, mono_ms: float) -> None:
        return None


class _Tel:
    def emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        pass

    @contextlib.contextmanager
    def span(self, name: SpanName, **fields: Any) -> Iterator[None]:
        yield

    def mark_span(self, name: SpanName, start: float, end: float, **fields: Any) -> None:
        pass


async def _open(path: Path, params: LedgerSettings = PARAMS) -> Ledger:
    store = await SqliteStore(path, clock=_Clock()).open()
    return Ledger(store, params, alias_b=False, clock=_Clock(), telemetry=_Tel())


@pytest.mark.asyncio
async def test_concurrent_reservations_cannot_overshoot(tmp_path: Path) -> None:
    """Three 'processes' (separate connections), 60 tasks each asking 7 credits: at most
    14 reservations (98 credits) can be granted for a 100-credit budget."""
    path = tmp_path / "sevi.sqlite"
    ledgers = [await _open(path) for _ in range(3)]
    granted = 0
    refused = 0

    async def call(led: Ledger) -> None:
        nonlocal granted, refused
        try:
            async with led.billed(KeyId.B, "dev", stt_s=7):
                granted += 1
                await asyncio.sleep(0)
        except BudgetExceeded:
            refused += 1

    await asyncio.gather(*(call(ledgers[i % 3]) for i in range(60)))
    assert granted == 14
    assert refused == 46
    used = await ledgers[0].used(KeyId.B)
    assert used == 98.0
    assert used <= PARAMS.budget_credits_per_key
    for led in ledgers:
        await led.store.close()


@pytest.mark.asyncio
async def test_spend_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "sevi.sqlite"
    led = await _open(path)
    await led.record(KeyId.A, "calibration", stt_s=60)
    r = await led.reserve(KeyId.A, "calibration", stt_s=30)  # in flight at the "crash"
    assert r.credits == 30
    await led.store.close()  # process dies without commit

    led2 = await _open(path)
    assert await led2.used(KeyId.A) == 90.0  # committed + orphan reservation (fail closed)
    with pytest.raises(BudgetExceeded):
        await led2.reserve(KeyId.A, "calibration", stt_s=11)
    await led2.record(KeyId.A, "calibration", stt_s=10)
    assert await led2.remaining(KeyId.A) == 0.0
    shown = await led2.show()
    assert shown["keys"]["A"]["spent"] == 70.0
    assert shown["keys"]["A"]["reserved"] == 30.0
    await led2.store.close()

    # a raised budget after restart applies to the persisted spend
    bigger = LedgerSettings(budget_credits_per_key=150.0, credits_per_stt_s=1.0)
    led3 = await _open(path, bigger)
    await led3.record(KeyId.A, "calibration", stt_s=50)
    with pytest.raises(BudgetExceeded):
        await led3.record(KeyId.A, "calibration", stt_s=1)
    await led3.store.close()
