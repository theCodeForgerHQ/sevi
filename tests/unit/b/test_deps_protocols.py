"""The real Ledger, BreakerRegistry and ProcessKeySemaphore, driven through the adapter
Protocols (adapters/deps.py). The static half is the TYPE_CHECKING block in deps.py, which
mypy --strict checks in `make check`."""

from __future__ import annotations

from pathlib import Path

import pytest

from sevi.adapters.breaker import BreakerRegistry
from sevi.adapters.deps import BillingLedger, BreakerSource, PermitSemaphore
from sevi.adapters.key_semaphore import ProcessKeySemaphore
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import BudgetExceeded
from sevi.fakes.clock import FakeClock
from sevi.fakes.ledger_kit import open_ledger
from sevi.fakes.null_telemetry import NullTelemetry


@pytest.mark.asyncio
async def test_real_ledger_through_protocol(tmp_path: Path) -> None:
    kit = await open_ledger(tmp_path / "l.sqlite", budget=100.0)
    led: BillingLedger = kit.ledger
    r = await led.reserve(KeyId.B, "public_voice", stt_s=60.0)
    assert (await kit.reservations())[0]["credits"] == 60.0
    await led.top_up(r, stt_s=30.0)
    with pytest.raises(BudgetExceeded):
        await led.top_up(r, stt_s=30.0)  # 90 + 30 > 100
    ev = await led.commit(r, stt_s=75.0)
    assert ev.stt_s == 75.0
    assert await kit.reservations() == []
    r2 = await led.reserve(KeyId.B, "public_voice", stt_s=10.0)
    await led.release(r2)
    assert await kit.reservations() == []
    async with led.billed(KeyId.B, "dev", stt_s=5.0) as bill:
        bill.actual(stt_s=4.0)
    assert [e.stt_s for e in await kit.events()] == [75.0, 4.0]
    assert await led.cal_stop() is False
    await led.mark_exhausted(KeyId.B)  # async: persisted 402 flag
    assert await kit.exhausted(KeyId.B)
    with pytest.raises(BudgetExceeded):
        await led.reserve(KeyId.B, "public_voice", stt_s=1.0)
    await kit.close()


@pytest.mark.asyncio
async def test_real_breakers_and_semaphore_through_protocols() -> None:
    clock, tel = FakeClock(), NullTelemetry()
    reg: BreakerSource = BreakerRegistry(clock, tel, alias_b=True)
    b = reg.get("prisma", KeyId.B)
    assert b is reg.get("prisma", KeyId.A)  # aliased
    b.force_open()
    assert b.is_open()
    sem: PermitSemaphore = ProcessKeySemaphore(
        {KeyId.A: 1, KeyId.B: 1},
        alias_b=False,
        stream_rest_coexist=False,
        clock=clock,
        telemetry=tel,
    )
    async with sem.acquire(KeyId.A, "rest", 1.0) as p:
        assert (p.key_id, p.purpose) == (KeyId.A, "rest")
