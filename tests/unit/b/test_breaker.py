"""Breaker (contracts §8.8, B spec §5.1): U-BRK-01..05, U-CFG-04 (breaker part)."""

from __future__ import annotations

from b_fakes import Clock, RecordingTelemetry

from sevi.adapters.breaker import Breaker, BreakerRegistry
from sevi.contracts.enums import KeyId


def _b() -> tuple[Breaker, Clock, RecordingTelemetry]:
    clock, tel = Clock(), RecordingTelemetry()
    return Breaker("prisma", KeyId.A, clock, tel), clock, tel


def test_U_BRK_01_three_failures_within_60s_open() -> None:
    b, clock, tel = _b()
    for t in (0.0, 10.0, 59.9):
        clock.at(t)
        assert not b.is_open()
        b.record_failure()
    assert b.is_open()
    assert tel.names().count("breaker.open") == 1
    assert tel.events[0][2] == {"dep": "prisma", "key_id": "A"}


def test_U_BRK_02_expired_failure_not_counted() -> None:
    b, clock, _ = _b()
    for t in (0.0, 30.0, 61.0):
        clock.at(t)
        b.record_failure()
    assert b.state() == "closed"


def test_U_BRK_03_open_for_exactly_60s_then_closed_once() -> None:
    b, clock, tel = _b()
    b.force_open()
    clock.at(59.999)
    assert b.is_open()
    clock.at(60.0)
    assert not b.is_open()
    assert not b.is_open()
    assert tel.names().count("breaker.closed") == 1


def test_U_BRK_04_force_open_calls_subscribers_once() -> None:
    b, _, _ = _b()
    calls: list[int] = []
    unsub = b.subscribe(lambda: calls.append(1))
    b.force_open()
    b.force_open()  # already open: no second notification
    assert b.is_open()
    assert calls == [1]
    unsub()
    unsub()


def test_U_BRK_05_successes_do_not_reset() -> None:
    b, clock, _ = _b()
    b.record_failure()
    clock.at(5)  # a success happens here: there is no API to reset, by design
    b.record_failure()
    clock.at(10)
    b.record_failure()
    assert b.is_open()


def test_failures_cleared_on_open_and_ignored_while_open() -> None:
    b, clock, tel = _b()
    for _ in range(3):
        b.record_failure()
    b.record_failure()  # while open: ignored
    clock.at(60)
    assert not b.is_open()
    b.record_failure()
    assert not b.is_open()
    assert tel.names().count("breaker.open") == 1


def test_U_CFG_04_prisma_breaker_b_is_a_when_aliased() -> None:
    reg = BreakerRegistry(Clock(), RecordingTelemetry(), alias_b=True)
    assert reg.get("prisma", KeyId.B) is reg.get("prisma", KeyId.A)
    assert reg.get("evon", None) is reg.get("evon", None)
    reg2 = BreakerRegistry(Clock(), RecordingTelemetry(), alias_b=False)
    assert reg2.get("prisma", KeyId.B) is not reg2.get("prisma", KeyId.A)
