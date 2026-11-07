"""Ledger (contracts §14.2, B spec §4.6): hard budget refusal, U-LED-01/04/05/06/10/13."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from b_fakes import Clock, RecordingTelemetry

from sevi.adapters.ledger import BUCKETS, Ledger, LedgerParams
from sevi.contracts.adapters import LedgerEvent
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import BudgetExceeded, ErrorCode, SeviError

pytestmark = pytest.mark.asyncio

RATES = {"credits_per_stt_s": 0.5, "credits_per_tts_char": 0.01}


async def _ledger(path: Path, *, alias_b: bool = False, **cfg: float) -> Ledger:
    params = LedgerParams(**{"budget_credits_per_key": 1000.0, **RATES, **cfg})  # type: ignore[arg-type]
    return await Ledger(
        path, params, alias_b=alias_b, clock=Clock(), telemetry=RecordingTelemetry()
    ).open()


@pytest_asyncio.fixture
async def led(tmp_path: Path) -> AsyncIterator[Ledger]:
    led = await _ledger(tmp_path / "ledger.sqlite")
    yield led
    await led.close()


async def test_U_LED_01_record_buckets_and_credits(led: Ledger) -> None:
    e1 = await led.record(KeyId.B, "dev", stt_s=100)
    e2 = await led.record(KeyId.A, "prompt_render", tts_chars=1000)
    assert (e1.key_id, e1.bucket, e1.credits_est, e1.origin) == (KeyId.B, "DEV", 50.0, "server")
    assert (e2.key_id, e2.bucket, e2.credits_est) == (KeyId.A, "OPS", 10.0)
    assert await led.used(KeyId.B) == 50.0
    assert await led.bucket_used(KeyId.A, "OPS") == 10.0


async def test_call_refused_over_budget_never_runs(led: Ledger) -> None:
    calls: list[int] = []
    async with led.billed(KeyId.B, "dev", stt_s=1800) as bill:  # 900 credits
        calls.append(1)
        bill.actual(stt_s=1800)
    with pytest.raises(BudgetExceeded) as ei:
        async with led.billed(KeyId.B, "dev", stt_s=201):  # 100.5 > 100 left
            calls.append(2)
    assert calls == [1]
    assert ei.value.code is ErrorCode.CREDIT_STOP
    assert ei.value.retryable is False
    assert ei.value.details["spent"] == 900.0
    # exactly reaching the budget is allowed; one more credit is not
    await led.record(KeyId.B, "dev", stt_s=200)
    with pytest.raises(BudgetExceeded):
        await led.record(KeyId.B, "dev", stt_s=0.01)
    assert await led.remaining(KeyId.B) == 0.0
    assert any(n == "ledger.stop" for n in led.telemetry.names())  # type: ignore[attr-defined]
    # the other key has its own budget
    await led.record(KeyId.A, "calibration", stt_s=10)


async def test_zero_budget_refuses_everything(tmp_path: Path) -> None:
    led = await _ledger(tmp_path / "z.sqlite", budget_credits_per_key=0.0)
    with pytest.raises(BudgetExceeded):
        await led.reserve(KeyId.A, "calibration", stt_s=1)
    await led.close()


async def test_unset_rate_refuses_billed_call(tmp_path: Path) -> None:
    led = await _ledger(tmp_path / "r.sqlite", credits_per_stt_s=None)  # type: ignore[arg-type]
    with pytest.raises(BudgetExceeded, match="credits_per_stt_s"):
        await led.reserve(KeyId.A, "calibration", stt_s=1)
    await led.close()


async def test_actual_above_estimate_is_recorded_and_blocks_next(led: Ledger) -> None:
    async with led.billed(KeyId.B, "dev", stt_s=10) as bill:
        bill.actual(stt_s=2100)  # the stream ran long: 1,050 credits really billed
    assert await led.used(KeyId.B) == 1050.0
    with pytest.raises(BudgetExceeded):
        await led.reserve(KeyId.B, "dev", stt_s=0.1)


async def test_error_paths_release_or_keep_reservation(led: Ledger) -> None:
    async def refused_connect() -> None:
        async with led.billed(KeyId.B, "dev", stt_s=100) as bill:
            bill.not_billed()
            raise RuntimeError("connect refused")

    async def unknown_failure() -> None:
        async with led.billed(KeyId.B, "dev", stt_s=100):
            raise RuntimeError("unknown failure mid-call")

    with pytest.raises(RuntimeError):
        await refused_connect()
    assert await led.used(KeyId.B) == 0.0
    with pytest.raises(RuntimeError):
        await unknown_failure()
    assert await led.used(KeyId.B) == 50.0  # fail closed: kept as reserved
    shown = await led.show()
    assert shown["keys"]["B"]["reserved"] == 50.0
    assert await led.clear_stale_reservations() == 1
    assert await led.used(KeyId.B) == 0.0


async def test_top_up_for_streams(led: Ledger) -> None:
    r = await led.reserve(KeyId.B, "dev", stt_s=60)
    await led.top_up(r, stt_s=1800)
    with pytest.raises(BudgetExceeded):
        await led.top_up(r, stt_s=200)
    await led.commit(r, stt_s=1860)
    assert await led.used(KeyId.B) == 930.0


async def test_aliased_b_spends_a_budget(tmp_path: Path) -> None:
    led = await _ledger(tmp_path / "a.sqlite", alias_b=True)
    await led.record(KeyId.A, "calibration", stt_s=1000)  # 500
    ev = await led.record(KeyId.B, "public_voice", stt_s=900)  # 450 on A
    assert ev.key_id is KeyId.A
    assert ev.bucket == "PUBLIC"
    with pytest.raises(BudgetExceeded):
        await led.record(KeyId.B, "dev", stt_s=101)
    assert list((await led.show())["keys"]) == ["A"]
    await led.close()


async def test_U_LED_04_cal_stop_server(tmp_path: Path) -> None:
    led = await _ledger(tmp_path / "c.sqlite", budget_credits_per_key=10_000.0)
    await led.record(KeyId.A, "calibration", stt_s=5498)  # 2,749
    assert await led.cal_stop() is False
    await led.record(KeyId.A, "bench", stt_s=2)  # 2,750
    assert await led.cal_stop() is True
    await led.close()


async def test_U_LED_05_402_exhausts_key(led: Ledger) -> None:
    assert await led.public_key_low(KeyId.B) is False
    led.mark_exhausted(KeyId.B)
    assert await led.public_key_low(KeyId.B) is True
    with pytest.raises(BudgetExceeded):
        await led.reserve(KeyId.B, "dev", stt_s=1)
    await led.reserve(KeyId.A, "calibration", stt_s=1)


def _manual(key: KeyId, bucket: str, used: float, at: object, **kw: float) -> LedgerEvent:
    return LedgerEvent.model_validate(
        {
            "event_id": uuid.uuid4().hex,
            "key_id": key,
            "bucket": bucket,
            "purpose": "dev",
            "window_id": "UNSCHEDULED",
            "origin": "manual",
            "dashboard_used": used,
            "at": at,
            **kw,
        }
    )


async def test_U_LED_06_reconcile_max_rule(tmp_path: Path) -> None:
    led = await _ledger(tmp_path / "d.sqlite", budget_credits_per_key=5000.0)
    clock = led.clock
    assert isinstance(clock, Clock)
    await led.record(KeyId.B, "dev", stt_s=1200)  # 600
    await led.record(KeyId.B, "public_voice", stt_s=800)  # 400
    clock.advance(1)
    await led.reconcile(_manual(KeyId.B, "DEV", 1300, clock.now_utc()))
    assert await led.bucket_used(KeyId.B, "DEV") == pytest.approx(780)
    assert await led.bucket_used(KeyId.B, "PUBLIC") == pytest.approx(520)
    assert await led.used(KeyId.B) == pytest.approx(1300)  # the budget sees the dashboard
    clock.advance(1)
    await led.reconcile(_manual(KeyId.B, "DEV", 900, clock.now_utc()))
    assert await led.bucket_used(KeyId.B, "DEV") == pytest.approx(600)  # estimate wins
    await led.close()


async def test_U_LED_10_reconcile_idempotent_and_validated(led: Ledger) -> None:
    ev = _manual(KeyId.A, "CAL", 2800, led.clock.now_utc())
    assert await led.reconcile(ev) == 1
    assert await led.reconcile(ev) == 0
    assert await led.cal_stop() is True  # the manual reading moves the 55 % stop
    server = ev.model_copy(update={"origin": "server", "event_id": uuid.uuid4().hex})
    with pytest.raises(SeviError):
        await led.reconcile(server)
    with pytest.raises(SeviError):
        await led.reconcile(ev.model_copy(update={"dashboard_used": None}))
    assert led.telemetry.names().count("ledger.reconciled") == 1  # type: ignore[attr-defined]


async def test_U_LED_13_purpose_to_bucket(led: Ledger) -> None:
    assert led.bucket_for(KeyId.A, "channel_check") == "OPS"
    assert led.bucket_for(KeyId.B, "video") == "DEV"
    assert led.bucket_for(KeyId.A, "validation") == "OPS"
    assert led.bucket_for(KeyId.B, "public_voice") == "PUBLIC"
    assert led.bucket_for(KeyId.A, "interview") == "RESERVE"
    assert set(BUCKETS[KeyId.A]) & {"public_voice", "dev", "video"} == set()
    with pytest.raises(SeviError):
        led.bucket_for(KeyId.A, "public_voice")
    with pytest.raises(SeviError):
        led.resolve(KeyId.NONE)


async def test_twilio_minutes_and_dial_stop(led: Ledger) -> None:
    assert await led.twilio_minutes_add("CA" + "1" * 32, 61) == 2
    assert await led.twilio_minutes_add("CA" + "1" * 32, 61) == 2  # idempotent per CallSid
    await led.check_dial(time_limit_s=600)
    await led.twilio_minutes_add("CA" + "2" * 32, 54 * 60)  # 56 used
    with pytest.raises(SeviError) as ei:
        await led.check_dial(time_limit_s=600)  # 56 + 10 > 65
    assert ei.value.code is ErrorCode.TWILIO_TRIAL_MINUTES_STOP
    assert (await led.show())["twilio"]["trial_minutes_used"] == 56


async def test_show_payload(led: Ledger) -> None:
    await led.record(KeyId.B, "dev", stt_s=100)
    shown = await led.show()
    b = shown["keys"]["B"]
    assert b["spent"] == 50.0
    assert b["remaining"] == 950.0
    assert b["budget"] == 1000.0
    assert b["buckets"]["DEV"] == 50.0
