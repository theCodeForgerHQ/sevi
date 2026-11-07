"""Orphan recovery (U-RET-09, B spec §4.1/§2.2 step 8, contracts §6.2 S18/S55/S64, §10.1)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from b_fakes import RecordingTelemetry
from b_store_helpers import session_json, trial_json

from sevi.adapters.ledger import Ledger
from sevi.app.retention import run_retention_once, session_max_lifetime_s
from sevi.app.settings import LedgerSettings, Settings
from sevi.app.store import SqliteStore
from sevi.contracts.enums import KeyId
from sevi.fakes.clock import FakeClock

pytestmark = pytest.mark.asyncio

LEDGER = LedgerSettings.model_validate(
    {"budget_credits_per_key": 1000.0, "credits_per_stt_s": 0.5, "credits_per_tts_char": 0.01}
)


def _settings(tmp: Path) -> Settings:
    return Settings(_env_file=None, log_dir=tmp / "logs", data_dir=tmp / "data")  # type: ignore[call-arg]


async def _pass(
    settings: Settings, store: SqliteStore, tel: RecordingTelemetry, clock: FakeClock, **kw: bool
) -> object:
    return await run_retention_once(
        settings=settings, store=store, telemetry=tel, now=clock.now_utc(), **kw
    )


async def test_max_lifetime_covers_consent_and_session_timers(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    assert session_max_lifetime_s(s) > s.session.t_consent_browser_s + s.session.max_s


async def test_U_RET_09_orphan_to_t8_and_live_session_untouched(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    clock = FakeClock()
    tel = RecordingTelemetry()
    async with SqliteStore(s.sqlite_path, clock=clock) as store:
        created = clock.now_utc()
        orphan = session_json(created, state="TEST", decision=None, end_reason=None)
        await store.put_session_json(orphan)
        await store.append_trial_json(orphan["session_id"], 1, trial_json(1))
        clock.advance(session_max_lifetime_s(s) + 1)
        live = session_json(clock.now_utc(), state="TEST", decision=None, end_reason=None)
        await store.put_session_json(live)

        report = await _pass(s, store, tel, clock, startup=True)

        assert report.orphans_recovered == 1  # type: ignore[attr-defined]
        rec = await store.get_session_json(orphan["session_id"])
        assert rec is not None
        assert rec["state"] == "ENDED"
        assert rec["end_reason"] == "SERVER_ERROR"
        assert rec["decision"]["rule"] == "T8"
        assert rec["decision"]["band"] == "FAILED"
        assert set(rec["decision"]["surfaces"]) == {"SPOKEN", "PAGE", "REPORT", "OPERATOR"}
        assert rec["delete_after"] == orphan["delete_after"]  # kept
        assert len(rec["trials"]) == 1  # trials survive the recovery
        assert await store.session_state(live["session_id"]) == "TEST"  # still within lifetime
        assert ("session.crash", "ERROR") in [(e[0], e[1]) for e in tel.events]


async def test_U_RET_09_crash_in_result_keeps_decision(tmp_path: Path) -> None:
    """S64: a crash in RESULT/CLOSING keeps the decision already made (no re-classification)."""
    s = _settings(tmp_path)
    clock = FakeClock()
    tel = RecordingTelemetry()
    async with SqliteStore(s.sqlite_path, clock=clock) as store:
        rec = session_json(
            clock.now_utc(), state="RESULT", end_reason="COMPLETED", decision={"rule": "T11"}
        )
        await store.put_session_json(rec)
        clock.advance(session_max_lifetime_s(s) + 1)
        await _pass(s, store, tel, clock)
        got = await store.get_session_json(rec["session_id"])
        assert got is not None
        assert got["state"] == "ENDED"
        assert got["end_reason"] == "COMPLETED"
        assert got["decision"] == {"rule": "T11"}


async def test_U_RET_09_crash_simulation_restart_t8_then_deleted_at_72h(tmp_path: Path) -> None:
    """A process dies mid-trial with an open Prisma reservation; the restart's startup pass
    closes the session as T8 and settles the reservation as spent; the 72 h pass deletes it."""
    s = _settings(tmp_path)
    clock = FakeClock()
    created = clock.now_utc()

    # --- process 1: session in TEST, two trials, an open ledger reservation, then a crash ---
    store = await SqliteStore(s.sqlite_path, clock=clock).open()
    led = Ledger(store, LEDGER, alias_b=False, clock=clock, telemetry=RecordingTelemetry())
    rec = session_json(created, state="TEST", decision=None, end_reason=None)
    sid = rec["session_id"]
    await store.put_session_json(rec)
    await store.append_trial_json(sid, 1, trial_json(1))
    await store.append_trial_json(sid, 2, trial_json(2))
    await led.reserve(KeyId.B, "public_voice", stt_s=100)  # 50 credits, never committed
    assert await led.used(KeyId.B) == 50.0
    await store.close()  # crash: no ENDED write, no commit/release

    # --- process 2: restart 30 min later; the startup pass runs before serving ---
    clock.advance(30 * 60)
    tel = RecordingTelemetry()
    store2 = await SqliteStore(s.sqlite_path, clock=clock).open()
    led2 = Ledger(store2, LEDGER, alias_b=False, clock=clock, telemetry=RecordingTelemetry())
    try:
        report = await _pass(s, store2, tel, clock, startup=True)
        assert report.orphans_recovered == 1  # type: ignore[attr-defined]
        assert report.reservations_settled == 1  # type: ignore[attr-defined]
        assert report.sessions_deleted == 0  # type: ignore[attr-defined]
        got = await store2.get_session_json(sid)
        assert got is not None
        assert (got["state"], got["end_reason"], got["decision"]["rule"]) == (
            "ENDED",
            "SERVER_ERROR",
            "T8",
        )
        # reservation released but still counted as spent (fail closed)
        assert await led2.used(KeyId.B) == 50.0
        rows = await store2._all("SELECT COUNT(*) AS n FROM ledger_reservations")
        assert rows[0]["n"] == 0
        assert ("ledger.update", "WARNING") in [(e[0], e[1]) for e in tel.events]

        # second pass is a no-op for this session and the ledger
        again = await _pass(s, store2, tel, clock)
        assert again.orphans_recovered == 0  # type: ignore[attr-defined]
        assert again.reservations_settled == 0  # type: ignore[attr-defined]
        assert await led2.used(KeyId.B) == 50.0

        # just before 72 h (minus the job lookahead): still there
        clock.set(created + timedelta(hours=72) - timedelta(seconds=s.retention.job_interval_s + 1))
        await _pass(s, store2, tel, clock)
        assert await store2.session_state(sid) == "ENDED"

        # at 72 h: deleted, trials cascade
        clock.set(created + timedelta(hours=72))
        final = await _pass(s, store2, tel, clock)
        assert final.sessions_deleted == 1  # type: ignore[attr-defined]
        assert await store2.get_session_json(sid) is None
        trials = await store2._all("SELECT COUNT(*) AS n FROM trials")
        assert trials[0]["n"] == 0
    finally:
        await store2.close()


async def test_recent_reservation_is_not_settled(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    clock = FakeClock()
    async with SqliteStore(s.sqlite_path, clock=clock) as store:
        led = Ledger(store, LEDGER, alias_b=False, clock=clock, telemetry=RecordingTelemetry())
        await led.reserve(KeyId.A, "prompt_render", tts_chars=1000)
        clock.advance(60)
        report = await _pass(s, store, RecordingTelemetry(), clock)
        assert report.reservations_settled == 0  # type: ignore[attr-defined]
        rows = await store._all("SELECT COUNT(*) AS n FROM ledger_reservations")
        assert rows[0]["n"] == 1
