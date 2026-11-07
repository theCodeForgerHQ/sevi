"""SQLite store (B spec §4.1): U-RET-05..08, U-RET-14, ledger rows."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from b_store_helpers import session_json, trial_json

from sevi.app.store import SqliteStore
from sevi.contracts.adapters import LedgerEvent
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import ErrorCode, SeviError
from sevi.fakes.clock import FakeClock

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    s = await SqliteStore(tmp_path / "sevi.sqlite", clock=FakeClock()).open()
    yield s
    await s.close()


async def test_U_RET_05_round_trip_json_columns(store: SqliteStore) -> None:
    rec = session_json(store.clock.now_utc())
    await store.put_session_json(rec)
    for pid in (1, 2, 3):
        assert await store.append_trial_json(rec["session_id"], pid, trial_json(pid)) == pid
    got = await store.get_session_json(rec["session_id"])
    assert got is not None
    assert [t["presentation_id"] for t in got["trials"]] == [1, 2, 3]
    assert got["trials"][0]["noise_seed"] == 2**64 - 1  # JSON TEXT, no integer seed column
    assert {k: v for k, v in got.items() if k != "trials"} == rec
    assert got["report_token_sha256"] == rec["report_token_sha256"]
    # upsert: a second put replaces the row, trials kept
    await store.put_session_json({**rec, "state": "ENDED", "evon": "TEMPLATE"})
    again = await store.get_session_json(rec["session_id"])
    assert again is not None
    assert again["evon"] == "TEMPLATE"
    assert len(again["trials"]) == 3


async def test_U_RET_05_typed_round_trip_when_A_contracts_land(store: SqliteStore) -> None:
    pytest.importorskip("sevi.contracts.trial", reason="Person A's TrialRecord not on main yet")
    from sevi.contracts.session import SessionRecord

    assert SessionRecord.__pydantic_complete__


async def test_U_RET_06_twilio_status_upsert(store: SqliteStore) -> None:
    sid = "CA" + "0" * 32
    assert await store.twilio_status_upsert(sid, "ringing", 1) is True
    assert await store.twilio_status_upsert(sid, "ringing", 1) is False
    assert await store.twilio_status_upsert(sid, "completed", 2) is True


async def test_U_RET_07_ip_bucket(store: SqliteStore) -> None:
    clock = store.clock
    assert isinstance(clock, FakeClock)
    await store.ip_bucket_hit("old", 3600)  # will be > 24 h old
    for _ in range(3):
        await store.ip_bucket_hit("ip", 3600)
    clock.advance(3601)  # those three are outside the 1 h window
    counts = []
    for _ in range(21):
        counts.append(await store.ip_bucket_hit("ip", 3600))
        clock.advance(1)
    assert counts[-1] == 21
    clock.advance(86_400)
    await store.ip_bucket_hit("other", 3600)
    row = await store._one("SELECT COUNT(*) FROM ip_hits")
    assert row is not None
    assert row[0] == 1


async def test_U_RET_08_pragmas(store: SqliteStore) -> None:
    assert (await store._one("PRAGMA journal_mode"))[0] == "wal"  # type: ignore[index]
    assert (await store._one("PRAGMA foreign_keys"))[0] == 1  # type: ignore[index]
    assert (await store._one("PRAGMA busy_timeout"))[0] == 5000  # type: ignore[index]
    assert (await store._one("PRAGMA synchronous"))[0] == 1  # type: ignore[index]  NORMAL


async def test_no_panel_tables(store: SqliteStore) -> None:
    rows = await store._all("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r[0] for r in rows}
    assert "panel_responses" not in names
    assert {"sessions", "trials", "ledger_events", "ip_hits"} <= names


async def test_delete_cascades_trials(store: SqliteStore) -> None:
    rec = session_json(store.clock.now_utc())
    await store.put_session_json(rec)
    await store.append_trial_json(rec["session_id"], 1, trial_json(1))
    await store.delete_session(rec["session_id"])
    assert await store.get_session_json(rec["session_id"]) is None
    assert (await store._one("SELECT COUNT(*) FROM trials"))[0] == 0  # type: ignore[index]


async def test_U_RET_14_study_export_rows_strip_transcripts(store: SqliteStore) -> None:
    now = store.clock.now_utc()
    for _ in range(2):
        rec = session_json(now, retention="STUDY_KEEP", participant_type="STUDY")
        await store.put_session_json(rec)
        await store.append_trial_json(rec["session_id"], 1, trial_json(1))
    await store.put_session_json(session_json(now))  # PUBLIC: not exported
    rows = await store.study_export_rows()
    assert len(rows) == 2
    blob = json.dumps(rows)
    assert "transcript" not in blob
    assert "normalised_tokens" not in blob
    for r in rows:
        assert r["trials"][0]["digits"] == [1, 2, 3]
        assert r["srt"] == {"srt_db": -9.5}
        assert r["pseudonym"] == "S-001"
        assert r["age_band"] == "A18_29"


async def test_idempotency_ttl(store: SqliteStore) -> None:
    clock = store.clock
    assert isinstance(clock, FakeClock)
    await store.idempotency_put("dial", "k", "v", ttl_s=10)
    assert await store.idempotency_get("dial", "k") == "v"
    clock.advance(11)
    assert await store.idempotency_get("dial", "k") is None
    assert await store.purge_expired(clock.now_utc()) == (1, 0)


async def test_trial_minutes(store: SqliteStore) -> None:
    sid = "CA" + "1" * 32
    assert await store.trial_minutes_add(sid, 3) == 3
    assert await store.trial_minutes_add(sid, 3) == 3  # idempotent per call
    assert await store.trial_minutes_add("CA" + "2" * 32, 2) == 5


def _ev(key: KeyId, bucket: str, credits: float, **kw: object) -> LedgerEvent:
    return LedgerEvent.model_validate(
        {
            "event_id": uuid.uuid4().hex,
            "key_id": key,
            "bucket": bucket,
            "purpose": kw.pop("purpose", "dev"),
            "window_id": "UNSCHEDULED",
            "origin": kw.pop("origin", "server"),
            "credits_est": credits,
            "at": kw.pop("at"),
            **kw,
        }
    )


async def test_ledger_import_idempotent_and_bucket_used(store: SqliteStore) -> None:
    t0 = store.clock.now_utc()
    ev = _ev(KeyId.B, "DEV", 600, at=t0)
    assert await store.ledger_import([ev]) == 1
    assert await store.ledger_import([ev]) == 0
    await store.ledger_import([_ev(KeyId.B, "PUBLIC", 400, at=t0, purpose="public_voice")])
    assert await store.ledger_bucket_used(KeyId.B, "DEV") == 600
    # dashboard 1,300 > estimates 1,000: split by share (60/40)
    manual = _ev(
        KeyId.B, "DEV", 0, at=t0 + timedelta(seconds=1), origin="manual", dashboard_used=1300
    )
    await store.ledger_import([manual])
    assert await store.ledger_bucket_used(KeyId.B, "DEV") == pytest.approx(780)
    assert await store.ledger_bucket_used(KeyId.B, "PUBLIC") == pytest.approx(520)
    row = await store.ledger_get(KeyId.B)
    assert row.credits_used_est == 1000


async def test_store_errors_map_to_store_failed(tmp_path: Path) -> None:
    s = SqliteStore(tmp_path / "x.sqlite")
    with pytest.raises(SeviError) as ei:
        await s.get_session_json("x")
    assert ei.value.code is ErrorCode.STORE_FAILED
    s2 = await SqliteStore(tmp_path / "y.sqlite").open()
    await s2.db.execute("DROP TABLE sessions")
    with pytest.raises(SeviError) as ei:
        await s2.get_session_json("x")
    assert ei.value.code is ErrorCode.STORE_FAILED
    await s2.close()
