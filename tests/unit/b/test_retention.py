"""72 h retention (contracts §10.1, B spec §4.5): U-RET-01..04, U-RET-10."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from b_store_helpers import session_json, trial_json

from sevi.app.retention import run_retention_once
from sevi.app.settings import Settings
from sevi.app.store import SqliteStore
from sevi.app.telemetry import Telemetry, log_file_name
from sevi.contracts.telemetry import EventName
from sevi.fakes.clock import FakeClock

pytestmark = pytest.mark.asyncio


class Env:
    def __init__(self, tmp: Path) -> None:
        self.clock = FakeClock()
        self.settings = Settings(_env_file=None, log_dir=tmp / "logs", data_dir=tmp / "data")  # type: ignore[call-arg]
        self.out = io.StringIO()
        self.tel = Telemetry(self.settings.log_dir, clock=self.clock, stdout=self.out, strict=True)
        self.store = SqliteStore(self.settings.sqlite_path, clock=self.clock, telemetry=self.tel)

    async def run(self, **kw: object) -> object:
        return await run_retention_once(
            settings=self.settings,
            store=self.store,
            telemetry=self.tel,
            now=self.clock.now_utc(),
            **kw,  # type: ignore[arg-type]
        )

    def events(self) -> list[str]:
        self.tel.drain()
        return [
            json.loads(x)["event"]
            for p in sorted(self.settings.log_dir.glob("*.jsonl"))
            for x in p.read_text().splitlines()
        ]


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[Env]:
    e = Env(tmp_path)
    await e.store.open()
    yield e
    await e.store.close()
    e.tel.close()


async def test_U_RET_01_logs_deleted_with_lookahead_after_aggregates(env: Env) -> None:
    log_dir = env.settings.log_dir
    log_dir.mkdir(parents=True)
    now = env.clock.now_utc().replace(minute=0, second=0, microsecond=0)
    env.clock.set(now)
    for h in range(74):
        (log_dir / log_file_name(now - timedelta(hours=h))).write_text("{}\n")
    order: list[str] = []

    def latency_report() -> None:
        order.append("agg")
        assert len(list(log_dir.glob("*.jsonl"))) == 74  # aggregates ran before deletion

    def ops_report() -> None:
        raise RuntimeError("boom")  # a failing aggregate never blocks deletion

    report = await env.run(aggregates=[latency_report, ops_report])
    assert order == ["agg"]
    assert report.aggregates_failed == ["ops_report"]  # type: ignore[attr-defined]
    cutoff = now - timedelta(hours=72) + timedelta(seconds=600)
    for p in log_dir.glob("sevi-*.jsonl"):
        from sevi.app.retention import log_hour

        hour = log_hour(p)
        assert hour is not None
        assert hour > cutoff
    assert report.logs_deleted == 2  # type: ignore[attr-defined]  # the files of hours now-72, now-73
    assert "result.decided" not in env.out.getvalue()


async def test_U_RET_02_lookahead_on_sessions(env: Env) -> None:
    now = env.clock.now_utc()
    soon = session_json(now - timedelta(hours=72), delete_after=now + timedelta(minutes=5))
    later = session_json(now - timedelta(hours=72), delete_after=now + timedelta(minutes=11))
    await env.store.put_session_json(soon)
    await env.store.put_session_json(later)
    report = await env.run()
    assert await env.store.get_session_json(soon["session_id"]) is None
    assert await env.store.get_session_json(later["session_id"]) is not None
    assert report.sessions_deleted == 1  # type: ignore[attr-defined]


async def test_U_RET_03_study_stripped_once(env: Env) -> None:
    now = env.clock.now_utc()
    rec = session_json(
        now - timedelta(hours=72),
        retention="STUDY_KEEP",
        participant_type="STUDY",
        twilio_call_sid="CA" + "a" * 32,
    )
    await env.store.put_session_json(rec)
    await env.store.append_trial_json(rec["session_id"], 1, trial_json(1))
    r1 = await env.run()
    got = await env.store.get_session_json(rec["session_id"])
    assert got is not None
    for f in ("participant_id", "twilio_call_sid", "client", "report_token_sha256"):
        assert got[f] is None, f
    assert got["trials"][0]["transcript"] is None
    assert got["trials"][0]["digits"] == [1, 2, 3]
    assert got["srt"] == rec["srt"]
    assert got["decision"] == rec["decision"]
    assert got["pseudonym"] == "S-001"
    row = await env.store._one("SELECT stripped, report_token_sha256 FROM sessions")
    assert row is not None
    assert row[0] == 1
    assert row[1] is None
    assert r1.sessions_stripped == 1  # type: ignore[attr-defined]
    r2 = await env.run()
    assert r2.sessions_stripped == 0
    assert r2.compacted is False


async def test_U_RET_04_live_session_never_deleted(env: Env) -> None:
    now = env.clock.now_utc()
    live = session_json(
        now - timedelta(hours=80), state="TEST", delete_after=now - timedelta(hours=8)
    )
    await env.store.put_session_json(live)
    await env.run()
    assert await env.store.get_session_json(live["session_id"]) is not None


async def test_U_RET_10_delete_then_compact(env: Env) -> None:
    now = env.clock.now_utc()
    due = session_json(now - timedelta(hours=73))
    await env.store.put_session_json(due)
    for i in range(1, 30):
        await env.store.append_trial_json(due["session_id"], i, trial_json(i) + " " * 2000)
    report = await env.run()
    assert report.compacted  # type: ignore[attr-defined]
    assert await env.store.get_session_json(due["session_id"]) is None
    assert (await env.store._one("PRAGMA freelist_count"))[0] == 0  # type: ignore[index]
    wal = Path(f"{env.store.path}-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    evs = env.events()
    assert EventName.RETENTION_COMPACT.value in evs
    assert EventName.RETENTION_DELETED.value in evs


async def test_U_RET_10_nothing_deleted_no_compact_but_startup_compacts(env: Env) -> None:
    await env.store.put_session_json(session_json(env.clock.now_utc()))
    report = await env.run()
    assert report.compacted is False  # type: ignore[attr-defined]
    assert EventName.RETENTION_COMPACT.value not in env.events()
    report = await env.run(startup=True)
    assert report.compacted is True  # type: ignore[attr-defined]
    assert EventName.RETENTION_COMPACT.value in env.events()


async def test_twilio_delete_failure_keeps_row_until_grace(env: Env) -> None:
    now = env.clock.now_utc()
    rec = session_json(now - timedelta(hours=72), delete_after=now, twilio_call_sid="CA" + "b" * 32)
    await env.store.put_session_json(rec)

    class Failing:
        calls = 0

        async def delete_call(self, call_sid: str) -> None:
            self.calls += 1
            raise RuntimeError("500")

    tw = Failing()
    await env.run(twilio=tw)
    assert await env.store.get_session_json(rec["session_id"]) is not None
    env.clock.advance(3601)
    await env.run(twilio=tw)
    assert await env.store.get_session_json(rec["session_id"]) is None
    assert tw.calls == 2
