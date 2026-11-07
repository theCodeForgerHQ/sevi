"""SQLite session store (contracts §5.2 `SessionStore`, B spec §4.1).

- One `aiosqlite` connection in autocommit mode; writes use `BEGIN IMMEDIATE … COMMIT` under a
  per-store asyncio lock (one transaction at a time on the connection). Other processes
  (`sevi retention --once`, `sevi ledger …`) open their own connection; WAL + busy_timeout
  cover the overlap.
- Pragmas on connect, in order: WAL, synchronous=NORMAL, foreign_keys=ON, busy_timeout=5000.
- Records are JSON TEXT (`model_dump_json()`), never integer seed columns (R-10).
- Every public method maps `sqlite3.Error` to `SeviError(STORE_FAILED)`; the first failure of
  each method per minute is logged as `error` STORE_FAILED.

Person A's record models (`TrialRecord`, and therefore `SessionRecord`) are not on `main` yet,
so the typed methods are thin wrappers around JSON-level methods (`put_session_json`,
`get_session_json`, `append_trial_json`) which hold all the logic and are what the B tests
exercise until A's contracts land.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import aiosqlite

from sevi.app.clock import SystemClock
from sevi.contracts.adapters import Clock, LedgerEvent, LedgerRow, Telemetry
from sevi.contracts.enums import KeyId, RetentionClass, SessionState
from sevi.contracts.errors import ErrorCode, SeviError
from sevi.contracts.session import AllowlistEntry
from sevi.contracts.telemetry import EventName

if TYPE_CHECKING:
    from sevi.contracts.session import SessionRecord
    from sevi.contracts.trial import TrialRecord

STORE_BUSY_TIMEOUT_MS = 5_000
RETENTION_STUDY_AFTER = timedelta(hours=72)
IP_HITS_KEEP_S = 86_400
DB_FILE_NAME = "sevi.sqlite"
PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    f"PRAGMA busy_timeout={STORE_BUSY_TIMEOUT_MS}",
)
# Identifiers removed from STUDY records after 72 h (B spec §4.1, U-RET-03).
STUDY_STRIP_FIELDS = ("participant_id", "twilio_call_sid", "client", "report_token_sha256")

P = ParamSpec("P")
R = TypeVar("R")


def iso(d: datetime) -> str:
    """Canonical column timestamp: UTC, microseconds, 'Z' (sorts lexically)."""
    return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _iso_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return iso(v)
    return iso(datetime.fromisoformat(str(v)))


def _null_keys(obj: Any, keys: frozenset[str], *, remove: bool) -> Any:
    """Recursively null (or remove) every dict key in `keys`."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in keys:
                if not remove:
                    out[k] = None
                continue
            out[k] = _null_keys(v, keys, remove=remove)
        return out
    if isinstance(obj, list):
        return [_null_keys(x, keys, remove=remove) for x in obj]
    return obj


def _wrap(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    @functools.wraps(fn)
    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except sqlite3.Error as exc:
            store = args[0]
            assert isinstance(store, SqliteStore)
            store._log_failure(fn.__name__, exc)
            raise SeviError(ErrorCode.STORE_FAILED, f"{fn.__name__}: {exc}") from exc

    return inner


class SqliteStore:
    """Implements `sevi.contracts.adapters.SessionStore` (panel methods cut)."""

    def __init__(
        self, path: Path, *, clock: Clock | None = None, telemetry: Telemetry | None = None
    ) -> None:
        self.path = Path(path)
        self.clock: Clock = clock or SystemClock()
        self.telemetry = telemetry
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._failure_logged_at: dict[str, float] = {}

    # ---------- lifecycle ----------
    async def open(self) -> SqliteStore:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path, isolation_level=None)
        self._db.row_factory = aiosqlite.Row
        for pragma in PRAGMAS:
            async with self._db.execute(pragma) as cur:
                await cur.fetchall()
        await self.migrate()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> SqliteStore:
        return await self.open()

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise SeviError(ErrorCode.STORE_FAILED, "store is not open")
        return self._db

    async def migrate(self) -> None:
        migrations = resources.files("sevi.app").joinpath("migrations")
        for entry in sorted(migrations.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".sql"):
                await self.db.executescript(entry.read_text(encoding="utf-8"))

    def _log_failure(self, method: str, exc: Exception) -> None:
        now = time.monotonic()
        last = self._failure_logged_at.get(method)
        if last is not None and now - last < 60:
            return
        self._failure_logged_at[method] = now
        if self.telemetry is not None:
            self.telemetry.emit(
                EventName.ERROR,
                "ERROR",
                error_code=ErrorCode.STORE_FAILED.value,
                method=method,
                exc=type(exc).__name__,
            )

    @asynccontextmanager
    async def tx(self) -> AsyncIterator[aiosqlite.Connection]:
        """`BEGIN IMMEDIATE … COMMIT` (rollback on error), one at a time on this connection.

        Cancellation-safe: aiosqlite runs each statement on its thread even when the awaiting
        task is cancelled, so a task cancelled during `BEGIN` (e.g. a Prisma ledger top-up
        whose stream is being closed) would leave the connection inside a transaction and
        every later `tx()` failing with "cannot start a transaction within a transaction"
        (#72). Any exception, cancellation included, therefore queues a `ROLLBACK` behind the
        `BEGIN` (statements run in order) and waits for it, shielded from a second cancel."""
        async with self._lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                yield self.db
                await self.db.execute("COMMIT")
            except BaseException:
                await asyncio.shield(self._rollback_quietly())
                raise

    async def _rollback_quietly(self) -> None:
        with contextlib.suppress(sqlite3.OperationalError):  # no transaction is active
            await self.db.execute("ROLLBACK")

    async def _one(self, sql: str, params: tuple[Any, ...] = ()) -> aiosqlite.Row | None:
        async with self._lock, self.db.execute(sql, params) as cur:
            row: aiosqlite.Row | None = await cur.fetchone()
            return row

    async def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        async with self._lock, self.db.execute(sql, params) as cur:
            return list(await cur.fetchall())

    def _now(self) -> str:
        return iso(self.clock.now_utc())

    # ---------- sessions (JSON level) ----------
    @_wrap
    async def put_session_json(self, record: dict[str, Any]) -> None:
        """Upsert one session from its SessionRecord JSON (with or without `trials`)."""
        data = {**record, "trials": []}
        async with self.tx() as db:
            await db.execute(
                """INSERT INTO sessions(session_id, record_json, state, retention,
                     participant_type, created_at, started_at, delete_after,
                     report_token_sha256, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     record_json=excluded.record_json, state=excluded.state,
                     retention=excluded.retention, participant_type=excluded.participant_type,
                     created_at=excluded.created_at, started_at=excluded.started_at,
                     delete_after=excluded.delete_after,
                     report_token_sha256=excluded.report_token_sha256,
                     updated_at=excluded.updated_at""",
                (
                    data["session_id"],
                    json.dumps(data, ensure_ascii=False),
                    str(data["state"]),
                    str(data["retention"]),
                    str(data["participant_type"]),
                    _iso_or_none(data["created_at"]),
                    _iso_or_none(data.get("started_at")),
                    _iso_or_none(data.get("delete_after")),
                    data.get("report_token_sha256"),
                    self._now(),
                ),
            )

    @_wrap
    async def get_session_json(self, session_id: str) -> dict[str, Any] | None:
        row = await self._one("SELECT record_json FROM sessions WHERE session_id=?", (session_id,))
        if row is None:
            return None
        trials = await self._all(
            "SELECT record_json FROM trials WHERE session_id=? ORDER BY seq", (session_id,)
        )
        data: dict[str, Any] = json.loads(row["record_json"])
        data["trials"] = [json.loads(t["record_json"]) for t in trials]
        return data

    @_wrap
    async def append_trial_json(
        self, session_id: str, presentation_id: int, record_json: str
    ) -> int:
        """Append one trial; `seq = COALESCE(MAX(seq),0)+1` in the same transaction."""
        async with self.tx() as db:
            async with db.execute(
                "SELECT COALESCE(MAX(seq),0)+1 FROM trials WHERE session_id=?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            seq = int(row[0]) if row else 1
            await db.execute(
                "INSERT INTO trials(session_id, seq, presentation_id, record_json) "
                "VALUES (?,?,?,?)",
                (session_id, seq, presentation_id, record_json),
            )
        return seq

    @_wrap
    async def session_state(self, session_id: str) -> str | None:
        row = await self._one("SELECT state FROM sessions WHERE session_id=?", (session_id,))
        return None if row is None else str(row["state"])

    # ---------- sessions (typed, contracts §5.2) ----------
    async def put_session(self, rec: SessionRecord) -> None:
        await self.put_session_json(
            json.loads(rec.model_copy(update={"trials": ()}).model_dump_json())
        )

    async def get_session(self, session_id: str) -> SessionRecord | None:
        from sevi.contracts.session import SessionRecord

        data = await self.get_session_json(session_id)
        return None if data is None else SessionRecord.model_validate(data)

    async def append_trial(self, trial: TrialRecord) -> None:
        await self.append_trial_json(
            trial.session_id, trial.spec.presentation_id, trial.model_dump_json()
        )

    async def get_by_report_token(self, token_sha256: str) -> SessionRecord | None:
        row = await self._one(
            "SELECT session_id FROM sessions WHERE report_token_sha256=?", (token_sha256,)
        )
        return None if row is None else await self.get_session(str(row["session_id"]))

    # ---------- retention ----------
    @_wrap
    async def list_due_for_retention(self, now: datetime) -> list[str]:
        """DELETE_72H rows past `delete_after` plus unstripped STUDY rows older than 72 h; only
        ENDED sessions (live sessions are never deleted); ordered by created_at."""
        rows = await self._all(
            """SELECT session_id FROM sessions WHERE state=? AND (
                 (retention=? AND delete_after IS NOT NULL AND delete_after <= ?)
                 OR (retention=? AND stripped=0 AND created_at <= ?))
               ORDER BY created_at""",
            (
                SessionState.ENDED.value,
                RetentionClass.DELETE_72H.value,
                iso(now),
                RetentionClass.STUDY_KEEP.value,
                iso(now - RETENTION_STUDY_AFTER),
            ),
        )
        return [str(r["session_id"]) for r in rows]

    @_wrap
    async def list_orphans(self, last_activity_before: datetime) -> list[str]:
        """Non-ENDED sessions whose row was last written at or before the cutoff (U-RET-09:
        crash-orphaned, no task can still own them); ordered by created_at."""
        rows = await self._all(
            "SELECT session_id FROM sessions WHERE state<>? AND updated_at <= ? "
            "ORDER BY created_at",
            (SessionState.ENDED.value, iso(last_activity_before)),
        )
        return [str(r["session_id"]) for r in rows]

    @_wrap
    async def retention_class(self, session_id: str) -> str | None:
        row = await self._one("SELECT retention FROM sessions WHERE session_id=?", (session_id,))
        return None if row is None else str(row["retention"])

    @_wrap
    async def delete_session(self, session_id: str) -> None:
        async with self.tx() as db:
            await db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

    @_wrap
    async def strip_study_identifiers(self, session_id: str) -> None:
        async with self.tx() as db:
            async with db.execute(
                "SELECT record_json FROM sessions WHERE session_id=?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            if row is None:
                return
            data = json.loads(row[0])
            for f in STUDY_STRIP_FIELDS:
                data[f] = None
            await db.execute(
                "UPDATE sessions SET record_json=?, report_token_sha256=NULL, stripped=1, "
                "updated_at=? WHERE session_id=?",
                (json.dumps(data, ensure_ascii=False), self._now(), session_id),
            )
            async with db.execute(
                "SELECT seq, record_json FROM trials WHERE session_id=?", (session_id,)
            ) as cur:
                trials = list(await cur.fetchall())
            for seq, tjson in trials:
                stripped = _null_keys(json.loads(tjson), frozenset({"transcript"}), remove=False)
                await db.execute(
                    "UPDATE trials SET record_json=? WHERE session_id=? AND seq=?",
                    (json.dumps(stripped, ensure_ascii=False), session_id, seq),
                )

    @_wrap
    async def list_study_sessions_json(self) -> list[dict[str, Any]]:
        rows = await self._all(
            "SELECT session_id FROM sessions WHERE participant_type='STUDY' "
            "ORDER BY started_at, session_id"
        )
        out = []
        for r in rows:
            data = await self.get_session_json(str(r["session_id"]))
            if data is not None:
                out.append(data)
        return out

    async def list_study_sessions(self) -> list[SessionRecord]:
        from sevi.contracts.session import SessionRecord

        return [SessionRecord.model_validate(d) for d in await self.list_study_sessions_json()]

    async def study_export_rows(self) -> list[dict[str, Any]]:
        """STUDY rows for `sevi study export`: scored digits only, no `transcript` and no
        `normalised_tokens` anywhere in the trials (C§10.1, U-RET-14)."""
        drop = frozenset({"transcript", "normalised_tokens"})
        rows = []
        for data in await self.list_study_sessions_json():
            data["trials"] = [_null_keys(t, drop, remove=True) for t in data["trials"]]
            rows.append(data)
        return rows

    @_wrap
    async def purge_expired(self, now: datetime) -> tuple[int, int]:
        """Delete expired idempotency rows and ip_hits older than 24 h."""
        async with self.tx() as db:
            c1 = await db.execute("DELETE FROM idempotency WHERE expires_at <= ?", (iso(now),))
            c2 = await db.execute(
                "DELETE FROM ip_hits WHERE t < ?", (now.timestamp() - IP_HITS_KEEP_S,)
            )
            return c1.rowcount, c2.rowcount

    @_wrap
    async def compact(self) -> dict[str, Any]:
        """wal_checkpoint(TRUNCATE) -> VACUUM -> wal_checkpoint(TRUNCATE); event
        `retention.compact` (C§5.2 v4)."""
        wal = Path(f"{self.path}-wal")
        wal_before = wal.stat().st_size if wal.exists() else 0
        async with self._lock:
            for sql in (
                "PRAGMA wal_checkpoint(TRUNCATE)",
                "VACUUM",
                "PRAGMA wal_checkpoint(TRUNCATE)",
            ):
                async with self.db.execute(sql) as cur:
                    await cur.fetchall()
        if self.telemetry is not None:
            self.telemetry.emit(
                EventName.RETENTION_COMPACT, "INFO", wal_bytes_before=wal_before, vacuumed=True
            )
        return {"wal_bytes_before": wal_before, "vacuumed": True}

    # ---------- twilio / idempotency / allowlist ----------
    @_wrap
    async def twilio_status_upsert(self, call_sid: str, status: str, seq: int) -> bool:
        async with self.tx() as db:
            cur = await db.execute(
                "INSERT OR IGNORE INTO twilio_status(call_sid, call_status, seq, received_at) "
                "VALUES (?,?,?,?)",
                (call_sid, status, seq, self._now()),
            )
            return cur.rowcount == 1

    @_wrap
    async def idempotency_get(self, scope: str, key: str) -> str | None:
        row = await self._one(
            "SELECT value FROM idempotency WHERE scope=? AND key=? AND expires_at > ?",
            (scope, key, self._now()),
        )
        return None if row is None else str(row["value"])

    @_wrap
    async def idempotency_put(self, scope: str, key: str, value: str, ttl_s: int) -> None:
        expires = iso(self.clock.now_utc() + timedelta(seconds=ttl_s))
        async with self.tx() as db:
            await db.execute(
                "INSERT OR REPLACE INTO idempotency(scope, key, value, expires_at) "
                "VALUES (?,?,?,?)",
                (scope, key, value, expires),
            )

    @_wrap
    async def allowlist_get(self, participant_id: str) -> AllowlistEntry | None:
        row = await self._one(
            "SELECT entry_json FROM allowlist WHERE participant_id=?", (participant_id,)
        )
        return None if row is None else AllowlistEntry.model_validate_json(row["entry_json"])

    @_wrap
    async def allowlist_put(self, entry: AllowlistEntry) -> None:
        async with self.tx() as db:
            await db.execute(
                "INSERT OR REPLACE INTO allowlist(participant_id, number_hmac, entry_json) "
                "VALUES (?,?,?)",
                (entry.participant_id, entry.number_hmac, entry.model_dump_json()),
            )

    @_wrap
    async def allowlist_list(self) -> list[AllowlistEntry]:
        rows = await self._all("SELECT entry_json FROM allowlist ORDER BY participant_id")
        return [AllowlistEntry.model_validate_json(r["entry_json"]) for r in rows]

    # ---------- rate limiting ----------
    @_wrap
    async def ip_bucket_hit(self, ip_hmac: str, window_s: int) -> int:
        now = self.clock.now_utc().timestamp()
        async with self.tx() as db:
            await db.execute("INSERT INTO ip_hits(ip_hmac, t) VALUES (?,?)", (ip_hmac, now))
            await db.execute("DELETE FROM ip_hits WHERE t < ?", (now - IP_HITS_KEEP_S,))
            async with db.execute(
                "SELECT COUNT(*) FROM ip_hits WHERE ip_hmac=? AND t >= ?",
                (ip_hmac, now - window_s),
            ) as cur:
                row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ---------- trial minutes ----------
    @_wrap
    async def trial_minutes_add(self, call_sid: str, minutes: int) -> float:
        key = hashlib.sha256(call_sid.encode()).hexdigest()
        async with self.tx() as db:
            await db.execute(
                "INSERT OR IGNORE INTO trial_minutes(call_sha256, minutes, added_at) "
                "VALUES (?,?,?)",
                (key, minutes, self._now()),
            )
        return await self.trial_minutes_total()

    @_wrap
    async def trial_minutes_total(self) -> float:
        row = await self._one("SELECT COALESCE(SUM(minutes),0) FROM trial_minutes")
        return float(row[0]) if row else 0.0

    # ---------- ledger (rows; policy lives in adapters/ledger.py) ----------
    @_wrap
    async def ledger_import(self, events: list[LedgerEvent]) -> int:
        """INSERT OR IGNORE per event_id; returns rows inserted (U-LED-10)."""
        n = 0
        async with self.tx() as db:
            for ev in events:
                cur = await db.execute(
                    "INSERT OR IGNORE INTO ledger_events(event_id, key_id, bucket, purpose, "
                    "window_id, origin, at, event_json) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        ev.event_id,
                        ev.key_id.value,
                        ev.bucket,
                        ev.purpose,
                        ev.window_id,
                        ev.origin,
                        iso(ev.at),
                        ev.model_dump_json(),
                    ),
                )
                n += cur.rowcount
        return n

    @_wrap
    async def ledger_events(self, key_id: KeyId | None = None) -> list[LedgerEvent]:
        if key_id is None:
            rows = await self._all("SELECT event_json FROM ledger_events ORDER BY at, event_id")
        else:
            rows = await self._all(
                "SELECT event_json FROM ledger_events WHERE key_id=? ORDER BY at, event_id",
                (key_id.value,),
            )
        return [LedgerEvent.model_validate_json(r["event_json"]) for r in rows]

    async def ledger_add(
        self, key_id: KeyId, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0
    ) -> LedgerRow:
        """Compatibility wrapper (tests only): one server event in the key's dev bucket."""
        import uuid

        bucket = "DEV" if key_id is KeyId.B else "OPS"
        await self.ledger_import(
            [
                LedgerEvent(
                    event_id=uuid.uuid4().hex,
                    key_id=key_id,
                    bucket=bucket,
                    purpose="dev",
                    window_id="UNSCHEDULED",
                    origin="server",
                    stt_s=stt_s,
                    tts_chars=tts_chars,
                    stt_rest_calls=stt_rest_calls,
                    at=self.clock.now_utc(),
                )
            ]
        )
        return await self.ledger_get(key_id)

    async def ledger_get(self, key_id: KeyId, credits_total: float = 0.0) -> LedgerRow:
        events = await self.ledger_events(key_id)
        return LedgerRow(
            key_id=key_id,
            stt_s=sum(e.stt_s for e in events),
            tts_chars=sum(e.tts_chars for e in events),
            stt_rest_calls=sum(e.stt_rest_calls for e in events),
            credits_used_est=sum(e.credits_est for e in events),
            credits_total=credits_total,
            updated_at=max((e.at for e in events), default=self.clock.now_utc()),
        )

    async def ledger_bucket_used(self, key_id: KeyId, bucket: str) -> float:
        """C§14.2: max(Σ credits_est(bucket), dashboard_used_latest * bucket_share + the
        bucket's estimates after that reading); share = the bucket's share of credits_est up to
        the latest manual reconcile of the key."""
        events = await self.ledger_events(key_id)
        est = sum(e.credits_est for e in events if e.bucket == bucket)
        manual = [e for e in events if e.origin == "manual" and e.dashboard_used is not None]
        if not manual:
            return est
        latest = manual[-1]
        before = [e for e in events if e.at <= latest.at]
        total_before = sum(e.credits_est for e in before)
        bucket_before = sum(e.credits_est for e in before if e.bucket == bucket)
        if total_before > 0:
            share = bucket_before / total_before
        else:  # no estimates at all: the reading belongs to the reconciled bucket(s)
            same = {e.bucket for e in manual if e.at == latest.at}
            share = 1.0 / len(same) if bucket in same else 0.0
        after = sum(e.credits_est for e in events if e.at > latest.at and e.bucket == bucket)
        assert latest.dashboard_used is not None
        return max(est, latest.dashboard_used * share + after)
