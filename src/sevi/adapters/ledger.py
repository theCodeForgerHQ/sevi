"""Credit + Twilio spend ledger with a HARD per-key budget (contracts §14.2, B spec §4.6, §15).

BUILD-BRIEF credit safety: every Gnani call goes through this module, which **refuses** the call
(raises `BudgetExceeded`) when its reservation would take the key's spend past
`SEVI_LEDGER__BUDGET_CREDITS_PER_KEY`. The protocol is

    async with ledger.billed(KeyId.B, "dev", stt_s=est_seconds) as bill:
        result = await call_gnani(...)            # never reached when refused
        bill.actual(stt_s=measured_seconds)       # what was really billed

1. **reserve** (before the call): in one `BEGIN IMMEDIATE` transaction, spent + outstanding
   reservations + this estimate must be <= budget, else `BudgetExceeded`; the reservation row
   is written in the same transaction. Because the check and the insert share one SQLite write
   lock, concurrent reservations (tasks *or* processes on the same file) cannot overshoot.
2. **commit** (after the call): one `LedgerEvent(origin="server")` with the actual usage is
   written and the reservation deleted, atomically. Actual usage is always recorded, even past
   the budget (the money is spent); the next reservation is then refused.
3. **release**: the call failed before anything was billed; the reservation is dropped.
   An exception inside `billed()` without `actual()` keeps the reservation as spent
   (fail closed: we cannot prove the call was free).

Everything is persisted in SQLite (`ledger_events`, `ledger_reservations`, `trial_minutes`),
so a restart does not reset spend; reservations left by a crash keep counting as spent until
an operator reconciles. Spend of a key = max(sum(credits_est), latest dashboard reading + the
estimates recorded after it) + outstanding reservations (C§14.2 "a higher dashboard figure
always wins").

Credits are estimated from configured rates; with a rate unset (dev/test) a call that needs it
is refused, so no billed call happens without a known price. Key B aliases Key A when no Key B
is registered (one physical budget). The ledger tables use the same DDL as
`app/migrations/0001_init.sql` (CREATE IF NOT EXISTS), so this module works on the app's store
file or on its own file.

Twilio: trial minutes per call (idempotent per CallSid) and the dial stop
(`TWILIO_TRIAL_MINUTES_STOP`).

Reduced scope (BUILD-BRIEF: no calendar): `window_id` defaults to "UNSCHEDULED"; the
observed-rate / exposure admission rule (§9.2.1) is not part of this module yet.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import aiosqlite

from sevi.contracts.adapters import Clock, LedgerEvent, Telemetry
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import BudgetExceeded, ErrorCode, SeviError
from sevi.contracts.telemetry import EventName

Bucket = Literal["CAL", "OPS", "RESERVE", "DEV", "PUBLIC"]
Purpose = Literal[
    "calibration",
    "bench",
    "vt",
    "g5",
    "prompt_render",
    "validation",
    "video",
    "dev",
    "public_voice",
    "phone_record",
    "interview",
    "channel_check",
]

# Purpose -> bucket per key (C§14.2 v4, B spec §4.6).
BUCKETS: dict[KeyId, dict[str, Bucket]] = {
    KeyId.A: {
        "calibration": "CAL",
        "bench": "CAL",
        "vt": "CAL",
        "g5": "CAL",
        "prompt_render": "OPS",
        "validation": "OPS",
        "phone_record": "OPS",
        "channel_check": "OPS",
        "interview": "RESERVE",
    },
    KeyId.B: {
        "g5": "DEV",
        "vt": "DEV",
        "dev": "DEV",
        "video": "DEV",
        "public_voice": "PUBLIC",
        "interview": "RESERVE",
    },
}
UNSCHEDULED = "UNSCHEDULED"
BUSY_TIMEOUT_MS = 5_000

_DDL = """
CREATE TABLE IF NOT EXISTS ledger_events (
  event_id TEXT PRIMARY KEY, key_id TEXT NOT NULL, bucket TEXT NOT NULL, purpose TEXT NOT NULL,
  window_id TEXT NOT NULL, origin TEXT NOT NULL, at TEXT NOT NULL, event_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ledger_k ON ledger_events(key_id, bucket, at);
CREATE TABLE IF NOT EXISTS trial_minutes (
  call_sha256 TEXT PRIMARY KEY, minutes INTEGER NOT NULL, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ledger_reservations (
  reservation_id TEXT PRIMARY KEY, key_id TEXT NOT NULL, purpose TEXT NOT NULL,
  credits REAL NOT NULL, at TEXT NOT NULL);
"""


class LedgerConfig(Protocol):
    """The fields of `SEVI_LEDGER__*` this module reads (app.settings.LedgerSettings fits)."""

    @property
    def credits_per_stt_s(self) -> float | None: ...
    @property
    def credits_per_tts_char(self) -> float | None: ...
    @property
    def key_a_total(self) -> float: ...
    @property
    def key_b_total(self) -> float: ...
    @property
    def reserve_frac(self) -> float: ...
    @property
    def key_a_cal_stop_frac(self) -> float: ...
    @property
    def key_b_public_stop_sessions(self) -> int: ...
    @property
    def voice_session_credits_est(self) -> float | None: ...
    @property
    def budget_credits_per_key(self) -> float: ...


@dataclass(frozen=True)
class LedgerParams:
    """Plain implementation of `LedgerConfig` (tests, CLI)."""

    budget_credits_per_key: float = 0.0
    credits_per_stt_s: float | None = None
    credits_per_tts_char: float | None = None
    key_a_total: float = 5000.0
    key_b_total: float = 5000.0
    reserve_frac: float = 0.10
    key_a_cal_stop_frac: float = 0.55
    key_b_public_stop_sessions: int = 20
    voice_session_credits_est: float | None = None


@dataclass(frozen=True)
class TwilioLimits:
    trial_minutes_total: float = 75.0
    trial_minutes_stop: float = 65.0


@dataclass
class Reservation:
    reservation_id: str
    key_id: KeyId  # physical key
    logical_key: KeyId
    purpose: Purpose
    credits: float
    _actual: dict[str, float] | None = field(default=None, repr=False)
    _not_billed: bool = field(default=False, repr=False)

    def not_billed(self) -> None:
        """The call provably billed nothing (e.g. connect refused): release on error."""
        self._not_billed = True

    def actual(self, *, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0) -> None:
        """Record what the call really billed; committed when `billed()` exits."""
        self._actual = {
            "stt_s": float(stt_s),
            "tts_chars": float(tts_chars),
            "stt_rest_calls": float(stt_rest_calls),
        }


def _iso(d: datetime) -> str:
    return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Ledger:
    def __init__(
        self,
        path: Path,
        cfg: LedgerConfig,
        *,
        alias_b: bool,
        clock: Clock,
        telemetry: Telemetry,
        twilio: TwilioLimits | None = None,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.alias_b = alias_b
        self.clock = clock
        self.telemetry = telemetry
        self.twilio = twilio or TwilioLimits()
        self.exhausted: dict[KeyId, bool] = {KeyId.A: False, KeyId.B: False}
        self._stops_emitted: set[str] = set()
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # ---------- lifecycle ----------
    async def open(self) -> Ledger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path, isolation_level=None)
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}",
        ):
            async with self._db.execute(pragma) as cur:
                await cur.fetchall()
        await self._db.executescript(_DDL)
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise SeviError(ErrorCode.STORE_FAILED, "ledger is not open")
        return self._db

    @asynccontextmanager
    async def _tx(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._lock:
            await self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                await self.db.execute("ROLLBACK")
                raise
            await self.db.execute("COMMIT")

    # ---------- mapping ----------
    def resolve(self, key_id: KeyId) -> KeyId:
        """Physical key: B -> A when Key B is aliased."""
        if key_id is KeyId.NONE:
            raise SeviError(ErrorCode.INTERNAL, "KeyId.NONE is never billed")
        return KeyId.A if key_id is KeyId.B and self.alias_b else key_id

    def window_now(self, key_id: KeyId) -> str:
        return UNSCHEDULED  # BUILD-BRIEF: no calendar

    def bucket_for(self, key_id: KeyId, purpose: Purpose) -> Bucket:
        try:
            return BUCKETS[key_id][purpose]
        except KeyError:
            raise SeviError(
                ErrorCode.INTERNAL, f"purpose {purpose!r} is not billable on key {key_id}"
            ) from None

    def total(self, key_id: KeyId) -> float:
        return self.cfg.key_a_total if self.resolve(key_id) is KeyId.A else self.cfg.key_b_total

    def credits(self, *, stt_s: float = 0, tts_chars: int = 0) -> float:
        """credits_est = stt_s * rate + chars * rate; a needed but unset rate refuses (fail
        closed)."""
        c = 0.0
        if stt_s > 0:
            if self.cfg.credits_per_stt_s is None:
                raise BudgetExceeded("credits_per_stt_s is unset: refusing a billed STT call")
            c += stt_s * self.cfg.credits_per_stt_s
        if tts_chars > 0:
            if self.cfg.credits_per_tts_char is None:
                raise BudgetExceeded("credits_per_tts_char is unset: refusing a billed TTS call")
            c += tts_chars * self.cfg.credits_per_tts_char
        return c

    # ---------- spend ----------
    @staticmethod
    async def _spent(db: aiosqlite.Connection, key: KeyId) -> tuple[float, float]:
        """(committed spend incl. dashboard rule, outstanding reservations) for a physical key."""
        async with db.execute(
            "SELECT event_json FROM ledger_events WHERE key_id=? ORDER BY at, event_id",
            (key.value,),
        ) as cur:
            events = [LedgerEvent.model_validate_json(r[0]) for r in await cur.fetchall()]
        est = sum(e.credits_est for e in events)
        manual = [e for e in events if e.origin == "manual" and e.dashboard_used is not None]
        if manual:
            last = manual[-1]
            after = sum(e.credits_est for e in events if e.at > last.at)
            est = max(est, (last.dashboard_used or 0.0) + after)
        async with db.execute(
            "SELECT COALESCE(SUM(credits),0) FROM ledger_reservations WHERE key_id=?",
            (key.value,),
        ) as cur:
            row = await cur.fetchone()
        return est, float(row[0]) if row else 0.0

    async def used(self, key_id: KeyId) -> float:
        """Committed spend + outstanding reservations of the physical key."""
        async with self._lock:
            spent, reserved = await self._spent(self.db, self.resolve(key_id))
        return spent + reserved

    async def remaining(self, key_id: KeyId) -> float:
        return max(self.cfg.budget_credits_per_key - await self.used(key_id), 0.0)

    async def reserve(
        self, key_id: KeyId, purpose: Purpose, *, stt_s: float = 0, tts_chars: int = 0
    ) -> Reservation:
        """Reserve the estimated credits BEFORE a Gnani call; raises BudgetExceeded."""
        physical = self.resolve(key_id)
        self.bucket_for(key_id, purpose)  # validates the purpose for that key
        if self.exhausted[physical]:
            raise BudgetExceeded(
                f"key {physical} returned 402 (credits exhausted)", details={"key_id": physical}
            )
        est = self.credits(stt_s=stt_s, tts_chars=tts_chars)
        budget = self.cfg.budget_credits_per_key
        rid = uuid.uuid4().hex
        async with self._tx() as db:
            spent, reserved = await self._spent(db, physical)
            if spent + reserved + est > budget or budget <= 0:
                details: dict[str, Any] = {
                    "key_id": physical.value,
                    "budget": budget,
                    "spent": round(spent, 3),
                    "reserved": round(reserved, 3),
                    "requested": round(est, 3),
                }
                refused = True
            else:
                await db.execute(
                    "INSERT INTO ledger_reservations(reservation_id, key_id, purpose, credits, at)"
                    " VALUES (?,?,?,?,?)",
                    (rid, physical.value, purpose, est, _iso(self.clock.now_utc())),
                )
                refused = False
        if refused:
            self._stop_once(f"budget:{physical.value}", physical, "budget", details)
            raise BudgetExceeded(
                f"credit budget for key {physical.value} would be exceeded", details=details
            )
        return Reservation(rid, physical, key_id, purpose, est)

    async def top_up(self, r: Reservation, *, stt_s: float = 0, tts_chars: int = 0) -> None:
        """Grow a reservation (long streams, every flush); raises BudgetExceeded, after which
        the caller must stop the stream and `commit` what was used."""
        more = self.credits(stt_s=stt_s, tts_chars=tts_chars)
        budget = self.cfg.budget_credits_per_key
        async with self._tx() as db:
            spent, reserved = await self._spent(db, r.key_id)
            ok = spent + reserved + more <= budget
            if ok:
                await db.execute(
                    "UPDATE ledger_reservations SET credits = credits + ? WHERE reservation_id=?",
                    (more, r.reservation_id),
                )
        if not ok:
            details = {"key_id": r.key_id.value, "budget": budget, "requested": round(more, 3)}
            self._stop_once(f"budget:{r.key_id.value}", r.key_id, "budget", details)
            raise BudgetExceeded("credit budget reached during a stream", details=details)
        r.credits += more

    async def commit(
        self,
        r: Reservation,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
        window_id: str | None = None,
    ) -> LedgerEvent:
        """Record the actual usage and drop the reservation, atomically."""
        actual = self._actual_credits(stt_s, tts_chars, fallback=r.credits)
        ev = LedgerEvent(
            event_id=uuid.uuid4().hex,
            key_id=r.key_id,
            bucket=self.bucket_for(r.logical_key, r.purpose),
            purpose=r.purpose,
            window_id=window_id or self.window_now(r.key_id),
            origin="server",
            stt_s=stt_s,
            tts_chars=tts_chars,
            stt_rest_calls=stt_rest_calls,
            credits_est=actual,
            at=self.clock.now_utc(),
        )
        async with self._tx() as db:
            await db.execute(
                "DELETE FROM ledger_reservations WHERE reservation_id=?", (r.reservation_id,)
            )
            await self._insert(db, ev)
        self.telemetry.emit(
            EventName.LEDGER_UPDATE,
            "DEBUG",
            key_id=r.key_id.value,
            bucket=ev.bucket,
            purpose=r.purpose,
            credits_est=actual,
        )
        await self._check_budget_stop(r.key_id)
        return ev

    def _actual_credits(self, stt_s: float, tts_chars: int, *, fallback: float) -> float:
        try:
            return self.credits(stt_s=stt_s, tts_chars=tts_chars)
        except BudgetExceeded:  # rate vanished between reserve and commit: keep the estimate
            return fallback

    async def release(self, r: Reservation) -> None:
        """The call failed before anything was billed."""
        async with self._tx() as db:
            await db.execute(
                "DELETE FROM ledger_reservations WHERE reservation_id=?", (r.reservation_id,)
            )

    @asynccontextmanager
    async def billed(
        self,
        key_id: KeyId,
        purpose: Purpose,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
    ) -> AsyncIterator[Reservation]:
        """Reserve, run the call, commit. Refused calls never enter the block.

        - block exits normally: commit `actual()` if set, else the estimate;
        - block raises after `actual()`: commit actual, re-raise;
        - block raises after `not_billed()`: release the reservation, re-raise;
        - block raises otherwise: keep the reservation (counted as spent; fail closed).
        """
        r = await self.reserve(key_id, purpose, stt_s=stt_s, tts_chars=tts_chars)
        try:
            yield r
        except BaseException:
            if r._actual is not None:
                await self._commit_actual(r, stt_rest_calls)
            elif r._not_billed:
                await self.release(r)
            raise
        if r._actual is None:
            r.actual(stt_s=stt_s, tts_chars=tts_chars, stt_rest_calls=stt_rest_calls)
        await self._commit_actual(r, stt_rest_calls)

    async def _commit_actual(self, r: Reservation, default_rest_calls: int) -> None:
        a = r._actual or {}
        await self.commit(
            r,
            stt_s=a.get("stt_s", 0.0),
            tts_chars=int(a.get("tts_chars", 0)),
            stt_rest_calls=int(a.get("stt_rest_calls", default_rest_calls)),
        )

    async def record(
        self,
        key_id: KeyId,
        purpose: Purpose,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
        window_id: str | None = None,
    ) -> LedgerEvent:
        """Spec `record()`: reserve + commit in one go (refused over budget)."""
        r = await self.reserve(key_id, purpose, stt_s=stt_s, tts_chars=tts_chars)
        return await self.commit(
            r, stt_s=stt_s, tts_chars=tts_chars, stt_rest_calls=stt_rest_calls, window_id=window_id
        )

    @staticmethod
    async def _insert(db: aiosqlite.Connection, ev: LedgerEvent) -> int:
        cur = await db.execute(
            "INSERT OR IGNORE INTO ledger_events(event_id, key_id, bucket, purpose, window_id, "
            "origin, at, event_json) VALUES (?,?,?,?,?,?,?,?)",
            (
                ev.event_id,
                ev.key_id.value,
                ev.bucket,
                ev.purpose,
                ev.window_id,
                ev.origin,
                _iso(ev.at),
                ev.model_dump_json(),
            ),
        )
        return cur.rowcount

    async def reconcile(self, ev: LedgerEvent) -> int:
        """Manual dashboard reconcile (origin="manual", dashboard_used set); idempotent on
        event_id. Returns rows inserted (0 or 1)."""
        if ev.origin != "manual" or ev.dashboard_used is None:
            raise SeviError(ErrorCode.INTERNAL, "reconcile needs origin=manual and dashboard_used")
        async with self._tx() as db:
            n = await self._insert(db, ev)
        if n:
            self.telemetry.emit(
                EventName.LEDGER_RECONCILED,
                "INFO",
                key_id=ev.key_id.value,
                window_id=ev.window_id,
                bucket=ev.bucket,
                dashboard_used=ev.dashboard_used,
                stt_s=ev.stt_s,
                tts_chars=ev.tts_chars,
            )
        return n

    async def clear_stale_reservations(self) -> int:
        """Operator action after a crash + reconcile: drop every outstanding reservation."""
        async with self._tx() as db:
            cur = await db.execute("DELETE FROM ledger_reservations")
            return cur.rowcount

    def mark_exhausted(self, key_id: KeyId) -> None:
        """A 402 from any Gnani call: refuse every further call on that key until restart."""
        physical = self.resolve(key_id)
        self.exhausted[physical] = True
        self._stop_once(f"exhausted:{physical.value}", physical, "exhausted", {})

    # ---------- stops ----------
    async def bucket_used(self, key_id: KeyId, bucket: Bucket) -> float:
        physical = self.resolve(key_id)
        async with (
            self._lock,
            self.db.execute(
                "SELECT event_json FROM ledger_events WHERE key_id=? ORDER BY at, event_id",
                (physical.value,),
            ) as cur,
        ):
            events = [LedgerEvent.model_validate_json(r[0]) for r in await cur.fetchall()]
        est = sum(e.credits_est for e in events if e.bucket == bucket)
        manual = [e for e in events if e.origin == "manual" and e.dashboard_used is not None]
        if not manual:
            return est
        last = manual[-1]
        before = [e for e in events if e.at <= last.at]
        tot = sum(e.credits_est for e in before)
        share = (
            sum(e.credits_est for e in before if e.bucket == bucket) / tot
            if tot > 0
            else (1.0 if bucket in {e.bucket for e in manual if e.at == last.at} else 0.0)
        )
        after = sum(e.credits_est for e in events if e.at > last.at and e.bucket == bucket)
        return max(est, (last.dashboard_used or 0.0) * share + after)

    async def cal_stop(self) -> bool:
        return (
            await self.bucket_used(KeyId.A, "CAL")
            >= self.cfg.key_a_cal_stop_frac * self.cfg.key_a_total
        )

    async def ops_stop(self) -> bool:
        cal = await self.bucket_used(KeyId.A, "CAL")
        return await self.bucket_used(KeyId.A, "OPS") >= 0.90 * self.cfg.key_a_total - cal

    async def dev_stop(self) -> bool:
        return await self.bucket_used(KeyId.B, "DEV") >= 0.40 * 0.9 * self.cfg.key_b_total

    async def public_key_low(self, public_key: KeyId) -> bool:
        """Public stop (C§3.4 / §9.2.1, configured rates only): used >= total * (1 - reserve)
        - stop_sessions * EST; also true after a 402 or when the hard budget is reached."""
        physical = self.resolve(public_key)
        if self.exhausted[physical]:
            return True
        used = await self.used(physical)
        est = self.cfg.voice_session_credits_est or 0.0
        line = (
            self.total(physical) * (1 - self.cfg.reserve_frac)
            - self.cfg.key_b_public_stop_sessions * est
        )
        low = used >= line or used + est > self.cfg.budget_credits_per_key
        if low:
            self._stop_once(f"public:{physical.value}", physical, "public", {"used": used})
        return low

    async def _check_budget_stop(self, key: KeyId) -> None:
        if await self.used(key) >= self.cfg.budget_credits_per_key:
            self._stop_once(f"budget:{key.value}", key, "budget", {})

    def _stop_once(self, tag: str, key: KeyId, stop: str, details: dict[str, Any]) -> None:
        if tag in self._stops_emitted:
            return
        self._stops_emitted.add(tag)
        self.telemetry.emit(
            EventName.LEDGER_STOP,
            "WARNING",
            key_id=key.value,
            stop=stop,
            **{k: v for k, v in details.items() if k != "key_id"},
        )

    # ---------- Twilio trial minutes ----------
    async def twilio_minutes_add(self, call_sid: str, seconds: float) -> float:
        """Billed minutes of one finished call (rounded up, idempotent per CallSid)."""
        minutes = max(1, math.ceil(seconds / 60.0)) if seconds > 0 else 0
        key = hashlib.sha256(call_sid.encode()).hexdigest()
        async with self._tx() as db:
            await db.execute(
                "INSERT OR IGNORE INTO trial_minutes(call_sha256, minutes, added_at) "
                "VALUES (?,?,?)",
                (key, minutes, _iso(self.clock.now_utc())),
            )
        return await self.twilio_minutes_total()

    async def twilio_minutes_total(self) -> float:
        async with (
            self._lock,
            self.db.execute("SELECT COALESCE(SUM(minutes),0) FROM trial_minutes") as cur,
        ):
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def check_dial(self, time_limit_s: int = 600) -> None:
        """Refuse a dial whose worst case would pass the trial-minutes stop."""
        total = await self.twilio_minutes_total()
        if total + math.ceil(time_limit_s / 60.0) > self.twilio.trial_minutes_stop:
            raise SeviError(
                ErrorCode.TWILIO_TRIAL_MINUTES_STOP,
                "Twilio trial minutes stop reached",
                details={"used": total, "stop": self.twilio.trial_minutes_stop},
            )

    # ---------- `sevi ledger show` ----------
    async def show(self) -> dict[str, Any]:
        keys = [KeyId.A] if self.alias_b else [KeyId.A, KeyId.B]
        out: dict[str, Any] = {"alias_b": self.alias_b, "keys": {}}
        for k in keys:
            async with self._lock:
                spent, reserved = await self._spent(self.db, k)
            out["keys"][k.value] = {
                "budget": self.cfg.budget_credits_per_key,
                "spent": spent,
                "reserved": reserved,
                "remaining": max(self.cfg.budget_credits_per_key - spent - reserved, 0.0),
                "total": self.total(k),
                "exhausted": self.exhausted[k],
                "buckets": {
                    b: await self.bucket_used(k, b)
                    for b in sorted(set(BUCKETS[KeyId.A].values()) | set(BUCKETS[KeyId.B].values()))
                },
            }
        out["twilio"] = {
            "trial_minutes_used": await self.twilio_minutes_total(),
            "trial_minutes_stop": self.twilio.trial_minutes_stop,
            "trial_minutes_total": self.twilio.trial_minutes_total,
        }
        return out
