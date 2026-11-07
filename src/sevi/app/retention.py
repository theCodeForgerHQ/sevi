"""72-hour retention (contracts §10.1, B spec §4.5).

`run_retention_once`, in order:
1. aggregates (session-free reports) — failures are logged and never block deletion;
2. hourly JSONL logs whose hour start <= now - LOG_RETENTION_HOURS + JOB_INTERVAL_S (lookahead);
3. sessions due at now + JOB_INTERVAL_S: STUDY → strip identifiers, else (optional Twilio
   delete, stretch) → delete; live (non-ENDED) sessions are never touched;
4. expired idempotency rows and ip_hits older than 24 h;
5. `store.compact()` (WAL checkpoint + VACUUM) iff a row was deleted or stripped, and always on
   the startup pass (which runs before the listeners accept traffic).

`retention_loop` repeats it every `SEVI_RETENTION__JOB_INTERVAL_S`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from sevi.app.settings import Settings
from sevi.app.store import SqliteStore
from sevi.app.telemetry import LOG_FILE_RE
from sevi.contracts.adapters import Clock, Telemetry
from sevi.contracts.enums import RetentionClass
from sevi.contracts.errors import ErrorCode
from sevi.contracts.telemetry import EventName

AGGREGATE_TIMEOUT_S = 120.0
TWILIO_GRACE = timedelta(hours=1)
_LOG_RE = re.compile(LOG_FILE_RE)


class TwilioDeleter(Protocol):
    """Stretch (phone IVR): delete a call and its recordings. Raises on failure."""

    async def delete_call(self, call_sid: str) -> None: ...


@dataclass
class RetentionReport:
    logs_deleted: int = 0
    sessions_deleted: int = 0
    sessions_stripped: int = 0
    kept_for_twilio: int = 0
    aggregates_failed: list[str] = field(default_factory=list)
    compacted: bool = False


def log_hour(path: Path) -> datetime | None:
    m = _LOG_RE.match(path.name)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=UTC)


def delete_old_logs(log_dir: Path, now: datetime, retention_hours: int, lookahead_s: int) -> int:
    cutoff = now - timedelta(hours=retention_hours) + timedelta(seconds=lookahead_s)
    n = 0
    if not log_dir.is_dir():
        return 0
    for p in sorted(log_dir.iterdir()):
        hour = log_hour(p)
        if hour is not None and hour <= cutoff:
            p.unlink(missing_ok=True)
            n += 1
    return n


async def run_retention_once(
    *,
    settings: Settings,
    store: SqliteStore,
    telemetry: Telemetry,
    now: datetime,
    startup: bool = False,
    aggregates: Sequence[Callable[[], None]] = (),
    twilio: TwilioDeleter | None = None,
) -> RetentionReport:
    report = RetentionReport()
    lookahead = settings.retention.job_interval_s

    # 1. aggregates first (never block deletion)
    for agg in aggregates:
        name = getattr(agg, "__name__", "aggregate")
        try:
            async with asyncio.timeout(AGGREGATE_TIMEOUT_S):
                await asyncio.to_thread(agg)
        except Exception as exc:
            report.aggregates_failed.append(name)
            telemetry.emit(
                EventName.ERROR,
                "ERROR",
                error_code=ErrorCode.INTERNAL.value,
                aggregate=name,
                exc=type(exc).__name__,
            )

    # 2. logs (lookahead: no line is older than the limit when the next pass runs)
    report.logs_deleted = delete_old_logs(
        Path(settings.log_dir), now, settings.log_retention_hours, lookahead
    )

    # 3. sessions
    for sid in await store.list_due_for_retention(now + timedelta(seconds=lookahead)):
        if await store.retention_class(sid) == RetentionClass.STUDY_KEEP.value:
            await store.strip_study_identifiers(sid)
            report.sessions_stripped += 1
            continue
        if twilio is not None and settings.retention.twilio_delete:
            data = await store.get_session_json(sid)
            call_sid = data.get("twilio_call_sid") if data else None
            if data and call_sid and not data.get("twilio_deleted"):
                try:
                    await twilio.delete_call(call_sid)
                except Exception:
                    delete_after = data.get("delete_after")
                    due = datetime.fromisoformat(delete_after) if delete_after else now
                    if now <= due + TWILIO_GRACE:
                        report.kept_for_twilio += 1
                        continue
                    telemetry.emit(
                        EventName.ERROR,
                        "ERROR",
                        error_code=ErrorCode.TWILIO_REST_FAILED.value,
                    )
        await store.delete_session(sid)
        report.sessions_deleted += 1
    if report.sessions_deleted:
        telemetry.emit(EventName.RETENTION_DELETED, "INFO", count=report.sessions_deleted)
    if report.sessions_stripped:
        telemetry.emit(EventName.RETENTION_STRIPPED, "INFO", count=report.sessions_stripped)

    # 4. expired idempotency rows, ip_hits > 24 h
    await store.purge_expired(now)

    # 5. compact
    if startup or report.sessions_deleted or report.sessions_stripped:
        await store.compact()
        report.compacted = True
    return report


async def retention_loop(
    *,
    settings: Settings,
    store: SqliteStore,
    telemetry: Telemetry,
    clock: Clock,
    aggregates: Sequence[Callable[[], None]] = (),
    twilio: TwilioDeleter | None = None,
) -> None:
    """Started in the lifespan after the startup pass (which ran with startup=True)."""
    while True:
        await asyncio.sleep(settings.retention.job_interval_s)
        try:
            await run_retention_once(
                settings=settings,
                store=store,
                telemetry=telemetry,
                now=clock.now_utc(),
                aggregates=aggregates,
                twilio=twilio,
            )
        except Exception as exc:  # keep the loop alive; the next pass retries
            telemetry.emit(
                EventName.ERROR,
                "ERROR",
                error_code=ErrorCode.INTERNAL.value,
                exc=type(exc).__name__,
            )
