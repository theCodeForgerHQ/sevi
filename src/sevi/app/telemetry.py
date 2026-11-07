"""Telemetry: hourly JSONL files + stdout for WARNING and above (contracts §10.1, B spec §4.2).

- `emit` builds a `TelemetryEvent` (redacted, §10.5), renders it and puts the line on a bounded
  queue (10,000). It never blocks and never raises (except unknown event names in strict mode,
  which tests use). A full queue drops the line and counts it; once per 60 s with drops an
  `error` INTERNAL line (`fields.dropped`) is written synchronously to stdout.
- The writer task (`run`) appends to `<log_dir>/sevi-YYYYMMDDHH.jsonl` (UTC hour of `ts`),
  opening a new file on hour change, flushing every 1 s or 500 lines; fsync on close only.
- stdout carries only level >= WARNING, and never `result.decided`, `session.ended` or
  `stt.transcript` (whatever their level).
- No SSE (Scope v3 (c)): `subscribe` raises NotImplementedError.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import IO, Any

import structlog

from sevi.app.clock import SystemClock
from sevi.app.logging import redact
from sevi.contracts.adapters import Clock
from sevi.contracts.errors import ErrorCode
from sevi.contracts.telemetry import EventName, SpanName, TelemetryEvent

TELEMETRY_QUEUE_MAX = 10_000
FLUSH_EVERY_S = 1.0
FLUSH_EVERY_LINES = 500
DROP_REPORT_EVERY_MS = 60_000.0
LOG_FILE_PREFIX = "sevi-"
LOG_FILE_RE = r"^sevi-(\d{10})\.jsonl$"

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
STDOUT_NEVER = frozenset(
    {EventName.RESULT_DECIDED, EventName.SESSION_ENDED, EventName.STT_TRANSCRIPT}
)
_CTX_KEYS = ("session_id", "request_id", "channel", "mode", "presentation_id", "scored_index")


def log_file_name(ts: datetime) -> str:
    return f"{LOG_FILE_PREFIX}{ts.strftime('%Y%m%d%H')}.jsonl"


def _flat(v: Any) -> str | int | float | bool | None:
    if v is None or isinstance(v, str | int | float | bool):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(v)


class Telemetry:
    """Implements `sevi.contracts.adapters.Telemetry`."""

    def __init__(
        self,
        log_dir: Path,
        *,
        clock: Clock | None = None,
        level: str = "INFO",
        service: str = "app",
        stdout: IO[str] | None = None,
        strict: bool = False,
        queue_max: int = TELEMETRY_QUEUE_MAX,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.clock: Clock = clock or SystemClock()
        self.min_level = LEVELS[level]
        self.service = service
        self.stdout = stdout or sys.stdout
        self.strict = strict
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=queue_max)
        self.dropped = 0
        self._dropped_reported_at: float | None = None
        self._file: IO[str] | None = None
        self._file_name: str | None = None

    # ---------- emit ----------
    def emit(self, event: EventName | str, level: str = "INFO", **fields: Any) -> None:
        try:
            name = EventName(event)
        except ValueError:
            if self.strict:
                raise
            self._stdout_error(f"unknown telemetry event {event!r}")
            return
        try:
            self._emit(name, level, fields)
        except Exception as exc:  # emit never raises in prod
            if self.strict:
                raise
            self._stdout_error(f"telemetry emit failed: {type(exc).__name__}")

    def _emit(self, name: EventName, level: str, fields: dict[str, Any]) -> None:
        if LEVELS.get(level, 20) < self.min_level:
            return
        line, ts = self._render(name, level, fields)
        if LEVELS[level] >= LEVELS["WARNING"] and name not in STDOUT_NEVER:
            self._write_stdout(line)
        try:
            self.queue.put_nowait((log_file_name(ts), line))
        except asyncio.QueueFull:
            self.dropped += 1
        self._maybe_report_drops()

    def _render(self, name: EventName, level: str, fields: dict[str, Any]) -> tuple[str, datetime]:
        ctx = structlog.contextvars.get_contextvars()
        top: dict[str, Any] = {k: ctx[k] for k in _CTX_KEYS if ctx.get(k) is not None}
        for k in (*_CTX_KEYS, "span", "dur_ms", "error_code"):
            if k in fields:
                top[k] = fields.pop(k)
        if isinstance(top.get("error_code"), ErrorCode):
            top["error_code"] = top["error_code"].value
        clean = {k: _flat(v) for k, v in redact(fields).items()}
        ts = self.clock.now_utc()
        ev = TelemetryEvent(
            ts=ts,
            t_mono_ms=max(self.clock.mono_ms(), 0.0),
            event=name,
            level=level,
            service=self.service,
            fields=clean,
            **top,
        )
        data = ev.model_dump(mode="json", exclude_none=True)
        data["ts"] = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"
        return json.dumps(data, ensure_ascii=False), ts

    def _write_stdout(self, line: str) -> None:
        with contextlib.suppress(Exception):
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def _stdout_error(self, message: str, **fields: Any) -> None:
        ts = self.clock.now_utc()
        data = {
            "ts": ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z",
            "event": EventName.ERROR.value,
            "level": "ERROR",
            "service": self.service,
            "error_code": ErrorCode.INTERNAL.value,
            "fields": {"message": message, **fields},
        }
        self._write_stdout(json.dumps(data, ensure_ascii=False))

    def _maybe_report_drops(self) -> None:
        if self.dropped <= 0:
            return
        now = self.clock.mono_ms()
        last = self._dropped_reported_at
        if last is None or now - last >= DROP_REPORT_EVERY_MS:
            self._dropped_reported_at = now
            self._stdout_error("telemetry queue full", dropped=self.dropped)

    # ---------- spans ----------
    @contextlib.contextmanager
    def span(self, name: SpanName, **fields: Any) -> Iterator[None]:
        start = self.clock.mono_ms()
        self.emit(EventName.SPAN_START, "DEBUG", span=SpanName(name).value, **fields)
        try:
            yield
        finally:
            self.mark_span(name, start, self.clock.mono_ms(), **fields)

    def mark_span(
        self, name: SpanName, start_mono_ms: float, end_mono_ms: float, **fields: Any
    ) -> None:
        dur = max(end_mono_ms - start_mono_ms, 0.0)
        self.emit(EventName.SPAN_END, "INFO", span=SpanName(name).value, dur_ms=dur, **fields)

    def subscribe(self, *_a: Any, **_k: Any) -> Any:
        raise NotImplementedError("SSE timeline is cut (Scope v3 (c)); use `sevi tail`")

    # ---------- writer ----------
    def _write_line(self, file_name: str, line: str) -> None:
        if file_name != self._file_name or self._file is None:
            if self._file is not None:
                self._file.close()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_dir / file_name, "a", encoding="utf-8")  # noqa: SIM115
            self._file_name = file_name
        self._file.write(line + "\n")

    def drain(self) -> int:
        """Write every queued line now (synchronous). Returns lines written."""
        n = 0
        while True:
            try:
                file_name, line = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._write_line(file_name, line)
            n += 1
        if self._file is not None:
            self._file.flush()
        return n

    async def run(self) -> None:
        """Writer task: flush every 1 s or every 500 lines."""
        pending = 0
        loop = asyncio.get_running_loop()
        last_flush = loop.time()
        while True:
            try:
                async with asyncio.timeout(FLUSH_EVERY_S):
                    file_name, line = await self.queue.get()
                self._write_line(file_name, line)
                pending += 1
            except TimeoutError:
                pass
            now = loop.time()
            if pending and (pending >= FLUSH_EVERY_LINES or now - last_flush >= FLUSH_EVERY_S):
                if self._file is not None:
                    self._file.flush()
                pending, last_flush = 0, now

    def close(self) -> None:
        """Drain, flush and fsync (shutdown only)."""
        self.drain()
        if self._file is not None:
            self._file.flush()
            with contextlib.suppress(OSError):
                os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
            self._file_name = None
