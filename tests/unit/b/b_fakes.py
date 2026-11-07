"""Test doubles for B's adapter tests (clock + recording telemetry)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from sevi.contracts.telemetry import EventName, SpanName


class Clock:
    def __init__(self, start: datetime = datetime(2026, 11, 7, 10, tzinfo=UTC)) -> None:
        self._now = start
        self._mono = 0.0

    def now_utc(self) -> datetime:
        return self._now

    def mono_ms(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds * 1000.0

    def at(self, seconds: float) -> None:
        """Set monotonic time to `seconds` since start."""
        self.advance(seconds - self._mono / 1000.0)

    async def sleep_until(self, mono_ms: float) -> None:
        if mono_ms > self._mono:
            self.advance((mono_ms - self._mono) / 1000.0)


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        self.events.append((EventName(event).value, level, fields))

    def names(self) -> list[str]:
        return [e[0] for e in self.events]

    @contextlib.contextmanager
    def span(self, name: SpanName, **fields: Any) -> Iterator[None]:
        yield

    def mark_span(self, name: SpanName, start: float, end: float, **fields: Any) -> None:
        pass
