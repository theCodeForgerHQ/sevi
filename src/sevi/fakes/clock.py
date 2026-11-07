"""Deterministic clock for tests and demo mode (contracts §5.2 `Clock`)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

DEFAULT_START = datetime(2026, 11, 7, 10, 0, 0, tzinfo=UTC)


class FakeClock:
    """Wall and monotonic time advance together, only when told to."""

    def __init__(self, start: datetime = DEFAULT_START, mono_ms: float = 1_000.0) -> None:
        self._now = start
        self._mono = mono_ms

    def now_utc(self) -> datetime:
        return self._now

    def mono_ms(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds * 1000.0

    def set(self, now: datetime) -> None:
        self.advance((now - self._now).total_seconds())

    async def sleep_until(self, mono_ms: float) -> None:
        if mono_ms > self._mono:
            self.advance((mono_ms - self._mono) / 1000.0)
