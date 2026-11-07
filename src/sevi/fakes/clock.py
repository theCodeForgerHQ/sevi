"""Clocks for tests and demo mode (contracts §5.2 `Clock`; B spec §12.1).

- `FakeClock`: wall and monotonic time advance together and only when told to; `sleep_until`
  advances instantly (and records the requested sleep), so pacing and backoff are asserted
  without real waiting.
- `ScaledClock`: real time multiplied by `speed` (default 10) for socket-level end-to-end runs;
  `sleep_until` really sleeps `delta / speed`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

DEFAULT_START = datetime(2026, 11, 7, 10, 0, 0, tzinfo=UTC)


class FakeClock:
    """Wall and monotonic time advance together, only when told to."""

    def __init__(self, start: datetime = DEFAULT_START, mono_ms: float = 1_000.0) -> None:
        self._now = start
        self._mono = mono_ms
        self.sleeps: list[float] = []  # requested sleep durations (ms), for assertions

    def now_utc(self) -> datetime:
        return self._now

    def mono_ms(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self.advance_ms(seconds * 1000.0)

    def advance_ms(self, ms: float) -> None:
        if ms < 0:
            raise ValueError("time does not run backwards")
        self._now += timedelta(milliseconds=ms)
        self._mono += ms

    def set(self, now: datetime) -> None:
        self.advance((now - self._now).total_seconds())

    async def sleep_until(self, mono_ms: float) -> None:
        delta = mono_ms - self._mono
        self.sleeps.append(max(0.0, delta))
        if delta > 0:
            self.advance_ms(delta)
        await asyncio.sleep(0)  # still a scheduling point, like a real sleep

    async def run_until_idle(self, rounds: int = 50) -> None:
        """Let every ready task run (no time passes)."""
        for _ in range(rounds):
            await asyncio.sleep(0)


class ScaledClock:
    def __init__(self, speed: float = 10.0, start: datetime = DEFAULT_START) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = speed
        self._t0 = time.monotonic()
        self._start = start

    def mono_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0 * self.speed

    def now_utc(self) -> datetime:
        return self._start + timedelta(milliseconds=self.mono_ms())

    async def sleep_until(self, mono_ms: float) -> None:
        delta = mono_ms - self.mono_ms()
        if delta > 0:
            await asyncio.sleep(delta / 1000.0 / self.speed)
