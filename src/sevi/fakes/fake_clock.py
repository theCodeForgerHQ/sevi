"""Clocks for tests and demo mode (B spec §12.1, contracts §5.2 `Clock`).

- `FakeClock`: wall and monotonic time advance together and only when told to; `sleep_until`
  advances instantly, so pacing and backoff are asserted without real waiting.
- `ScaledClock`: real time multiplied by `speed` (default 10) for socket-level end-to-end runs;
  `sleep_until` really sleeps `delta / speed`.

PR #26 ships a `FakeClock` in `fakes/clock.py` with the same `advance(seconds)` API; once it
merges, one of the two is kept and the other re-exports it.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

DEFAULT_START_UTC = datetime(2026, 11, 8, 4, 30, tzinfo=UTC)


class FakeClock:
    def __init__(self, start_mono: float = 0.0, start_utc: datetime = DEFAULT_START_UTC) -> None:
        self._mono = float(start_mono)
        self._utc = start_utc
        self.sleeps: list[float] = []  # requested sleep durations (ms), for assertions

    def now_utc(self) -> datetime:
        return self._utc

    def mono_ms(self) -> float:
        return self._mono

    def advance_ms(self, ms: float) -> None:
        if ms < 0:
            raise ValueError("time does not run backwards")
        self._mono += ms
        self._utc += timedelta(milliseconds=ms)

    def advance(self, seconds: float) -> None:
        self.advance_ms(seconds * 1000.0)

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
    def __init__(self, speed: float = 10.0, start_utc: datetime = DEFAULT_START_UTC) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = speed
        self._t0 = time.monotonic()
        self._utc0 = start_utc

    def mono_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0 * self.speed

    def now_utc(self) -> datetime:
        return self._utc0 + timedelta(milliseconds=self.mono_ms())

    async def sleep_until(self, mono_ms: float) -> None:
        delta = mono_ms - self.mono_ms()
        if delta > 0:
            await asyncio.sleep(delta / 1000.0 / self.speed)
