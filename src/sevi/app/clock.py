"""Wall and monotonic clock (contracts §5.2 `Clock`)."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def mono_ms(self) -> float:
        return time.monotonic_ns() / 1e6

    async def sleep_until(self, mono_ms: float) -> None:
        delay = (mono_ms - self.mono_ms()) / 1000.0
        if delay > 0:
            await asyncio.sleep(delay)
