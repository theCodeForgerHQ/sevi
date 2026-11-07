"""Two-state circuit breakers (contracts §8.8, B spec §5.1).

Closed, or open for 60 s after 3 failures within 60 s, then closed again (no half-open probe,
R-57). Successes never reset the failure list. `force_open` (auth, credits, handshake rejected)
opens at once. Opening clears the failures, emits `breaker.open` (WARNING) and calls every
subscriber synchronously; the first `is_open()` at or after `open_until` closes the breaker and
emits `breaker.closed` once.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Literal

from sevi.contracts.adapters import Clock, Telemetry
from sevi.contracts.enums import KeyId
from sevi.contracts.telemetry import EventName

Dep = Literal["prisma", "evon"]
THRESHOLD = 3
WINDOW_MS = 60_000.0
OPEN_MS = 60_000.0


class Breaker:
    def __init__(
        self,
        dep: Dep,
        key_id: KeyId | None,
        clock: Clock,
        telemetry: Telemetry,
        *,
        threshold: int = THRESHOLD,
        window_ms: float = WINDOW_MS,
        open_ms: float = OPEN_MS,
    ) -> None:
        self.dep = dep
        self.key_id = key_id
        self.clock = clock
        self.telemetry = telemetry
        self.threshold = threshold
        self.window_ms = window_ms
        self.open_ms = open_ms
        self._failures: deque[float] = deque()
        self._open_until: float | None = None
        self._subscribers: list[Callable[[], None]] = []

    def _fields(self) -> dict[str, str | None]:
        return {"dep": self.dep, "key_id": self.key_id.value if self.key_id else None}

    def is_open(self) -> bool:
        if self._open_until is None:
            return False
        if self.clock.mono_ms() >= self._open_until:
            self._open_until = None
            self.telemetry.emit(EventName.BREAKER_CLOSED, "INFO", **self._fields())
            return False
        return True

    def state(self) -> Literal["closed", "open"]:
        return "open" if self.is_open() else "closed"

    def record_failure(self) -> None:
        if self.is_open():
            return
        now = self.clock.mono_ms()
        self._failures.append(now)
        while self._failures and now - self._failures[0] >= self.window_ms:
            self._failures.popleft()
        if len(self._failures) >= self.threshold:
            self._open(now)

    def force_open(self) -> None:
        if self.is_open():
            return
        self._open(self.clock.mono_ms())

    def _open(self, now: float) -> None:
        self._open_until = now + self.open_ms
        self._failures.clear()
        self.telemetry.emit(EventName.BREAKER_OPEN, "WARNING", **self._fields())
        for cb in list(self._subscribers):
            cb()

    def subscribe(self, cb: Callable[[], None]) -> Callable[[], None]:
        self._subscribers.append(cb)

        def unsubscribe() -> None:
            if cb in self._subscribers:
                self._subscribers.remove(cb)

        return unsubscribe


class BreakerRegistry:
    """One breaker per (dep, key). Prisma Key B maps to the Key A object when aliased."""

    def __init__(self, clock: Clock, telemetry: Telemetry, *, alias_b: bool) -> None:
        self.clock = clock
        self.telemetry = telemetry
        self.alias_b = alias_b
        self._breakers: dict[tuple[Dep, KeyId | None], Breaker] = {}

    def get(self, dep: Dep, key_id: KeyId | None) -> Breaker:
        if dep == "prisma" and key_id is KeyId.B and self.alias_b:
            key_id = KeyId.A
        k = (dep, key_id)
        if k not in self._breakers:
            self._breakers[k] = Breaker(dep, key_id, self.clock, self.telemetry)
        return self._breakers[k]
