"""In-memory ledger with the real ledger's reserve -> commit/release semantics and HARD budget
refusal (BUILD-BRIEF §5; `adapters/ledger.py` is the persistent implementation).

Rates default to 1 credit per STT second and 0.01 per TTS char. `reserve` raises
`BudgetExceeded` when spent + outstanding + estimate > `budget`, before the caller does any I/O.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sevi.contracts.enums import KeyId
from sevi.contracts.errors import BudgetExceeded


@dataclass
class MemReservation:
    rid: int
    key_id: KeyId
    purpose: str
    credits: float
    _actual: dict[str, float] | None = None
    _not_billed: bool = False

    def actual(self, *, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0) -> None:
        self._actual = {"stt_s": stt_s, "tts_chars": tts_chars, "stt_rest_calls": stt_rest_calls}

    def not_billed(self) -> None:
        self._not_billed = True


@dataclass
class Committed:
    key_id: KeyId
    purpose: str
    stt_s: float
    tts_chars: int
    stt_rest_calls: int
    credits: float


@dataclass
class MemoryLedger:
    budget: float = 1000.0
    credits_per_stt_s: float = 1.0
    credits_per_tts_char: float = 0.01
    cal_stopped: bool = False
    events: list[Committed] = field(default_factory=list)
    reservations: dict[int, MemReservation] = field(default_factory=dict)
    exhausted: set[KeyId] = field(default_factory=set)
    refused: int = 0
    _ids: itertools.count[int] = field(default_factory=itertools.count)

    def credits(self, *, stt_s: float = 0, tts_chars: float = 0) -> float:
        return stt_s * self.credits_per_stt_s + tts_chars * self.credits_per_tts_char

    def spent(self, key_id: KeyId) -> float:
        return sum(e.credits for e in self.events if e.key_id is key_id)

    def reserved(self, key_id: KeyId) -> float:
        return sum(r.credits for r in self.reservations.values() if r.key_id is key_id)

    async def reserve(
        self, key_id: KeyId, purpose: str, *, stt_s: float = 0, tts_chars: int = 0
    ) -> MemReservation:
        if key_id in self.exhausted:
            self.refused += 1
            raise BudgetExceeded(f"key {key_id} exhausted")
        est = self.credits(stt_s=stt_s, tts_chars=tts_chars)
        if self.spent(key_id) + self.reserved(key_id) + est > self.budget or self.budget <= 0:
            self.refused += 1
            raise BudgetExceeded("credit budget would be exceeded", details={"key_id": key_id})
        r = MemReservation(next(self._ids), key_id, purpose, est)
        self.reservations[r.rid] = r
        return r

    async def top_up(self, r: MemReservation, *, stt_s: float = 0, tts_chars: int = 0) -> None:
        more = self.credits(stt_s=stt_s, tts_chars=tts_chars)
        if self.spent(r.key_id) + self.reserved(r.key_id) + more > self.budget:
            self.refused += 1
            raise BudgetExceeded("credit budget reached during a stream")
        r.credits += more

    async def commit(
        self, r: MemReservation, *, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0
    ) -> Committed:
        self.reservations.pop(r.rid, None)
        ev = Committed(
            r.key_id,
            r.purpose,
            stt_s,
            tts_chars,
            stt_rest_calls,
            self.credits(stt_s=stt_s, tts_chars=tts_chars),
        )
        self.events.append(ev)
        return ev

    async def release(self, r: MemReservation) -> None:
        self.reservations.pop(r.rid, None)

    def mark_exhausted(self, key_id: KeyId) -> None:
        self.exhausted.add(key_id)

    async def cal_stop(self) -> bool:
        return self.cal_stopped

    @asynccontextmanager
    async def billed(
        self,
        key_id: KeyId,
        purpose: str,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
    ) -> AsyncIterator[MemReservation]:
        r = await self.reserve(key_id, purpose, stt_s=stt_s, tts_chars=tts_chars)
        try:
            yield r
        except BaseException:
            if r._actual is not None:
                await self.commit(r, **r._actual)  # type: ignore[arg-type]
            elif r._not_billed:
                await self.release(r)
            raise
        a = r._actual or {"stt_s": stt_s, "tts_chars": tts_chars, "stt_rest_calls": stt_rest_calls}
        await self.commit(r, **a)  # type: ignore[arg-type]


class MemorySemaphore:
    """Records every permit taken; never blocks (tests that need contention use the real one)."""

    def __init__(self) -> None:
        self.taken: list[tuple[KeyId, str]] = []

    @asynccontextmanager
    async def acquire(self, key_id: KeyId, purpose: str, timeout_s: float) -> AsyncIterator[object]:
        self.taken.append((key_id, purpose))
        yield object()


class MemoryBreaker:
    def __init__(self) -> None:
        self.open = False
        self.failures = 0
        self.forced = 0

    def is_open(self) -> bool:
        return self.open

    def record_failure(self) -> None:
        self.failures += 1

    def force_open(self) -> None:
        self.forced += 1
        self.open = True


class MemoryBreakers:
    def __init__(self) -> None:
        self.by_key: dict[tuple[str, KeyId | None], MemoryBreaker] = {}

    def get(self, dep: str, key_id: KeyId | None) -> MemoryBreaker:
        return self.by_key.setdefault((dep, key_id), MemoryBreaker())
