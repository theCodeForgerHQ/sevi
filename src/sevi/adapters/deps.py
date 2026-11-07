"""Structural dependencies of the Gnani/Evon/Twilio adapters (B spec §5).

The adapters depend on the ledger (`adapters/ledger.py`), the breakers (`adapters/breaker.py`)
and the key semaphore (`adapters/key_semaphore.py`) only through these small Protocols, so they
are testable with `fakes/memory_ledger.py` and work unchanged with the real classes.

Credit-safety protocol (BUILD-BRIEF §5): every billed call is wrapped in
`ledger.billed(key, purpose, ...)` (or `reserve` -> `commit`/`release` for streams). `reserve`
raises `BudgetExceeded` BEFORE any network I/O, so a refused call never reaches the socket.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

from sevi.contracts.adapters import Permit, PermitPurpose
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import ErrorCode, SeviError

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
CAL_PURPOSES: frozenset[str] = frozenset({"calibration", "bench", "vt"})


class Billable(Protocol):
    """A reservation (`ledger.Reservation`)."""

    def actual(self, *, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0) -> None: ...
    def not_billed(self) -> None: ...


class BillingLedger(Protocol):
    def billed(
        self,
        key_id: KeyId,
        purpose: Purpose,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
    ) -> AbstractAsyncContextManager[Billable]: ...

    async def reserve(
        self, key_id: KeyId, purpose: Purpose, *, stt_s: float = 0, tts_chars: int = 0
    ) -> Billable: ...

    async def top_up(self, r: Billable, *, stt_s: float = 0, tts_chars: int = 0) -> None: ...

    async def commit(
        self, r: Billable, *, stt_s: float = 0, tts_chars: int = 0, stt_rest_calls: int = 0
    ) -> object: ...

    async def release(self, r: Billable) -> None: ...

    def mark_exhausted(self, key_id: KeyId) -> None: ...

    async def cal_stop(self) -> bool: ...


class BreakerLike(Protocol):
    def is_open(self) -> bool: ...
    def record_failure(self) -> None: ...
    def force_open(self) -> None: ...


class BreakerSource(Protocol):
    def get(self, dep: Literal["prisma", "evon"], key_id: KeyId | None) -> BreakerLike: ...


class PermitSemaphore(Protocol):
    def acquire(
        self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float
    ) -> AbstractAsyncContextManager[Permit]: ...


def secret(v: object) -> str:
    """SecretStr or str -> str (never logged)."""
    get = getattr(v, "get_secret_value", None)
    return str(get()) if callable(get) else str(v)


@dataclass
class _SimplePermit:
    key_id: KeyId
    purpose: PermitPurpose


class SerialSemaphore:
    """One permit per key, FIFO; for single-process offline tools (calibration render) that do
    not run the server's `ProcessKeySemaphore`."""

    def __init__(self) -> None:
        self._locks: dict[KeyId, asyncio.Lock] = {}

    @asynccontextmanager
    async def acquire(
        self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float
    ) -> AsyncIterator[Permit]:
        lock = self._locks.setdefault(key_id, asyncio.Lock())
        try:
            async with asyncio.timeout(timeout_s):
                await lock.acquire()
        except TimeoutError:
            raise SeviError(ErrorCode.KEY_SEMAPHORE_TIMEOUT, "permit timeout") from None
        try:
            yield _SimplePermit(key_id, purpose)
        finally:
            lock.release()
