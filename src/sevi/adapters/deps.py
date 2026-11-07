"""Structural dependencies of the Gnani/Evon/Twilio adapters (B spec §5).

The adapters see the ledger (`adapters/ledger.py`), the breakers (`adapters/breaker.py`) and the
key semaphore (`adapters/key_semaphore.py`) through these Protocols. The real classes satisfy
them exactly (checked by mypy --strict and run in `tests/unit/b/test_deps_protocols.py`).

Credit-safety protocol (BUILD-BRIEF §5): every billed call is wrapped in
`ledger.billed(key, purpose, ...)` (or `reserve` -> `top_up` -> `commit`/`release` for streams).
`reserve` raises `BudgetExceeded` BEFORE any network I/O, so a refused call never reaches the
socket. A 402 from Gnani is persisted with `await ledger.mark_exhausted(key)`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from sevi.adapters.ledger import Purpose, Reservation
from sevi.contracts.adapters import LedgerEvent, Permit, PermitPurpose
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import ErrorCode, SeviError

__all__ = [
    "CAL_PURPOSES",
    "BillingLedger",
    "BreakerLike",
    "BreakerSource",
    "PermitSemaphore",
    "Purpose",
    "Reservation",
    "SerialSemaphore",
    "secret",
]

CAL_PURPOSES: frozenset[str] = frozenset({"calibration", "bench", "vt"})


class BillingLedger(Protocol):
    def billed(
        self,
        key_id: KeyId,
        purpose: Purpose,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
    ) -> AbstractAsyncContextManager[Reservation]: ...

    async def reserve(
        self, key_id: KeyId, purpose: Purpose, *, stt_s: float = 0, tts_chars: int = 0
    ) -> Reservation: ...

    async def top_up(self, r: Reservation, *, stt_s: float = 0, tts_chars: int = 0) -> None: ...

    async def commit(
        self,
        r: Reservation,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
        window_id: str | None = None,
    ) -> LedgerEvent: ...

    async def release(self, r: Reservation) -> None: ...

    async def mark_exhausted(self, key_id: KeyId) -> None: ...

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


if TYPE_CHECKING:  # mypy --strict (make check) proves the real classes satisfy the Protocols
    from sevi.adapters.breaker import BreakerRegistry
    from sevi.adapters.key_semaphore import ProcessKeySemaphore
    from sevi.adapters.ledger import Ledger

    def _conforms(led: Ledger, reg: BreakerRegistry, sem: ProcessKeySemaphore) -> None:
        _ledger: BillingLedger = led
        _breakers: BreakerSource = reg
        _sem: PermitSemaphore = sem
