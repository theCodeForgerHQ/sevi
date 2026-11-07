"""Per-key Gnani permits (contracts §5.2 `KeySemaphore`, B spec §5.2).

One pool per physical key (Key B uses Key A's pool when aliased). Grant rule for purpose p:
- total held < permits;
- "tts" only while no "stream" is held; "stream" only while no "tts" is held;
- "rest" alongside a "stream" (and "stream" alongside a "rest") only if `stream_rest_coexist`;
- strict FIFO among waiters (head-of-line blocking accepted): a new request is granted at once
  only when nobody is waiting.

`timeout_s == 0` grants now or raises `SeviError(KEY_SEMAPHORE_TIMEOUT)` without queueing.
A voice slot IS a "stream" permit (R-14). Process-local: cross-process exclusion is not this
module's job.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field

from sevi.contracts.adapters import Clock, PermitPurpose, Telemetry
from sevi.contracts.enums import KeyId
from sevi.contracts.errors import ErrorCode, SeviError
from sevi.contracts.telemetry import EventName


@dataclass
class _Pool:
    permits: int
    held: Counter[str] = field(default_factory=Counter)
    waiters: deque[tuple[str, asyncio.Future[HeldPermit]]] = field(default_factory=deque)


class HeldPermit:
    """Implements contracts `Permit`."""

    def __init__(self, sem: ProcessKeySemaphore, pool: _Pool, key_id: KeyId, purpose: str) -> None:
        self._sem = sem
        self._pool = pool
        self.key_id = key_id
        self.purpose: PermitPurpose = purpose  # type: ignore[assignment]
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Idempotent; wakes waiters in FIFO order."""
        if self._released:
            return
        self._released = True
        self._pool.held[self.purpose] -= 1
        self._sem._dispatch(self._pool)


class ProcessKeySemaphore:
    def __init__(
        self,
        permits: dict[KeyId, int],
        *,
        alias_b: bool,
        stream_rest_coexist: bool,
        clock: Clock,
        telemetry: Telemetry,
    ) -> None:
        self.alias_b = alias_b
        self.coexist = stream_rest_coexist
        self.clock = clock
        self.telemetry = telemetry
        self._pools: dict[KeyId, _Pool] = {KeyId.A: _Pool(permits[KeyId.A])}
        self._pools[KeyId.B] = self._pools[KeyId.A] if alias_b else _Pool(permits[KeyId.B])

    def pool(self, key_id: KeyId) -> _Pool:
        if key_id is KeyId.NONE:
            raise SeviError(ErrorCode.INTERNAL, "KeyId.NONE holds no Gnani permit")
        return self._pools[key_id]

    def _grantable(self, pool: _Pool, purpose: str) -> bool:
        held = pool.held
        if sum(held.values()) >= pool.permits:
            return False
        if purpose == "tts":
            return held["stream"] == 0
        if purpose == "rest":
            return held["stream"] == 0 or self.coexist
        # stream
        return held["tts"] == 0 and (held["rest"] == 0 or self.coexist)

    def _grant(self, pool: _Pool, key_id: KeyId, purpose: str) -> HeldPermit:
        pool.held[purpose] += 1
        return HeldPermit(self, pool, key_id, purpose)

    def _dispatch(self, pool: _Pool) -> None:
        while pool.waiters:
            purpose, fut = pool.waiters[0]
            if fut.done():  # cancelled / timed out
                pool.waiters.popleft()
                continue
            if not self._grantable(pool, purpose):
                return  # head-of-line blocking
            pool.waiters.popleft()
            key = KeyId.A if pool is self._pools[KeyId.A] else KeyId.B
            fut.set_result(self._grant(pool, key, purpose))

    def try_take_stream(self, key_id: KeyId) -> HeldPermit | None:
        """Admission: never awaits."""
        pool = self.pool(key_id)
        if pool.waiters or not self._grantable(pool, "stream"):
            return None
        return self._grant(pool, key_id, "stream")

    async def take(self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float) -> HeldPermit:
        pool = self.pool(key_id)
        if not pool.waiters and self._grantable(pool, purpose):
            return self._grant(pool, key_id, purpose)
        if timeout_s <= 0:
            raise SeviError(
                ErrorCode.KEY_SEMAPHORE_TIMEOUT,
                f"no {purpose} permit on key {key_id.value}",
                details={"key_id": key_id.value, "purpose": purpose},
            )
        fut: asyncio.Future[HeldPermit] = asyncio.get_running_loop().create_future()
        pool.waiters.append((purpose, fut))
        start = self.clock.mono_ms()
        try:
            async with asyncio.timeout(timeout_s):
                permit = await asyncio.shield(fut)
        except TimeoutError:
            if fut.done() and not fut.cancelled():
                fut.result().release()  # granted at the deadline: give it back
            else:
                fut.cancel()
            self._dispatch(pool)
            self.telemetry.emit(
                EventName.SEMAPHORE_TIMEOUT, "WARNING", key_id=key_id.value, purpose=purpose
            )
            raise SeviError(
                ErrorCode.KEY_SEMAPHORE_TIMEOUT,
                f"timed out waiting for a {purpose} permit on key {key_id.value}",
                details={"key_id": key_id.value, "purpose": purpose},
            ) from None
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                fut.result().release()
            else:
                fut.cancel()
            self._dispatch(pool)
            raise
        self.telemetry.emit(
            EventName.SEMAPHORE_WAIT,
            "DEBUG",
            key_id=key_id.value,
            purpose=purpose,
            waited_ms=max(self.clock.mono_ms() - start, 0.0),
        )
        return permit

    def acquire(
        self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float
    ) -> AbstractAsyncContextManager[HeldPermit]:
        return self._acquire(key_id, purpose, timeout_s)

    @asynccontextmanager
    async def _acquire(
        self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float
    ) -> AsyncIterator[HeldPermit]:
        permit = await self.take(key_id, purpose, timeout_s)
        try:
            yield permit
        finally:
            permit.release()

    def free_stream(self, key_id: KeyId) -> int:
        """Stream permits that could be granted now (admission display)."""
        pool = self.pool(key_id)
        if pool.held["tts"] or (pool.held["rest"] and not self.coexist):
            return 0
        return max(pool.permits - sum(pool.held.values()), 0)
