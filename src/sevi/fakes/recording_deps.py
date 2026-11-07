"""Recording breaker and semaphore stand-ins for adapter tests (counts are asserted directly).
The ledger is never faked: tests use the real `adapters.ledger.Ledger` on a tmp SQLite file
(`fakes/ledger_kit.py`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sevi.contracts.adapters import Permit, PermitPurpose
from sevi.contracts.enums import KeyId


@dataclass
class _Permit:
    key_id: KeyId
    purpose: PermitPurpose


class RecordingSemaphore:
    """Never blocks; records every permit taken."""

    def __init__(self) -> None:
        self.taken: list[tuple[KeyId, str]] = []

    @asynccontextmanager
    async def acquire(
        self, key_id: KeyId, purpose: PermitPurpose, timeout_s: float
    ) -> AsyncIterator[Permit]:
        self.taken.append((key_id, purpose))
        yield _Permit(key_id, purpose)


class RecordingBreaker:
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


class RecordingBreakers:
    def __init__(self) -> None:
        self.by_key: dict[tuple[str, KeyId | None], RecordingBreaker] = {}

    def get(self, dep: str, key_id: KeyId | None) -> RecordingBreaker:
        return self.by_key.setdefault((dep, key_id), RecordingBreaker())
