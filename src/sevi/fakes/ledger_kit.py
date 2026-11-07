"""The real persistent ledger on a throwaway SQLite file, plus read helpers, for tests and demo
runs (`Ledger` + `SqliteStore`, contracts §14.2). Nothing here fakes the credit rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sevi.adapters.ledger import Ledger
from sevi.app.settings import LedgerSettings
from sevi.app.store import SqliteStore
from sevi.contracts.adapters import Clock, LedgerEvent, Telemetry
from sevi.contracts.enums import KeyId
from sevi.fakes.clock import FakeClock
from sevi.fakes.null_telemetry import NullTelemetry


@dataclass
class LedgerKit:
    store: SqliteStore
    ledger: Ledger

    async def events(self) -> list[LedgerEvent]:
        async with self.store.db.execute("SELECT event_json FROM ledger_events ORDER BY at") as c:
            rows = await c.fetchall()
        return [LedgerEvent.model_validate(json.loads(r[0])) for r in rows]

    async def reservations(self) -> list[dict[str, Any]]:
        async with self.store.db.execute(
            "SELECT reservation_id, key_id, purpose, credits FROM ledger_reservations"
        ) as c:
            rows = await c.fetchall()
        return [dict(r) for r in rows]

    async def exhausted(self, key_id: KeyId) -> bool:
        return await self.ledger.is_exhausted(key_id)

    async def close(self) -> None:
        await self.store.close()


async def open_ledger(
    path: Path,
    *,
    budget: float = 1000.0,
    credits_per_stt_s: float = 1.0,
    credits_per_tts_char: float = 0.01,
    alias_b: bool = False,
    clock: Clock | None = None,
    telemetry: Telemetry | None = None,
) -> LedgerKit:
    clk = clock or FakeClock()
    cfg = LedgerSettings(
        budget_credits_per_key=budget,
        credits_per_stt_s=credits_per_stt_s,
        credits_per_tts_char=credits_per_tts_char,
    )
    store = await SqliteStore(path, clock=clk).open()
    led = Ledger(store, cfg, alias_b=alias_b, clock=clk, telemetry=telemetry or NullTelemetry())
    return LedgerKit(store, led)
