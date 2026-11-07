"""#72: a task cancelled inside `SqliteStore.tx()` must not leave the shared connection in an open
transaction (the flaky wired-voice test crashed in RESULT with STORE_FAILED: "cannot start a
transaction within a transaction" after a ledger top-up was cancelled while closing STT)."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from sevi.app.store import SqliteStore


@pytest.mark.asyncio
@pytest.mark.parametrize("ticks", range(8))
async def test_tx_cancelled_at_any_point_leaves_no_open_transaction(
    tmp_path: Path, ticks: int
) -> None:
    store = await SqliteStore(tmp_path / "s.sqlite").open()
    try:

        async def write() -> None:
            async with store.tx() as db:
                await db.execute("SELECT 1")
                await asyncio.sleep(0)

        t = asyncio.create_task(write())
        for _ in range(ticks):
            await asyncio.sleep(0)
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
        await write()  # raised OperationalError before the fix (ticks=1)
        assert not store.db.in_transaction
    finally:
        await store.close()
