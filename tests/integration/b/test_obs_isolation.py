"""IT-OBS-02: concurrent sessions never mix their contextvars in telemetry lines."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
import structlog

from sevi.app.store import SqliteStore
from sevi.app.telemetry import Telemetry
from sevi.contracts.telemetry import EventName
from sevi.fakes.clock import FakeClock

SIDS = (
    "01890a5d-ac96-774b-bcce-b302099a8057",
    "01890a5d-ac96-774b-8cce-b302099a8058",
)


@pytest.mark.asyncio
async def test_IT_OBS_02_contextvars_isolation(tmp_path: Path) -> None:
    tel = Telemetry(tmp_path / "logs", clock=FakeClock(), stdout=io.StringIO(), strict=True)
    writer = asyncio.create_task(tel.run())

    async def session_main(sid: str) -> None:
        structlog.contextvars.bind_contextvars(session_id=sid, channel="browser")
        for i in range(1, 51):
            structlog.contextvars.bind_contextvars(presentation_id=i)
            tel.emit(EventName.PRESENTATION_BEGIN, "INFO", owner=sid)
            await asyncio.sleep(0)

    await asyncio.gather(*(asyncio.create_task(session_main(s)) for s in SIDS))
    writer.cancel()
    tel.close()
    lines = [
        json.loads(x)
        for p in (tmp_path / "logs").glob("*.jsonl")
        for x in p.read_text().splitlines()
    ]
    assert len(lines) == 100
    for line in lines:
        assert line["session_id"] == line["fields"]["owner"]
        assert line["channel"] == "browser"
    assert structlog.contextvars.get_contextvars() == {}


@pytest.mark.asyncio
async def test_store_and_telemetry_share_a_data_dir(tmp_path: Path) -> None:
    """Store errors surface as `error` STORE_FAILED lines in the JSONL log."""
    out = io.StringIO()
    tel = Telemetry(tmp_path / "logs", clock=FakeClock(), stdout=out, strict=True)
    store = await SqliteStore(tmp_path / "data" / "sevi.sqlite", telemetry=tel).open()
    await store.db.execute("DROP TABLE ip_hits")
    with pytest.raises(Exception, match="ip_bucket_hit"):
        await store.ip_bucket_hit("x", 60)
    await store.close()
    tel.close()
    err = json.loads(out.getvalue().splitlines()[0])
    assert err["error_code"] == "STORE_FAILED"
