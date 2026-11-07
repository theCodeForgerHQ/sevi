"""The real orchestrator behind the app (#20 PR 1): admission -> WS -> transport -> orchestrator.

Smoke level only (the scenario suite is `test_e2e_app.py`): one keypad and one voice session
complete over a real localhost WebSocket in demo mode, and `/version` reports prompt hashes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from e2e_browser import Browser, admit, serving

from sevi.app.runtime import OrchestratorPort
from sevi.contracts.enums import EndReason, Mode


async def _session(srv, mode: str) -> tuple[Browser, object]:  # type: ignore[no-untyped-def]
    b = Browser(srv, await admit(srv, mode))
    async with b.connect():
        t = await b.run()
    return b, t


def test_wired_keypad_session_completes(tmp_path: Path) -> None:
    with serving(tmp_path) as srv:
        assert isinstance(srv.st.port, OrchestratorPort)
        b, t = asyncio.run(_session(srv, "keypad"))
        assert t.close_code == 1000  # type: ignore[attr-defined]
        assert t.last("result") is not None  # type: ignore[attr-defined]
        rec = b.session.rec
        assert rec.end_reason is EndReason.COMPLETED
        assert b.session.keypad_scored == 24


def test_wired_voice_session_runs_on_fake_prisma(tmp_path: Path) -> None:
    with serving(tmp_path) as srv:
        b, t = asyncio.run(_session(srv, "voice"))
        assert t.last("result") is not None  # type: ignore[attr-defined]
        assert b.session.rec.mode_effective is Mode.VOICE
        assert b.session.voice_scored == 24


def test_version_reports_prompt_hashes(tmp_path: Path) -> None:
    async def get(srv) -> dict:  # type: ignore[no-untyped-def,type-arg]
        async with srv.http() as c:
            return (await c.get("/version")).json()  # type: ignore[no-any-return]

    with serving(tmp_path) as srv:
        v = asyncio.run(get(srv))
        assert set(v["prompts"]) == {"ta"}
        assert len(v["prompts"]["ta"]) == 64
