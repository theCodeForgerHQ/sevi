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


async def _session(srv, mode: str, timeout_s: float = 60.0) -> tuple[Browser, object]:  # type: ignore[no-untyped-def]
    b = Browser(srv, await admit(srv, mode))
    async with b.connect():
        t = await b.run(timeout_s=timeout_s)
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
    """Wiring smoke only: real WS -> transport -> uplink -> Prisma stream (fake) -> orchestrator.
    How many of the 24 answers land inside their windows depends on CPU scheduling against the
    10x session clock, so the exact count (24/24) is asserted only in the e2e suite (#72).
    `session.max_s` is raised so a heavily loaded machine cannot hit T_SESSION_MAX (720 s of
    session time = 72 s of wall time at 10x) before the 24 trials finish."""
    with serving(tmp_path, session={"max_s": 7200}) as srv:
        b, t = asyncio.run(_session(srv, "voice", timeout_s=180.0))
        assert t.last("result") is not None  # type: ignore[attr-defined]
        assert t.of("mode") == []  # type: ignore[attr-defined]  # never switched to keypad
        rec = b.session.rec
        assert rec.end_reason is EndReason.COMPLETED
        assert rec.mode_effective is Mode.VOICE
        assert b.session.voice_scored >= 1
        assert b.session.keypad_scored == 0


def test_version_reports_prompt_hashes(tmp_path: Path) -> None:
    async def get(srv) -> dict:  # type: ignore[no-untyped-def,type-arg]
        async with srv.http() as c:
            return (await c.get("/version")).json()  # type: ignore[no-any-return]

    with serving(tmp_path) as srv:
        v = asyncio.run(get(srv))
        assert set(v["prompts"]) == {"ta"}
        assert len(v["prompts"]["ta"]) == 64
