"""Whole-app E2E with fakes (#20; B spec §12, contracts §6, §9).

The real FastAPI app runs on uvicorn on 127.0.0.1 (lifespan on, real `startup()`); a scripted
browser drives complete sessions over a real WebSocket. Demo mode: fake Prisma and Evon,
synthetic test bank, placeholder prompts. No network beyond localhost, no keys.
Run alone with `make e2e`.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from e2e_browser import PAGES, Browser, E2EClock, Server, admit, create_body, serving

from sevi.app.main import StartupError, startup
from sevi.app.operator import hash_password
from sevi.app.settings import Settings
from sevi.contracts.enums import Band, EndReason, Mode, SessionState
from sevi.contracts.errors import ErrorCode

pytestmark = pytest.mark.e2e


async def _drive(srv: Server, mode: str, **run: Any) -> Browser:
    b = Browser(srv, await admit(srv, mode))
    async with b.connect():
        await b.run(**run)
    return b


def _ended(b: Browser) -> Any:
    """The client sees the close a moment before S73 finishes on the server thread."""
    sess = b.session
    for _ in range(250):
        if sess.state is SessionState.ENDED and sess.rec.ended_at is not None:
            break
        time.sleep(0.02)
    assert sess.state is SessionState.ENDED
    return sess.rec


# ---------- keypad ----------


def test_E2E_SIM_02_keypad_happy_result_and_report_expiry(tmp_path: Path) -> None:
    """24 keypad trials -> result payload == the stored decision; the report link works and
    expires after 72 h (wall clock jumped)."""
    clock = E2EClock()

    async def go(srv: Server) -> None:
        b = await _drive(srv, "keypad")
        rec = _ended(b)
        assert rec.end_reason is EndReason.COMPLETED
        assert b.session.keypad_scored == 24
        assert b.t.close_code == 1000
        res = b.t.last("result")
        assert res is not None
        d = rec.decision
        assert res["band"] == d.band.value
        assert res["wording_class"] == d.wording_class.value
        assert res["srt_display"] == d.srt_display
        assert res["page_text"]
        assert res["report_url"].startswith(f"{PAGES}/sevi/r.html#")
        assert b.t.last("closed") is not None
        token = res["report_url"].split("#", 1)[1]
        auth = {"Authorization": f"Bearer {token}"}
        async with srv.http() as c:
            r = await c.get("/api/v1/report", headers=auth)
            assert r.status_code == 200, r.text
            rep = r.json()
            assert rep["band"] == d.band.value
            assert len(rep["trials"]) >= 24
            clock.jump(hours=72, minutes=1)
            r = await c.get("/api/v1/report", headers=auth)
            assert r.status_code == 410
            assert r.json()["error"]["code"] == ErrorCode.REPORT_EXPIRED.value

    with serving(tmp_path, clock=clock) as srv:
        asyncio.run(go(srv))


def test_E2E_SIM_11_cant_hear_refers(tmp_path: Path) -> None:
    """`cant_hear_confirmed` during the test -> RESULT with REFER (T4), end reason CANT_HEAR."""
    sent = False

    async def hook(b: Browser, ui: dict[str, Any]) -> bool:
        nonlocal sent
        if ui["type"] == "phase" and ui.get("scored_index") == 2 and not sent:
            sent = True
            await b.ui("cant_hear_confirmed")
            return True
        return False

    with serving(tmp_path) as srv:
        b = asyncio.run(_drive(srv, "keypad", on_ui=hook))
        rec = _ended(b)
        assert sent
        assert rec.end_reason is EndReason.CANT_HEAR
        assert rec.decision.band is Band.REFER
        res = b.t.last("result")
        assert res is not None
        assert res["band"] == Band.REFER.value


# ---------- voice ----------


def test_E2E_SIM_22_voice_happy_via_fake_prisma(tmp_path: Path) -> None:
    with serving(tmp_path) as srv:
        b = asyncio.run(_drive(srv, "voice"))
        rec = _ended(b)
        assert rec.end_reason is EndReason.COMPLETED
        assert rec.mode_effective is Mode.VOICE
        assert b.session.voice_scored == 24
        assert b.session.keypad_scored == 0
        assert b.t.of("mode") == []
        assert b.t.media_out > 0
        assert b.t.last("result") is not None


def test_E2E_SIM_23_mic_denied_switches_to_keypad(tmp_path: Path) -> None:
    async def hook(b: Browser, ui: dict[str, Any]) -> bool:
        if ui["type"] == "mic" and ui.get("start"):
            await b.ui("mic_denied")
            return True  # never start the mic
        return False

    with serving(tmp_path) as srv:
        b = asyncio.run(_drive(srv, "voice", on_ui=hook))
        rec = _ended(b)
        assert [m["mode"] for m in b.t.of("mode")] == ["keypad"]
        assert rec.mode_effective is Mode.KEYPAD
        assert b.session.voice_scored == 0
        assert b.session.keypad_scored == 24
        assert "S28" in b.session.rules
        assert b.t.last("result") is not None


def test_E2E_budget_stop_mid_voice_switches_to_keypad(tmp_path: Path) -> None:
    """Ledger budget reached during the stream (top-up refused) -> SttClosed(budget) -> S51:
    the session continues and finishes in keypad. No credit is spent past the budget."""
    with serving(
        tmp_path, ledger={"credits_per_stt_s": 1.0, "budget_credits_per_key": 121.0}
    ) as srv:
        b = asyncio.run(_drive(srv, "voice"))
        rec = _ended(b)
        sess = b.session
        assert "S51" in sess.rules
        assert [m["mode"] for m in b.t.of("mode")] == ["keypad"]
        assert sess.voice_scored > 0
        assert sess.keypad_scored > 0
        assert sess.voice_scored + sess.keypad_scored == 24
        assert rec.end_reason is EndReason.COMPLETED
        assert b.t.last("result") is not None

        async def used() -> float:
            return await srv.st.ledger.used(rec.key_id)

        fut = asyncio.run_coroutine_threadsafe(used(), _server_loop(srv))
        assert fut.result(timeout=5) <= 121.0


def _server_loop(srv: Server) -> asyncio.AbstractEventLoop:
    """The uvicorn thread's loop (the store/ledger connection lives there)."""
    for t in srv.st.tasks:
        return t.get_loop()
    raise AssertionError("no server loop")


# ---------- admission / startup ----------


def test_IT_ADM_06_L4_draining_refuses(tmp_path: Path) -> None:
    async def go(srv: Server) -> None:
        srv.st.draining = True
        async with srv.http() as c:
            r = await c.post("/api/v1/sessions", json=create_body("keypad"))
            assert r.status_code == 503
            assert r.json()["error"]["code"] == ErrorCode.SERVICE_DEGRADED.value
            r = await c.get("/readyz")
            assert r.status_code == 503
            assert r.json()["status"] == "down"
        assert list(srv.st.port.all()) == []

    with serving(tmp_path) as srv:
        asyncio.run(go(srv))


def test_prod_refuses_test_bank(tmp_path: Path) -> None:
    """env=prod with the synthetic test bank: startup fails (exit 78 path), no app runs."""
    from e2e_browser import TEST_BANK_DIR

    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        env="prod",
        public_base_url="https://api.example.invalid",
        pages_origin="https://pages.example.invalid",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        gnani={"key_a": "not-a-real-key"},
        evon={"api_key": "not-a-real-key"},
        admission={"ip_hash_key": "k" * 32},
        operator={"password_scrypt": hash_password("not-a-real-password", salt=b"0" * 16)},
        allowlist={"fernet_key": "x" * 44, "hmac_key": "h" * 32},
        ledger={
            "credits_per_stt_s": 1.0,
            "credits_per_tts_char": 1.0,
            "voice_session_credits_est": 1.0,
            "budget_credits_per_key": 1.0,
        },
        result={"bank_dir": TEST_BANK_DIR, "enabled_langs": "ta", "experimental_langs": "en"},
    )
    with pytest.raises(StartupError) as ei:
        asyncio.run(startup(s, background=False))
    assert ei.value.code is ErrorCode.BANK_HASH_REJECTED
