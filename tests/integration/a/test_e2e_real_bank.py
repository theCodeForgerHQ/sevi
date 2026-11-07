"""E2E on the SHIPPED Tamil and English banks (frozen, hash-pinned, no unfrozen fallback).

Skips when `stimuli/<lang>/manifest.json` is absent. No network beyond localhost, no keys.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from e2e_browser import REPO, Browser, Server, admit, serving

from sevi.contracts.enums import EndReason, Lang, SessionState
from sevi.core.bank import load_bank

pytestmark = pytest.mark.e2e

STIMULI = REPO / "stimuli"
LANGS = (Lang.TA, Lang.EN)


def _hash(lang: Lang) -> str:
    m = STIMULI / lang.value / "manifest.json"
    if not m.exists():
        pytest.skip(f"no shipped {lang.value} bank")
    doc = json.loads(m.read_text(encoding="utf-8"))
    assert doc["frozen"] is True
    return str(doc["bank_hash"])


@pytest.mark.parametrize("lang", LANGS)
def test_shipped_bank_loads_frozen_in_prod(lang: Lang) -> None:
    """The shipped bank passes every sha256 + bank_hash check with prod=True."""
    h = _hash(lang)
    load_bank(lang, str(STIMULI), {lang: h}, False, prod=True)


def test_E2E_keypad_session_on_shipped_ta_bank(tmp_path: Path) -> None:
    """24 keypad trials over the real Tamil bank -> COMPLETED with a stored decision."""
    hashes = {lg.value: _hash(lg) for lg in LANGS}

    async def go(srv: Server) -> Any:
        b = Browser(srv, await admit(srv, "keypad"))
        async with b.connect():
            await b.run()
        return b

    over: dict[str, Any] = {
        "result": {
            "bank_dir": STIMULI,
            "enabled_langs": "ta,en",
            "allow_unfrozen_bank": False,
            "frozen_bank_hashes": hashes,
        }
    }
    with serving(tmp_path, **over) as srv:
        b = asyncio.run(go(srv))
        sess = b.session
        for _ in range(250):
            if sess.state is SessionState.ENDED and sess.rec.ended_at is not None:
                break
            time.sleep(0.02)
        assert sess.rec.end_reason is EndReason.COMPLETED
        assert sess.keypad_scored == 24
        assert b.t.last("result") is not None
