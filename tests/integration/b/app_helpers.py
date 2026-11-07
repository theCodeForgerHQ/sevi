"""Shared helpers for B's app/API integration tests (TestClient + fakes, no network)."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sevi.app import main as app_main
from sevi.app.main import create_app
from sevi.app.operator import hash_password
from sevi.app.settings import Settings
from sevi.app.state import AppState
from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.enums import (
    Channel,
    KeyId,
    Lang,
    Mode,
    ParticipantType,
    RetentionClass,
    SessionState,
    Transducer,
)
from sevi.contracts.session import SessionRecord
from sevi.fakes.clock import FakeClock

_BANKS: dict[tuple[str, str, tuple[str, ...], int], Any] = {}
_real_load_banks = app_main.load_banks


def _cached_load_banks(s: Settings, telemetry: Any) -> Any:
    """Bank loading takes ~1 s; tests share one load per (dir, env, langs)."""
    langs = tuple(x.value for x in s.result.enabled_langs)
    key = (str(s.result.bank_dir), s.env, langs, s.prisma.sample_rate_hz)
    if key not in _BANKS:
        _BANKS[key] = _real_load_banks(s, telemetry)
    return dict(_BANKS[key])


app_main.load_banks = _cached_load_banks

REPO = Path(__file__).resolve().parents[3]
TEST_BANK_DIR = REPO / "stimuli" / "test-bank"
TEST_BANK_HASH = "2e2e98670db6df246ada022e0df5c378fd52c8fd90b0d1f02ed4f3d8a71938fb"
PAGES = "https://thecodeforgerhq.github.io"
OP_USER, OP_PASS = "sevi-op", "correct horse battery staple"
OP_HASH = hash_password(OP_PASS, salt=b"0123456789abcdef")
OP_HEADERS = {
    "Authorization": "Basic " + base64.b64encode(f"{OP_USER}:{OP_PASS}".encode()).decode(),
    "X-Sevi-Op": "1",
}
FWD = "203.0.113.9"


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def settings(tmp: Path, **over: Any) -> Settings:
    base: dict[str, Any] = {
        "env": "demo",
        "demo_mode": True,
        "data_dir": tmp / "data",
        "log_dir": tmp / "logs",
        "pages_origin": PAGES,
        "result": {
            "bank_dir": TEST_BANK_DIR,
            "enabled_langs": "ta",
            "experimental_langs": "en",
            "frozen_bank_hashes": {"ta": TEST_BANK_HASH},
        },
        "browser_voice": {"enabled": True},
        "operator": {"password_scrypt": OP_HASH},
        "admission": {"ip_hash_key": "k" * 32},
    }
    return Settings(_env_file=None, **_merge(base, over))  # type: ignore[call-arg]


@contextmanager
def running(
    tmp: Path, *, peer: str = "testclient", clock: FakeClock | None = None, **over: Any
) -> Iterator[tuple[TestClient, AppState, FastAPI]]:
    app = create_app(settings(tmp, **over), clock=clock)
    with TestClient(app, client=(peer, 50000)) as c:
        yield c, app.state.sevi, app


def body(mode: str = "keypad", *, earphones: bool = True, os: str = "Android") -> dict[str, Any]:
    return {
        "lang": "ta",
        "mode": mode,
        "earphones": earphones,
        "client": {
            "browser_family": "Chrome",
            "os_family": os,
            "ua": "Mozilla/5.0",
            "audio_worklet": True,
            "sample_rate_hz": 48000,
            "page_version": "abcdef123456",
        },
    }


def record(session_id: str, created_at: datetime, **kw: Any) -> SessionRecord:
    data: dict[str, Any] = {
        "session_id": session_id,
        "contracts_version": CONTRACTS_VERSION,
        "app_version": "0.0.1",
        "git_sha": "0" * 40,
        "channel": Channel.BROWSER,
        "mode_requested": Mode.KEYPAD,
        "mode_effective": Mode.KEYPAD,
        "transducer": Transducer.EARPHONES,
        "lang": Lang.TA,
        "participant_type": ParticipantType.PUBLIC,
        "participant_id": None,
        "pseudonym": None,
        "age_band": None,
        "key_id": KeyId.NONE,
        "state": SessionState.ENDED,
        "created_at": created_at,
        "started_at": created_at,
        "ended_at": created_at + timedelta(minutes=6),
        "end_reason": None,
        "consent": None,
        "d_est_ms": None,
        "t_guard_ms": 50,
        "echo_coupling_db": None,
        "bank_hash": TEST_BANK_HASH,
        "bank_version": "ta-test-0.1.0",
        "prompts_hash": "0" * 64,
        "retention": RetentionClass.DELETE_72H,
        "delete_after": created_at + timedelta(hours=72),
    }
    data.update(kw)
    return SessionRecord(**data)
