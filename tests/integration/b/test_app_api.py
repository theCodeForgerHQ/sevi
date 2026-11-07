"""App startup, readiness, report, CORS, operator API and WS accept (B spec §2.2, §9; §13.8)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from app_helpers import (
    OP_HASH,
    OP_HEADERS,
    PAGES,
    TEST_BANK_HASH,
    body,
    record,
    running,
    settings,
)
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sevi.app.main import StartupError, build_operator_app, serve, startup
from sevi.app.settings import EXIT_CONFIG_INVALID
from sevi.app.state import AppState
from sevi.contracts.adapters import LedgerEvent
from sevi.contracts.enums import EndReason, KeyId, SessionState, Surface
from sevi.contracts.errors import ErrorCode
from sevi.contracts.wire import ReadyzResponse
from sevi.core.result import classify
from sevi.fakes.clock import FakeClock
from sevi.fakes.scenarios.result_inputs import inp

ORIGIN = {"origin": PAGES}


def _sid() -> str:
    h = uuid.uuid4().hex
    return f"{h[:8]}-{h[8:12]}-7{h[13:16]}-8{h[17:20]}-{h[20:32]}"


def _prod(tmp: Path, **over: Any) -> Any:
    return settings(
        tmp,
        env="prod",
        demo_mode=False,
        public_base_url="https://sevi.tail1234.ts.net",
        gnani={"key_a": "dummy-a"},
        evon={"api_key": "dummy-evon"},
        allowlist={"fernet_key": "f" * 44, "hmac_key": "h" * 32},
        ledger={
            "credits_per_stt_s": 0.118,
            "credits_per_tts_char": 0.01,
            "voice_session_credits_est": 43,
            "budget_credits_per_key": 10,
        },
        **over,
    )


# ---------- startup ----------


def test_startup_demo_mode_with_fakes(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").json() == {"status": "ok"}
        assert st.startup_retention_done
        assert type(st.adapters["stt"]).__name__ == "PrismaStreamClient"
        assert st.adapters["stt"].opts.ws_url.startswith("ws://127.0.0.1:")  # FakePrisma
        assert st.stub_factories == []
    assert st.closers == []  # shutdown closed the store, the fakes and the HTTP client


def test_prod_startup_refuses_test_bank(tmp_path: Path) -> None:
    s = _prod(tmp_path)

    async def go() -> None:
        await startup(s, background=False)

    with pytest.raises(StartupError) as ei:
        asyncio.run(go())
    assert ei.value.code is ErrorCode.BANK_HASH_REJECTED
    assert asyncio.run(serve(s)) == EXIT_CONFIG_INVALID


def test_demo_bank_config_mismatch_exits(tmp_path: Path) -> None:
    s = settings(tmp_path, prisma={"sample_rate_hz": 16000})
    with pytest.raises(StartupError) as ei:
        asyncio.run(startup(s, background=False))
    assert ei.value.code is ErrorCode.BANK_CONFIG_MISMATCH


def test_one_worker_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(StartupError):
        asyncio.run(startup(settings(tmp_path), background=False))


# ---------- readiness / version ----------


def test_IT_API_06_public_readyz_only_status(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        assert set(c.get("/readyz").json()) == {"status"}
        st.startup_retention_done = False
        assert c.get("/readyz").status_code == 503
        st.startup_retention_done = True
        op = TestClient(build_operator_app(st), client=("127.0.0.1", 1))
        r = op.get("/op/api/readyz", headers=OP_HEADERS)
        assert r.status_code == 200
        assert set(r.json()) == set(ReadyzResponse.model_fields)
        assert r.json()["active_sessions"] == {"keypad": 0, "voice": 0, "phone": 0}
        assert "exposure_paused" in r.json()["credits"]["A"]  # Key B aliases Key A
        assert c.get("/op/api/readyz").status_code == 404  # not on the public listener


def test_IT_API_09_version(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        v = c.get("/version").json()
        assert v["banks"] == {"ta": TEST_BANK_HASH}
        assert v["git_sha"] == st.git_sha
        assert len(v["copy_sha256"]) == 64


def test_IT_API_10_status_cached_5s(tmp_path: Path) -> None:
    clock = FakeClock()
    with running(tmp_path, clock=clock) as (c, st, _):
        a = c.get("/api/v1/status").json()
        st.switches.browser_voice = False
        assert c.get("/api/v1/status").json() == a
        clock.advance(5.1)
        assert c.get("/api/v1/status").json()["voice_offered"] is False


# ---------- HTTP plumbing ----------


def test_IT_API_03_cors_pages_origin_only(tmp_path: Path) -> None:
    with running(tmp_path) as (c, _, _):
        r = c.get("/api/v1/status", headers=ORIGIN)
        assert r.headers["access-control-allow-origin"] == PAGES
        r = c.get("/api/v1/status", headers={"origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers
        pre = c.options(
            "/api/v1/sessions",
            headers={**ORIGIN, "access-control-request-method": "POST"},
        )
        assert pre.headers["access-control-allow-origin"] == PAGES


def test_IT_API_04_413_and_500_carry_cors(tmp_path: Path) -> None:
    with running(tmp_path) as (c, _, app):

        async def boom() -> None:
            raise RuntimeError("kaboom")

        app.add_api_route("/api/v1/boom", boom)
        r = c.post(
            "/api/v1/sessions",
            content=b"x" * (17 * 1024),
            headers={**ORIGIN, "content-type": "application/json"},
        )
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "BAD_REQUEST"
        assert r.headers["access-control-allow-origin"] == PAGES
        r = c.get("/api/v1/boom", headers=ORIGIN)
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "INTERNAL"
        assert r.headers["access-control-allow-origin"] == PAGES


def test_IT_API_08_security_headers(tmp_path: Path) -> None:
    with running(tmp_path) as (c, _, _):
        for path in ("/healthz", "/api/v1/status", "/nope"):
            assert c.get(path).headers["referrer-policy"] == "no-referrer"
        assert c.get("/api/v1/status").headers["cache-control"] == "no-store"
        assert c.get("/healthz", headers={"X-Request-ID": "abc-1"}).headers["x-request-id"] == (
            "abc-1"
        )


# ---------- report ----------


def _store_report(st: AppState, c: TestClient, *, decision: Any, age: timedelta) -> str:
    token = secrets.token_urlsafe(16)
    rec = record(
        _sid(),
        st.clock.now_utc() - age,
        srt=inp().srt,
        decision=decision,
        end_reason=EndReason.COMPLETED,
        report_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
    )
    c.portal.call(st.store.put_session, rec)
    return token


def test_IT_API_05_report_token_and_expiry(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        d = classify(inp())
        assert Surface.REPORT in d.surfaces
        good = _store_report(st, c, decision=d, age=timedelta(hours=1))
        live = _store_report(st, c, decision=None, age=timedelta(minutes=1))
        no_rep = d.model_copy(update={"surfaces": (Surface.SPOKEN, Surface.PAGE)})
        t10 = _store_report(st, c, decision=no_rep, age=timedelta(hours=1))
        old = _store_report(st, c, decision=d, age=timedelta(hours=72, seconds=1))

        def get(tok: str | None) -> Any:
            h = {"Authorization": f"Bearer {tok}"} if tok else {}
            return c.get("/api/v1/report", headers=h)

        assert get(None).status_code == 401
        assert get(secrets.token_urlsafe(16)).status_code == 410
        assert get(live).status_code == 404
        assert get(t10).status_code == 404
        assert get(old).status_code == 410
        r = get(good)
        assert r.status_code == 200
        j = r.json()
        assert j["band"] == "PASS"
        assert j["page_text"]
        assert j["disclaimer"]
        assert j["evon_text"] is None
        assert j["expires_at"] > j["created_at"]
        assert r.headers["cache-control"] == "no-store"


# ---------- operator ----------


def _op(st: AppState, peer: str = "127.0.0.1") -> TestClient:
    return TestClient(build_operator_app(st), client=(peer, 1))


def test_IT_API_07_operator_auth(tmp_path: Path) -> None:
    with running(tmp_path) as (_, st, _):
        op = _op(st)
        r = op.get("/op/api/sessions")
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Basic")
        bad = "Basic " + base64.b64encode(b"sevi-op:wrong").decode()
        assert op.get("/op/api/sessions", headers={"Authorization": bad}).status_code == 401
        other = "Basic " + base64.b64encode(b"root:correct horse battery staple").decode()
        assert op.get("/op/api/sessions", headers={"Authorization": other}).status_code == 401
        assert op.get("/op/api/sessions", headers=OP_HEADERS).status_code == 200
        assert _op(st, "8.8.8.8").get("/op/api/sessions", headers=OP_HEADERS).status_code == 403
        for h in ("X-Forwarded-For", "CF-Connecting-IP", "Cf-Ray"):
            r = op.get("/op/api/sessions", headers={**OP_HEADERS, h: "1.2.3.4"})
            assert r.status_code == 403
            assert r.json()["error"]["code"] == "FORBIDDEN_NETWORK"


def test_operator_no_hash_configured_fails_closed(tmp_path: Path) -> None:
    with running(tmp_path, operator={"password_scrypt": None}) as (_, st, _):
        assert _op(st).get("/op/api/sessions", headers=OP_HEADERS).status_code == 401
    assert OP_HASH.startswith("scrypt$")


def test_IT_API_11_operator_csrf(tmp_path: Path) -> None:
    with running(tmp_path) as (_, st, _):
        op = _op(st)
        sw = {"name": "evon", "value": True}
        r = op.post("/op/api/switches", json=sw, headers={**OP_HEADERS, "Origin": PAGES})
        assert r.status_code == 403
        no_hdr = {"Authorization": OP_HEADERS["Authorization"]}
        assert op.post("/op/api/switches", json=sw, headers=no_hdr).status_code == 403
        assert op.post("/op/api/switches", json=sw, headers=OP_HEADERS).status_code == 200


def test_IT_OP_01_switch_browser_voice_off(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        earlier = c.post("/api/v1/sessions", json=body("voice")).json()
        r = _op(st).post(
            "/op/api/switches", json={"name": "browser_voice", "value": False}, headers=OP_HEADERS
        )
        assert r.json()["switches"]["browser_voice"] is False
        assert c.post("/api/v1/sessions", json=body("voice")).status_code == 409
        h = st.port.get(earlier["session_id"])
        assert h is not None
        assert h.state is SessionState.AWAIT_START  # created earlier: unaffected


def test_IT_OP_02_station_idempotent(tmp_path: Path) -> None:
    with running(tmp_path) as (_, st, _):
        op = _op(st)
        h = {**OP_HEADERS, "Idempotency-Key": "station-1"}
        a = op.post("/op/api/stations/sessions", json={"lang": "ta"}, headers=h)
        assert a.status_code == 201
        assert "#s=" in a.json()["test_url"]
        assert "&k=" in a.json()["test_url"]
        b = op.post("/op/api/stations/sessions", json={"lang": "ta"}, headers=h)
        assert a.json() == b.json()
        en = op.post(
            "/op/api/stations/sessions",
            json={"lang": "en"},
            headers={**OP_HEADERS, "Idempotency-Key": "s2"},
        )
        assert en.status_code == 409
        assert op.post("/op/api/stations/sessions", json={}, headers=OP_HEADERS).status_code == 422


def test_operator_sessions_abort_ledger(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        sid = c.post("/api/v1/sessions", json=body()).json()["session_id"]
        op = _op(st)
        rows = op.get("/op/api/sessions?limit=50", headers=OP_HEADERS).json()
        assert [r["session_id"] for r in rows] == [sid]
        r = op.post(f"/op/api/sessions/{sid}/abort", headers=OP_HEADERS)
        assert r.json()["end_reason"] == "OPERATOR_ABORT"
        assert op.post(f"/op/api/sessions/{sid}/abort", headers=OP_HEADERS).status_code == 200
        assert op.post("/op/api/sessions/nope/abort", headers=OP_HEADERS).status_code == 404
        assert op.get("/op/api/participants", headers=OP_HEADERS).json() == []
        assert "keys" in op.get("/op/api/ledger", headers=OP_HEADERS).json()
        ev = LedgerEvent(
            event_id=uuid.uuid4().hex,
            key_id=KeyId.A,
            bucket="OPS",
            purpose="vt",
            window_id="W1",
            origin="manual",
            dashboard_used=12.5,
            at=st.clock.now_utc(),
        )
        body_ = ev.model_dump(mode="json")
        r = op.post("/op/api/ledger/reconcile", json=body_, headers=OP_HEADERS)
        assert r.status_code == 200
        assert r.json()["inserted"] == 1
        again = op.post("/op/api/ledger/reconcile", json=body_, headers=OP_HEADERS)
        assert again.json()["inserted"] == 0
        server = {**body_, "event_id": uuid.uuid4().hex, "origin": "server"}
        assert op.post("/op/api/ledger/reconcile", json=server, headers=OP_HEADERS).status_code == (
            422
        )


# ---------- WS accept (S13) ----------


def _ws_close_code(c: TestClient, url: str, headers: dict[str, str]) -> int:
    with pytest.raises(WebSocketDisconnect) as ei, c.websocket_connect(url, headers=headers) as ws:
        ws.receive_text()
    return ei.value.code


def test_IT_ADM_11_ws_accept_rules(tmp_path: Path) -> None:
    with running(tmp_path, peer="127.0.0.1") as (c, st, _):
        assert _ws_close_code(c, "/web/stream", ORIGIN) == 4401
        assert _ws_close_code(c, "/web/stream?sid=unknown", ORIGIN) == 4401
        xff = {"X-Forwarded-For": "203.0.113.9"}
        s1, s2, s3 = (c.post("/api/v1/sessions", json=body(), headers=xff).json() for _ in range(3))
        bad_origin = {"origin": "https://evil.example"}
        assert _ws_close_code(c, f"/web/stream?sid={s1['session_id']}", bad_origin) == 4401
        h = {**ORIGIN, **xff}
        with c.websocket_connect(f"/web/stream?sid={s1['session_id']}", headers=h):
            # one socket per session
            assert _ws_close_code(c, f"/web/stream?sid={s1['session_id']}", h) == 4409
            with c.websocket_connect(f"/web/stream?sid={s2['session_id']}", headers=h):
                # third pre-start socket from one bucket
                assert _ws_close_code(c, f"/web/stream?sid={s3['session_id']}", h) == 4429
        assert st.admission.pre_start_cap(None) == 20


def test_ws_start_token_checked(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        s = c.post("/api/v1/sessions", json=body()).json()
        url = f"/web/stream?sid={s['session_id']}"
        start = {
            "event": "start",
            "sequenceNumber": "1",
            "start": {"streamSid": "x", "customParameters": {}},
        }
        with c.websocket_connect(url, headers=ORIGIN) as ws:
            start["start"]["customParameters"] = {"token": "A" * 22}  # type: ignore[index]
            ws.send_json(start)
            assert '"code":"SESSION_TOKEN_INVALID"' in ws.receive_text().replace(" ", "")
            with pytest.raises(WebSocketDisconnect) as ei:
                ws.receive_text()
            assert ei.value.code == 4401
        h = st.port.get(s["session_id"])
        assert h is not None
        assert h.state is SessionState.ENDED
