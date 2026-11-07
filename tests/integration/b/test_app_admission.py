"""Admission ladder, waiting room and per-IP limits through the HTTP API (B spec §9.2, §13.8)."""

from __future__ import annotations

from pathlib import Path

from app_helpers import FWD, PAGES, body, running

from sevi.app.admission import QueueEntry
from sevi.contracts.enums import AdmissionLevel, EndReason, KeyId
from sevi.contracts.wire import CreateSessionRequest
from sevi.fakes.clock import FakeClock


def _xff(ip: str = FWD) -> dict[str, str]:
    return {"X-Forwarded-For": f"1.2.3.4, {ip}"}


def test_IT_ADM_01_L0_keypad_and_voice_admitted(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        for _ in range(10):
            assert c.post("/api/v1/sessions", json=body()).json()["status"] == "ADMITTED"
        r = c.post("/api/v1/sessions", json=body("voice"))
        assert r.status_code == 201
        assert r.json()["status"] == "ADMITTED"
        assert r.json()["expires_in_s"] == 120
        assert c.get("/api/v1/status").json()["admission_level"] == "L0"
        assert st.admission.active_keypad() == 10
        assert st.admission.active_voice() == 1


def test_IT_ADM_02_L1_voice_queue_max_three(tmp_path: Path) -> None:
    with running(tmp_path) as (c, _, _):
        rs = [c.post("/api/v1/sessions", json=body("voice")) for _ in range(5)]
        assert rs[0].json()["status"] == "ADMITTED"
        assert [r.json()["queue_position"] for r in rs[1:4]] == [1, 2, 3]
        assert rs[1].json()["eta_s"] == 240
        assert rs[3].json()["eta_s"] == 720
        assert rs[4].status_code == 409
        assert rs[4].json()["error"]["code"] == "VOICE_UNAVAILABLE"
        assert c.post("/api/v1/sessions", json=body()).status_code == 201


def test_IT_ADM_03_queue_head_admitted_and_abandon(tmp_path: Path) -> None:
    clock = FakeClock()
    with running(tmp_path, clock=clock) as (c, st, _):
        first = c.post("/api/v1/sessions", json=body("voice")).json()
        q1 = c.post("/api/v1/sessions", json=body("voice")).json()
        q2 = c.post("/api/v1/sessions", json=body("voice")).json()
        auth1 = {"Authorization": f"Bearer {q1['ws_token']}"}
        assert (
            c.get(f"/api/v1/sessions/{q1['session_id']}", headers=auth1).json()["status"]
            == "QUEUED"
        )
        assert c.get(f"/api/v1/sessions/{q1['session_id']}").status_code == 401
        c.portal.call(st.port.end, first["session_id"], EndReason.CALLER_HANGUP)
        r = c.get(f"/api/v1/sessions/{q1['session_id']}", headers=auth1).json()
        assert r["status"] == "ADMITTED"
        # q2 is now head; it stops polling for > 30 s -> QUEUE_ABANDONED / EXPIRED
        clock.advance(31)
        c.portal.call(st.admission.sweep)
        assert st.admission.queue == []
        auth2 = {"Authorization": f"Bearer {q2['ws_token']}"}
        assert (
            c.get(f"/api/v1/sessions/{q2['session_id']}", headers=auth2).json()["status"]
            == "EXPIRED"
        )


def test_IT_ADM_04_L2_waiting_room(tmp_path: Path) -> None:
    with running(tmp_path, admission={"max_keypad_sessions": 3}) as (c, _, _):
        for _ in range(3):
            assert c.post("/api/v1/sessions", json=body()).status_code == 201
        r = c.post("/api/v1/sessions", json=body(), headers={"Origin": PAGES})
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "60"
        assert "Retry-After" in r.headers["Access-Control-Expose-Headers"]
        err = r.json()["error"]
        assert err["code"] == "ADMISSION_WAIT"
        assert err["details"]["ahead"] >= 1
        assert err["retry_after_s"] == 60
        assert c.post("/api/v1/sessions", json=body("voice")).status_code == 503
        assert c.get("/api/v1/status").json()["admission_level"] == "L2"


def test_IT_ADM_05_L3_public_stop(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        c.portal.call(st.ledger.mark_exhausted, KeyId.B)
        st.credit.invalidate()  # type: ignore[attr-defined]  # credit inputs are cached 5 s
        r = c.post("/api/v1/sessions", json=body("voice"))
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "VOICE_UNAVAILABLE"
        assert c.post("/api/v1/sessions", json=body()).status_code == 201
        s = c.get("/api/v1/status").json()
        assert s["voice_offered"] is False
        assert s["admission_level"] == "L3"
        c.portal.call(st.readiness.refresh)  # /readyz is cached 30 s
        assert c.get("/readyz").json() == {"status": "degraded"}


def test_IT_ADM_06_L4_draining(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        st.draining = True
        r = c.post("/api/v1/sessions", json=body())
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SERVICE_DEGRADED"
        assert r.json()["error"]["retry_after_s"] == 60
        r = c.get("/readyz")
        assert r.status_code == 503
        assert r.json() == {"status": "down"}
        assert c.portal.call(st.admission.level) is AdmissionLevel.L4
        st.draining = False


def test_U_ADM_01_level_table(tmp_path: Path) -> None:
    with running(tmp_path, admission={"max_keypad_sessions": 1}) as (c, st, _):
        adm = st.admission
        assert c.portal.call(adm.level) is AdmissionLevel.L0
        req = CreateSessionRequest.model_validate(body("voice"))
        adm.queue = [QueueEntry(f"q{i}", "0" * 64, req, 0.0) for i in range(4)]
        assert c.portal.call(adm.level) is AdmissionLevel.L1  # queue 4 only
        adm.queue = []
        c.post("/api/v1/sessions", json=body())
        assert c.portal.call(adm.level) is AdmissionLevel.L2
        st.switches.browser_voice = False
        assert c.portal.call(adm.level) is AdmissionLevel.L3  # switch off + keypad full
        st.switches.browser_voice = True
        st.breakers.get("prisma", KeyId.B).force_open()
        assert c.portal.call(adm.level) is AdmissionLevel.L3


def test_U_ADM_02_voice_offered_each_cause(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, _):
        adm = st.admission
        req = CreateSessionRequest.model_validate(body("voice"))
        adm.queue = [QueueEntry(f"q{i}", "0" * 64, req, 0.0) for i in range(3)]
        assert c.portal.call(adm.voice_offered) is True  # queue len 3 still offered
        adm.queue.append(QueueEntry("q3", "0" * 64, req, 0.0))
        assert c.portal.call(adm.voice_offered) is False
        adm.queue = []
        for name in ("browser_voice", "prisma"):
            setattr(st.switches, name, False)
            assert c.portal.call(adm.voice_offered) is False
            setattr(st.switches, name, True)
        st.breakers.get("prisma", st.settings.public_key).force_open()
        assert c.portal.call(adm.voice_offered) is False


def test_IT_ADM_07_cgnat_per_ip_keypad(tmp_path: Path) -> None:
    with running(tmp_path, peer="127.0.0.1") as (c, _, _):
        for _ in range(20):
            assert c.post("/api/v1/sessions", json=body(), headers=_xff()).status_code == 201
        r = c.post("/api/v1/sessions", json=body(), headers=_xff())
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "RATE_LIMITED_IP"
        assert r.json()["error"]["retry_after_s"] == 600
        assert c.post(
            "/api/v1/sessions", json=body(), headers=_xff("198.51.100.7")
        ).status_code == (201)


def test_IT_ADM_07_cf_connecting_ip_fallback(tmp_path: Path) -> None:
    with running(tmp_path, peer="127.0.0.1", client_ip_header="CF-Connecting-IP") as (c, _, _):
        h = {"CF-Connecting-IP": FWD}
        for _ in range(20):
            assert c.post("/api/v1/sessions", json=body(), headers=h).status_code == 201
        assert c.post("/api/v1/sessions", json=body(), headers=h).status_code == 429


def test_IT_ADM_08_voice_per_ip_separate_from_keypad(tmp_path: Path) -> None:
    with running(tmp_path, peer="127.0.0.1", gnani={"key_b_permits": 4}) as (c, _, _):
        for _ in range(3):
            assert c.post("/api/v1/sessions", json=body("voice"), headers=_xff()).status_code == 201
        assert c.post("/api/v1/sessions", json=body("voice"), headers=_xff()).status_code == 429
        assert c.post("/api/v1/sessions", json=body(), headers=_xff()).status_code == 201


def test_IT_ADM_09_capacity_rejection_not_counted(tmp_path: Path) -> None:
    with running(tmp_path, peer="127.0.0.1", admission={"max_keypad_sessions": 1}) as (c, st, _):
        assert c.post("/api/v1/sessions", json=body(), headers=_xff()).status_code == 201
        for _ in range(3):
            assert c.post("/api/v1/sessions", json=body(), headers=_xff()).status_code == 503
        rows = c.portal.call(st.store._all, "SELECT COUNT(*) AS n FROM ip_hits", ())
        assert rows[0]["n"] == 1


def test_U_ING_01_no_trusted_ip_uses_per_session_limits(tmp_path: Path) -> None:
    with running(tmp_path, peer="127.0.0.1", client_ip_header="") as (c, st, _):
        for _ in range(25):
            assert c.post("/api/v1/sessions", json=body(), headers=_xff()).status_code == 201
        assert st.admission.pre_start_cap(None) == 20
        assert st.admission._no_ip_warned


def test_IT_ADM_13_unconnected_keypad_frees_cap_after_20s(tmp_path: Path) -> None:
    clock = FakeClock()
    with running(tmp_path, clock=clock, admission={"max_keypad_sessions": 2}) as (c, st, _):
        for _ in range(2):
            c.post("/api/v1/sessions", json=body())
        assert c.post("/api/v1/sessions", json=body()).status_code == 503
        clock.advance(21)
        c.portal.call(st.admission.sweep)
        assert st.admission.active_keypad() == 0
        assert c.post("/api/v1/sessions", json=body()).status_code == 201


def test_IT_ADM_14_ios_speaker_voice(tmp_path: Path) -> None:
    with running(tmp_path) as (c, _, _):
        r = c.post("/api/v1/sessions", json=body("voice", earphones=False, os="iOS"))
        assert r.status_code == 409
        r = c.post("/api/v1/sessions", json=body("voice", earphones=True, os="iOS"))
        assert r.status_code == 201


def test_lang_unavailable(tmp_path: Path) -> None:
    with running(tmp_path) as (c, _, _):
        b = body()
        b["lang"] = "en"
        r = c.post("/api/v1/sessions", json=b)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "LANG_UNAVAILABLE"
        assert c.post("/api/v1/sessions", json={"lang": "ta"}).status_code == 422
