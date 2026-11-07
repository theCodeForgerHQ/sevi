"""`app.main` wiring of the Twilio trial IVR (#19): mounted only when SEVI_PHONE__ENABLED."""

from __future__ import annotations

from pathlib import Path

from app_helpers import OP_HEADERS, running
from fastapi.testclient import TestClient

from sevi.app.ivr import PhoneService
from sevi.app.main import build_operator_app
from sevi.fakes.fake_twilio_rest import TEST_AUTH_TOKEN

PHONE = {
    "phone": {"enabled": True},
    "twilio": {
        "account_sid": "AC" + "0" * 32,
        "auth_token": TEST_AUTH_TOKEN,
        "from_number": "+15005550006",
        "verified_to_numbers": "P-01:+919800000000",
    },
}
DIAL = {"participant_id": "P-01", "participant_type": "PILOT"}


def _paths(app: object) -> set[str]:
    return {getattr(r, "path", "") for r in getattr(app, "routes", [])}


def test_phone_routes_absent_by_default(tmp_path: Path) -> None:
    with running(tmp_path) as (c, st, app):
        assert "phone" not in st.adapters
        assert "/twilio/voice/{sid}" not in _paths(app)
        assert c.post("/twilio/voice/x?t=y").status_code == 404
        assert c.get("/audio/abcdefghijklmnop/GOODBYE.wav").status_code == 404
        op = TestClient(build_operator_app(st), client=("127.0.0.1", 1))
        assert "/op/api/calls" not in _paths(op.app)
        assert op.post("/op/api/calls", json=DIAL, headers=OP_HEADERS).status_code == 404


def test_phone_enabled_mounts_routes_and_dials_through_operator_auth(tmp_path: Path) -> None:
    with running(tmp_path, **PHONE) as (c, st, app):
        svc = st.adapters["phone"]
        assert isinstance(svc, PhoneService)
        assert {"/twilio/voice/{sid}", "/audio/{t}/{name}.wav"} <= _paths(app)
        op = TestClient(build_operator_app(st), client=("127.0.0.1", 1))
        unauth = op.post("/op/api/calls", json=DIAL, headers={"Idempotency-Key": "a"})
        assert unauth.status_code in (401, 403)  # operator auth guards the dial
        r = op.post("/op/api/calls", json=DIAL, headers={**OP_HEADERS, "Idempotency-Key": "a"})
        assert r.status_code == 201, r.text
        sid = r.json()["session_id"]
        created = svc.deps.twilio.created[-1]  # demo mode dials through FakeTwilioRest
        url = created.url
        path = url[url.index("/twilio/") :]
        resp = c.post(
            path,
            data={"CallSid": created.call_sid},
            headers={
                "X-Twilio-Signature": svc.deps.twilio.sign(url, {"CallSid": created.call_sid})
            },
        )
        assert resp.status_code == 200, resp.text
        assert "CONSENT_PUBLIC.wav" in resp.text
        assert svc.calls[sid].state.value == "CONSENT"
        bad = c.post(path, data={"CallSid": "x"}, headers={"X-Twilio-Signature": "nope"})
        assert bad.status_code == 403
