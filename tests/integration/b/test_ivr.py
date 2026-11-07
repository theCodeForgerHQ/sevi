"""Twilio trial phone IVR (contracts §6.7, §8.6; B spec §8): rows I01-I15, IT-TWL-*, IT-IVR-01,
IT-WEB-01 and E2E-SIM-01, end to end through the FastAPI routes with ``SimPhone`` + fakes.

No real Twilio: ``FakeTwilioRest`` records calls and signs webhooks with a test token.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest
from phone_harness import BASE, PhoneRig, phone_rig, phone_settings, wav_pcm16

from sevi.app.ivr import wav_data_chunk
from sevi.app.settings import Settings
from sevi.contracts.copy import OutcomeVar
from sevi.contracts.enums import (
    Band,
    Channel,
    ConsentOutcome,
    EndReason,
    KeyId,
    Mode,
    NoBandReason,
    SessionState,
    Transducer,
    TruthRule,
)
from sevi.contracts.errors import ErrorCode
from sevi.core.stimulus import BankStimulusProvider
from sevi.core.wording import REDUCED_SCOPE_OUTCOMES

pytestmark = pytest.mark.asyncio

Bank = BankStimulusProvider


def code(r: Any) -> str:
    return str(r.json()["error"]["code"])


async def answered(rig: PhoneRig) -> tuple[str, Any]:
    """Dial and let Twilio fetch /twilio/voice (I03). Returns (sid, sim)."""
    r = await rig.dial()
    assert r.status_code == 201, r.text
    sid = r.json()["session_id"]
    sim = await rig.sim(answer=rig.perfect(sid))
    st = await sim.post(sim.call.url)
    assert st.status == 200
    return sid, sim


# ---------- E2E-SIM-01 / IT-TWL-01: the whole simulated phone test ----------
async def test_E2E_SIM_01_phone_keypad_happy_path_T11_NO_CELL(
    test_bank: Bank, tmp_path: Path
) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await rig.run_call()
        rt = rig.call(sid)
        rec = rt.rec
        assert rec.state is SessionState.ENDED
        assert rec.end_reason is EndReason.COMPLETED
        assert rec.channel is Channel.PHONE
        assert rec.transducer is Transducer.HANDSET
        assert rec.mode_effective is Mode.KEYPAD
        assert rec.key_id is KeyId.NONE
        d = rec.decision
        assert d is not None
        assert d.rule is TruthRule.T11
        assert d.band is Band.NO_BAND
        assert d.no_band_reason is NoBandReason.NO_CELL
        assert rec.consent is not None
        assert rec.consent.outcome is ConsentOutcome.GIVEN
        # the full playlist, then hangup
        assert sim.played[:3] == ["CONSENT_PUBLIC", "INSTRUCTIONS", "VOLUME_INTRO"]
        assert sim.played[-3:] == [f"WORDING_{d.wording_class.value}", "DISCLAIMER", "GOODBYE"]
        assert sim.hung_up
        assert set(sim.audio_status) == {200}
        assert "<Hangup />" in sim.steps[-1].twiml
        # every presentation is one trial in the store; same procedure as the browser
        stored = await rig.store.get_session_json(sid)
        assert stored is not None
        assert stored["state"] == "ENDED"
        assert rec.srt is not None
        # trial minutes ledger: 240 s -> 4 min, keyed by sha256(CallSid)
        assert await rig.store.trial_minutes_total() == 4.0
        assert rt.rules.count("I03") == 1
        assert "I04" in rt.rules
        assert "I08" in rt.rules


async def test_IT_TWL_01_dial_uses_verified_number_and_signed_urls(
    test_bank: Bank, tmp_path: Path
) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial()
        assert r.status_code == 201
        body = r.json()
        c = rig.twilio.created[-1]
        assert c.to_e164 == "+919800000000"
        assert c.session_id == body["session_id"]
        assert c.call_sid == body["call_sid"]
        assert c.url.startswith(f"{BASE}/twilio/voice/{body['session_id']}?t=")
        assert c.status_callback.startswith(f"{BASE}/twilio/status/{body['session_id']}?t=")
        rec = rig.call(body["session_id"]).rec
        assert rec.state is SessionState.DIALING
        assert rec.twilio_call_sid == c.call_sid
        assert rec.media_token_sha256 is not None
        # the number is never logged
        assert "+919800000000" not in repr(rig.tm.events)


async def test_I01_dial_is_idempotent_by_key(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        a = await rig.dial(key="same")
        b = await rig.dial(key="same")
        assert a.json() == b.json()
        assert len(rig.twilio.created) == 1
        r = await rig.client.post(
            "/op/api/calls", json={"participant_id": "P-01", "participant_type": "PILOT"}
        )
        assert r.status_code == 400
        assert code(r) == "BAD_REQUEST"


# ---------- disabled by default ----------
async def test_phone_disabled_by_default_404_and_PHONE_DISABLED(
    test_bank: Bank, tmp_path: Path
) -> None:
    assert Settings(env="test").phone.enabled is False
    assert Settings(env="test").twilio.record_answers is False
    async with phone_rig(test_bank, tmp_path, settings=phone_settings(enabled=False)) as rig:
        r = await rig.dial()
        assert r.status_code == 409
        assert code(r) == "PHONE_DISABLED"
        assert rig.twilio.created == []
        for path in ("/twilio/voice/x?t=y", "/twilio/step/x/c1?t=y", "/twilio/status/x?t=y"):
            assert (await rig.client.post(path)).status_code == 404
        assert (await rig.client.get("/audio/abcdefghijklmnop/GOODBYE.wav")).status_code == 404


# ---------- allowlist / trial safety ----------
async def test_IT_TWL_02_unverified_number_refused(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial(pid="P-02")
        assert r.status_code == 403
        assert code(r) == "ALLOWLIST_DENIED"
        assert rig.twilio.created == []
        assert rig.svc.calls == {}


async def test_verified_numbers_setting_validates_pairs() -> None:
    s = phone_settings()
    assert s.twilio.verified_numbers() == {"P-01": "+919800000000"}
    with pytest.raises(ValueError, match="P-NN"):
        phone_settings(verified_to_numbers=["+919800000000"])
    with pytest.raises(ValueError, match="duplicate"):
        phone_settings(verified_to_numbers=["P-01:+919800000002", "P-01:+919800000001"])


async def test_IT_TWL_03_call_cap_is_ten_minutes() -> None:
    assert phone_settings().twilio.time_limit_s == 600
    with pytest.raises(ValueError, match="time_limit_s"):
        phone_settings(time_limit_s=601)


async def test_IT_TWL_04_trial_minutes_stop_refuses_dial(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        await rig.store.trial_minutes_add("CA" + "1" * 32, 56)  # 56 + 10 worst case > 65
        r = await rig.dial()
        assert r.status_code == 409
        assert code(r) == "TWILIO_TRIAL_MINUTES_STOP"
        assert rig.twilio.created == []
        ok = await rig.dial(key="k2", interview=True)  # interview may use the reserve (<= 75)
        assert ok.status_code == 201


async def test_one_call_at_a_time(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        assert (await rig.dial()).status_code == 201
        r = await rig.dial(key="k2")
        assert r.status_code == 409
        assert code(r) == "SESSION_STATE_CONFLICT"


async def test_I02_dial_failure_and_maybe_created(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        rig.twilio.fail_create(21219)
        r = await rig.dial()
        assert r.status_code == 502
        assert code(r) == "TWILIO_DIAL_FAILED"
        (rt,) = rig.svc.calls.values()
        assert rt.rec.end_reason is EndReason.DIAL_FAILED
        assert rt.state is SessionState.ENDED
        rig.twilio.timeout_create()
        r = await rig.dial(key="k2")
        assert r.status_code == 502
        assert r.json()["error"]["details"]["maybe_created"] is True
        stuck = [c for c in rig.svc.calls.values() if c.state is SessionState.DIALING]
        assert len(stuck) == 1
        assert "I02" in stuck[0].rules


# ---------- webhook security ----------
async def test_IT_TWL_05_bad_signature_rejected(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial()
        sid = r.json()["session_id"]
        url = rig.twilio.created[-1].url
        bad = await rig.client.post(
            url, data={"CallSid": "CA" + "0" * 32}, headers={"X-Twilio-Signature": "nope"}
        )
        assert bad.status_code == 403
        assert code(bad) == "TWILIO_SIGNATURE_INVALID"
        missing = await rig.client.post(url, data={"CallSid": "x"})
        assert missing.status_code == 403
        # a valid signature over a tampered form fails too
        form = {"CallSid": "CA" + "0" * 32}
        sig = rig.twilio.sign(url, form)
        tampered = await rig.client.post(
            url, data={**form, "Digits": "1"}, headers={"X-Twilio-Signature": sig}
        )
        assert tampered.status_code == 403
        assert rig.call(sid).state is SessionState.DIALING  # no state change


async def test_IT_TWL_06_wrong_call_token_rejected(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial()
        sid = r.json()["session_id"]
        url = f"{BASE}/twilio/voice/{sid}?t=AAAAAAAAAAAAAAAAAAAAAA"
        resp = await rig.twilio.emit_webhook(rig.client, url, {"CallSid": "x"})
        assert resp.status_code == 403
        assert code(resp) == "TWILIO_TOKEN_INVALID"
        unknown = await rig.twilio.emit_webhook(
            rig.client, f"{BASE}/twilio/voice/nosuch?t=x", {"CallSid": "x"}
        )
        assert unknown.status_code == 404
        assert rig.call(sid).state is SessionState.DIALING


# ---------- consent rows ----------
async def test_I05_consent_declined_T10_no_disclaimer(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await answered(rig)
        st = await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={_t(sim)}", {"Digits": "2"})
        rec = rig.call(sid).rec
        assert rec.end_reason is EndReason.CONSENT_DECLINED
        assert rec.decision is not None
        assert rec.decision.rule is TruthRule.T10
        assert "DISCLAIMER" not in st.twiml
        assert "GOODBYE.wav" in st.twiml
        assert "WORDING_W_DECLINED" in st.twiml
        assert st.twiml.endswith("<Hangup /></Response>")


async def test_I06_I07_consent_no_reply_twice(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await answered(rig)
        t = _t(sim)
        st = await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={t}", {"Digits": "7"})
        assert f"/twilio/step/{sid}/c2?t=" in st.twiml
        assert "CONSENT_PUBLIC" in st.twiml
        st = await sim.post(f"{BASE}/twilio/step/{sid}/c2?t={t}")
        assert st.twiml.endswith("<Response><Hangup /></Response>")
        rec = rig.call(sid).rec
        assert rec.end_reason is EndReason.CONSENT_NO_REPLY
        assert rec.consent is not None
        assert rec.consent.outcome is ConsentOutcome.NO_REPLY
        assert rec.consent.repeats == 1


# ---------- IT-IVR-01: Twilio retries ----------
async def test_IT_IVR_01_replayed_step_identical_twiml_no_state_change(
    test_bank: Bank, tmp_path: Path
) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await answered(rig)
        t = _t(sim)
        first = await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={t}", {"Digits": "1"})
        pending = rig.call(sid).pending
        assert pending is not None
        digits = "".join(map(str, pending.triplet))
        url = f"{BASE}/twilio/step/{sid}/p{pending.presentation_id}?t={t}"
        a = await sim.post(url, {"Digits": digits})
        rt = rig.call(sid)
        snap = (rt.proc, list(rt.rules), rt.rec)
        b = await sim.post(url, {"Digits": digits})
        c = await sim.post(url, {"Digits": "999"})  # even a different body: cached
        assert a.twiml == b.twiml == c.twiml
        assert (rt.proc, rt.rules, rt.rec) == snap
        again = await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={t}", {"Digits": "2"})
        assert again.twiml == first.twiml
        assert rt.rec.end_reason is None


async def test_I09_stale_pid_returns_last_twiml(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await answered(rig)
        t = _t(sim)
        st = await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={t}", {"Digits": "1"})
        stale = await sim.post(f"{BASE}/twilio/step/{sid}/p77?t={t}", {"Digits": "123"})
        assert stale.twiml == st.twiml
        assert "I09" in rig.call(sid).rules


# ---------- IT-WEB-01: phone audio == browser bytes ----------
class SpyBank:
    """Delegates to the real bank and remembers every spec it rendered."""

    def __init__(self, bank: Bank) -> None:
        self.bank = bank
        self.manifest = bank.manifest
        self.specs: dict[int, Any] = {}

    def render_trial(self, spec: Any) -> Any:
        self.specs[spec.presentation_id] = spec
        return self.bank.render_trial(spec)


async def test_IT_WEB_01_audio_is_the_browser_stimulus_bytes(
    test_bank: Bank, tmp_path: Path
) -> None:
    spy = SpyBank(test_bank)
    async with phone_rig(spy, tmp_path) as rig:  # type: ignore[arg-type]
        sid, sim = await rig.run_call()
        assert len(spy.specs) > 20
        for pid, spec in spy.specs.items():
            body = sim.audio[f"s{pid:03d}"]
            # 44-byte mu-law WAV header: format 7, mono, 8000 Hz, 8 bit
            assert body[20:36] == struct.pack("<HHIIHH", 7, 1, 8000, 8000, 1, 8)
            # the data chunk is exactly what the browser transport sends for this spec
            assert wav_data_chunk(body) == test_bank.render_trial(spec).ulaw


# ---------- audio serving ----------
async def test_audio_token_name_and_ttl(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await rig.run_call()
        t = _t(sim)
        ok = await rig.client.get(f"/audio/{t}/GOODBYE.wav")
        assert ok.status_code == 200
        assert ok.headers["cache-control"] == "no-store"
        assert ok.headers["content-type"] == "audio/wav"
        assert (await rig.client.get(f"/audio/{t}/NOT_A_KEY.wav")).status_code == 404
        assert (await rig.client.get(f"/audio/{t}/s999.wav")).status_code == 404
        assert (await rig.client.get(f"/audio/{t}/bad-name.wav")).status_code == 404
        other = "B" * len(t)
        assert (await rig.client.get(f"/audio/{other}/GOODBYE.wav")).status_code == 404
        rig.clock.advance(rig.svc.settings.twilio.audio_ttl_s + 1)
        assert (await rig.client.get(f"/audio/{t}/GOODBYE.wav")).status_code == 404
        assert rig.svc.sweep() == 1
        assert sid not in rig.svc.calls


# ---------- I10: status callback ----------
async def test_I10_hangup_mid_test_classifies_once(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await answered(rig)
        await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={_t(sim)}", {"Digits": "1"})
        r = await sim.status("completed", 61)
        assert r.status_code == 204
        rec = rig.call(sid).rec
        assert rec.end_reason is EndReason.CALLER_HANGUP
        assert rec.state is SessionState.ENDED
        first = rec.decision
        assert first is not None
        assert first.band is not Band.PASS  # phone never PASSes
        dup = await rig.twilio.emit_webhook(rig.client, sim.call.status_callback, sim_last(sim))
        assert dup.status_code == 204
        await sim.status("completed", 61)  # new seq, same sid: minutes added once
        assert rig.call(sid).rec.decision == first
        assert await rig.store.trial_minutes_total() == 2.0
        decided = [e for e in rig.tm.names() if e == "result.decided"]
        assert len(decided) == 1


def sim_last(sim: Any) -> dict[str, str]:
    return {
        **sim._base_form(),
        "CallStatus": "completed",
        "SequenceNumber": "0",
        "CallDuration": "61",
    }


async def test_I10_no_answer_while_dialing_is_DIAL_FAILED(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial()
        sid = r.json()["session_id"]
        sim = await rig.sim()
        await sim.status("no-answer")
        rec = rig.call(sid).rec
        assert rec.end_reason is EndReason.DIAL_FAILED
        assert rec.decision is None
        assert await rig.store.trial_minutes_total() == 0.0


# ---------- I11-I13: dial timeout and operator abort ----------
async def test_I11_dial_timeout_cancels_call(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial()
        sid, call_sid = r.json()["session_id"], r.json()["call_sid"]
        await rig.svc.on_dial_timeout(sid)
        assert rig.twilio.updated == [(call_sid, "canceled")]
        assert rig.call(sid).rec.end_reason is EndReason.DIAL_FAILED


async def test_I12_I13_operator_abort(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        r = await rig.dial()
        sid, call_sid = r.json()["session_id"], r.json()["call_sid"]
        assert (await rig.client.post(f"/op/api/calls/{sid}/abort")).status_code == 204
        assert rig.twilio.updated == [(call_sid, "canceled")]
        assert "I12" in rig.call(sid).rules
        assert rig.call(sid).rec.end_reason is EndReason.OPERATOR_ABORT
        r = await rig.dial(key="k2")
        sid2 = r.json()["session_id"]
        sim = await rig.sim()
        await sim.post(sim.call.url)
        assert (await rig.client.post(f"/op/api/calls/{sid2}/abort")).status_code == 204
        assert rig.twilio.updated[-1] == (r.json()["call_sid"], "completed")
        rec = rig.call(sid2).rec
        assert rec.end_reason is EndReason.OPERATOR_ABORT
        assert rec.decision is not None
        assert (await rig.client.post("/op/api/calls/nosuch/abort")).status_code == 404


# ---------- I14 / I15 ----------
class ExplodingScorer:
    def score(self, spec: Any, ev: Any) -> Any:
        raise RuntimeError("boom")


async def test_I14_handler_exception_sysfail_T8(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path, scorer=ExplodingScorer()) as rig:
        sid, sim = await answered(rig)
        t = _t(sim)
        await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={t}", {"Digits": "1"})
        st = await sim.post(f"{BASE}/twilio/step/{sid}/p1?t={t}", {"Digits": "123"})
        assert "WORDING_W_SYSFAIL.wav" in st.twiml
        assert st.twiml.endswith("<Hangup /></Response>")
        rec = rig.call(sid).rec
        assert rec.end_reason is EndReason.SERVER_ERROR
        assert rec.decision is not None
        assert rec.decision.rule is TruthRule.T8
        assert "result.decided" not in rig.tm.names()  # t8_decision without classify()


async def test_I15_time_limit_ends_with_T7_playlist(test_bank: Bank, tmp_path: Path) -> None:
    async with phone_rig(test_bank, tmp_path) as rig:
        sid, sim = await answered(rig)
        t = _t(sim)
        await sim.post(f"{BASE}/twilio/step/{sid}/c1?t={t}", {"Digits": "1"})
        rig.clock.advance(600 - 30 + 1)
        st = await sim.post(f"{BASE}/twilio/step/{sid}/p1?t={t}", {"Digits": "123"})
        rec = rig.call(sid).rec
        assert rec.end_reason is EndReason.SESSION_TIMEOUT
        assert "I15" in rig.call(sid).rules
        assert "GOODBYE_EARLY.wav" in st.twiml
        assert st.twiml.endswith("<Hangup /></Response>")


# ---------- record mode (SEVI_TWILIO__RECORD_ANSWERS=true) ----------
async def test_record_mode_transcribes_and_deletes_every_recording(
    test_bank: Bank, tmp_path: Path
) -> None:
    calls: list[int] = []
    rig_ref: dict[str, Any] = {}

    async def transcribe(wav: bytes, lang: Any) -> str:
        calls.append(len(wav))
        if len(calls) == 2:
            raise RuntimeError("prisma down")  # STT_ERROR -> still deleted
        rt = next(iter(rig_ref["rig"].svc.calls.values()))
        return " ".join(str(d) for d in rt.pending.triplet)

    s = phone_settings(record_answers=True)
    async with phone_rig(test_bank, tmp_path, settings=s, transcribe=transcribe) as rig:
        rig_ref["rig"] = rig
        n = iter(range(10_000))

        def record(action: str) -> tuple[str, str, float]:
            i = next(n)
            url = f"https://api.twilio.test/Recordings/RE{i:032d}"
            rig.twilio.recordings[url] = wav_pcm16(0.8)
            return f"RE{i:032d}", url, 0.8

        r = await rig.dial()
        sid = r.json()["session_id"]
        sim = await rig.sim(answer=lambda a, k: "1", record=record)
        await sim.run()
        rec = rig.call(sid).rec
        assert rec.mode_requested is Mode.VOICE
        assert rec.key_id is KeyId.A
        assert rec.state is SessionState.ENDED
        assert rec.decision is not None
        assert rec.decision.band is not Band.PASS
        assert "<Record " in sim.steps[1].twiml
        assert "<Gather" not in sim.steps[1].twiml
        assert len(rig.twilio.fetched_recordings) >= 3
        # every recording seen is deleted, including the one whose transcription failed
        seen = {st.form["RecordingSid"] for st in sim.steps if "RecordingSid" in st.form}
        assert seen
        assert set(rig.twilio.deleted_recordings) == seen


async def test_record_mode_without_transcriber_is_keypad(test_bank: Bank, tmp_path: Path) -> None:
    s = phone_settings(record_answers=True)
    async with phone_rig(test_bank, tmp_path, settings=s) as rig:
        assert rig.svc.record_mode is False


# ---------- copy: PHONE stays BROWSER_ONLY on browser surfaces ----------
async def test_copy_PHONE_variable_stays_BROWSER_ONLY() -> None:
    assert REDUCED_SCOPE_OUTCOMES[OutcomeVar.PHONE.value] == "BROWSER_ONLY"


# ---------- helpers ----------
def _t(sim: Any) -> str:
    return str(sim.call.url.split("?t=", 1)[1])


__all__ = ["ErrorCode"]
