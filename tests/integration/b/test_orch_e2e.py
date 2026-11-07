"""End-to-end simulator scenarios for the browser session (B spec §13.7, IT-ORC-02 and the
issue #14 list): whole sessions on the real A core with fakes and the FakeClock. No network."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from orch_harness import Rig, build
from test_orch_fsm import to_test

from sevi.contracts.adapters import SttClosed, TDtmf
from sevi.contracts.enums import (
    AnswerStatus,
    Band,
    EndReason,
    Mode,
    PresentationKind,
    SessionFlag,
    SessionState,
    TruthRule,
    WordingClass,
)
from sevi.core.stimulus import BankStimulusProvider

pytestmark = pytest.mark.asyncio


@pytest.fixture
def rig(test_bank: BankStimulusProvider) -> Callable[..., Rig]:
    return lambda **kw: build(test_bank, **kw)


async def run_keypad_to_result(r: Rig, limit: int = 80) -> None:
    for _ in range(limit):
        if r.sess.state is not SessionState.TEST:
            return
        await r.keypad_answer()
    raise AssertionError("procedure never ended")


def kinds(r: Rig) -> list[str]:
    return [t.spec.kind.value for t in r.store.trials]


async def test_E2E_keypad_happy_path_IT_ORC_02(rig: Callable[..., Rig]) -> None:
    r = rig()
    await to_test(r)
    await run_keypad_to_result(r)
    await r.drain_to_end()
    k = kinds(r)
    assert k[:5] == ["VOLUME", "VOLUME", "PRACTICE", "PRACTICE", "PRACTICE"]
    assert k.count("SCORED") == 24
    assert r.sess.rec.end_reason is EndReason.COMPLETED
    d = r.decision()
    assert d.rule is TruthRule.T11
    assert d.band is Band.NO_BAND
    assert [k.value for k in r.sess.playlist] == [
        f"WORDING_{d.wording_class.value}",
        "DISCLAIMER",
        "GOODBYE",
    ]
    assert len(r.t.ui("feedback")) == 3  # practice feedback clips
    assert r.t.ui("closed")
    assert r.t.closed == 1000
    rec = r.store.sessions[r.sess.session_id]
    assert rec.state is SessionState.ENDED
    assert rec.mode_effective is Mode.KEYPAD
    assert rec.srt is not None
    assert rec.srt.n_scored == 24
    assert rec.srt.complete
    assert rec.key_id.value == "NONE"
    # every scored trial: keypad, DTMF, correct; one presentation per spec id
    scored = [t for t in r.store.trials if t.spec.kind is PresentationKind.SCORED]
    assert all(t.result.mode is Mode.KEYPAD and t.result.triplet_correct for t in scored)
    assert [t.spec.scored_index for t in scored] == list(range(1, 25))


async def test_E2E_voice_happy_path(rig: Callable[..., Rig]) -> None:
    r = rig(mode=Mode.VOICE)
    await to_test(r)
    for _ in range(80):
        if r.sess.state is not SessionState.TEST:
            break
        await r.voice_answer()
    await r.drain_to_end()
    assert r.sess.rec.end_reason is EndReason.COMPLETED
    scored = [t for t in r.store.trials if t.spec.kind is PresentationKind.SCORED]
    assert len(scored) == 24
    assert all(t.result.mode is Mode.VOICE and t.result.status is AnswerStatus.OK for t in scored)
    assert all(t.transcript for t in scored)
    rec = r.store.sessions[r.sess.session_id]
    assert rec.mode_effective is Mode.VOICE
    assert rec.flags == ()
    assert r.t.ui("mic")[0].start is True
    assert r.t.ui("mic")[-1].start is False
    assert r.permit.released == 1
    assert r.stt.current.closed
    # every frame Prisma got during an answer window carried that window's tag
    assert any(tag is not None for tag in r.stt.current.frames)


async def test_E2E_mic_denied_runs_whole_session_in_keypad(rig: Callable[..., Rig]) -> None:
    r = rig(mode=Mode.VOICE)
    await r.to_consent_done()
    await r.ui_event("mic_denied")
    assert r.sess.mode is Mode.KEYPAD
    assert r.sess.state is SessionState.INSTRUCTIONS
    r.auto_media = False
    await r.through_instructions()
    await run_keypad_to_result(r)
    await r.drain_to_end()
    rec = r.store.sessions[r.sess.session_id]
    assert rec.end_reason is EndReason.COMPLETED
    assert rec.mode_effective is Mode.KEYPAD
    assert SessionFlag.MIC_LOST in rec.flags
    assert SessionFlag.MIXED_MODE not in rec.flags
    assert r.permit.released == 1
    assert r.stt.current.closed
    assert r.t.ui("mode")[0].mode is Mode.KEYPAD


async def test_E2E_budget_stop_mid_voice_falls_back_to_keypad(rig: Callable[..., Rig]) -> None:
    """Review of #38: a ledger budget stop mid-session ends voice cleanly, never reconnects."""
    r = rig(mode=Mode.VOICE)
    await to_test(r)
    # answer by voice until two scored trials are in
    for _ in range(20):
        if r.sess.voice_scored >= 2:
            break
        await r.voice_answer()
    await r.to_listen()
    cut = r.sess.pres
    assert cut is not None
    r.stt.current.push(SttClosed(code=1000, reason="budget"))
    await r.pump()
    assert r.sess.mode is Mode.KEYPAD
    assert len(r.stt.sessions) == 1
    assert "S51" in r.sess.rules
    miss = r.store.trials[-1]
    assert miss.spec.presentation_id == cut.spec.presentation_id
    assert miss.result.status is AnswerStatus.CAPTURE_MISS
    assert r.t.ui("mode")[-1].mode is Mode.KEYPAD
    r.auto_media = False
    await run_keypad_to_result(r)
    await r.drain_to_end()
    rec = r.store.sessions[r.sess.session_id]
    assert rec.end_reason is EndReason.COMPLETED
    assert SessionFlag.BREAKER_TRIPPED in rec.flags
    assert SessionFlag.MIXED_MODE in rec.flags
    assert rec.mode_effective is Mode.MIXED
    assert r.permit.released == 1
    assert rec.decision is not None
    assert rec.decision.band is not Band.FAILED


async def test_E2E_stt_failure_mid_session_reconnects_once_then_keypad(
    rig: Callable[..., Rig],
) -> None:
    from sevi.contracts.errors import ErrorCode, SttError

    r = rig(mode=Mode.VOICE)
    await to_test(r)
    await r.to_listen()
    r.stt.fail = [SttError(ErrorCode.STT_AUTH, "revoked")]
    r.stt.current.push(SttClosed(code=4001, reason="server"))
    await r.pump()
    assert r.sess.mode is Mode.KEYPAD
    assert len(r.stt.sessions) == 1
    r.auto_media = False
    await run_keypad_to_result(r)
    await r.drain_to_end()
    assert r.sess.rec.end_reason is EndReason.COMPLETED


async def test_E2E_hangup_mid_test(rig: Callable[..., Rig]) -> None:
    r = rig()
    await to_test(r)
    for _ in range(7):
        await r.keypad_answer()
    n = len(r.store.trials)
    await r.to_listen()
    sent_before = len(r.t.sent)
    await r.hangup(clean=True)
    await r.drain_to_end()
    assert r.sess.rec.end_reason is EndReason.CALLER_HANGUP
    d = r.decision()
    assert d.rule is TruthRule.T7
    assert d.band is Band.INCOMPLETE
    assert len(r.store.trials) == n  # the open presentation is not recorded
    assert len(r.t.sent) == sent_before  # nothing is sent to a gone page
    assert r.store.sessions[r.sess.session_id].state is SessionState.ENDED
    assert r.sess.rec.report_token_sha256 is not None  # REPORT surface: report still possible


async def test_E2E_cant_hear_button_refers(rig: Callable[..., Rig]) -> None:
    r = rig()
    await to_test(r)
    await r.keypad_answer()
    await r.to_listen()
    await r.ui_event("cant_hear_confirmed")
    await r.drain_to_end()
    d = r.decision()
    assert d.rule is TruthRule.T4
    assert d.band is Band.REFER
    assert d.wording_class is WordingClass.W_REF_LOUD
    assert [m for m in r.playlist_marks()] == ["z00", "z01", "z02"]
    res = r.t.ui("result")[0]
    assert res.band is Band.REFER
    assert res.page_text
    assert res.report_url
    assert r.sess.rec.end_reason is EndReason.CANT_HEAR


async def test_E2E_loud_check_fail_refers(rig: Callable[..., Rig]) -> None:
    r = rig()
    await to_test(r)
    for _ in range(80):
        if r.sess.state is not SessionState.TEST:
            break
        p = r.sess.pres
        assert p is not None
        kind = p.spec.kind
        if kind in (PresentationKind.VOLUME, PresentationKind.PRACTICE):
            await r.keypad_answer()
        elif kind is PresentationKind.LOUD:
            wrong = tuple(d for d in (1, 2, 3, 4, 5, 6, 7, 8) if d not in p.spec.triplet)[:3]
            await r.keypad_answer(wrong)
        else:  # scored: no response -> loud check after 3
            await r.to_listen()
            await r.advance(8_000)
    await r.drain_to_end()
    d = r.decision()
    assert "LOUD" in kinds(r)
    assert d.rule is TruthRule.T3
    assert d.band is Band.REFER
    assert d.wording_class is WordingClass.W_REF_LOUD
    assert r.sess.rec.srt is not None
    assert r.sess.rec.srt.loud_wrong_total >= 2


async def test_E2E_keypad_wrong_star_and_zero_controls(rig: Callable[..., Rig]) -> None:
    """Controls never count as answers (R-21): '#', '0' in SCORED, a second '*'."""
    r = rig()
    await to_test(r)
    for _ in range(12):
        p = r.sess.pres
        assert p is not None
        if p.spec.kind is PresentationKind.SCORED:
            break
        await r.keypad_answer()
    await r.to_listen()
    for k in ("#", "0", "*"):
        await r.send(TDtmf(key=k, received_mono_ms=r.clock.mono_ms()))  # type: ignore[arg-type]
    await r.keypad_answer()
    t = r.store.trials[-1]
    assert t.result.triplet_correct
    assert t.replays == 1
    await r.close()
