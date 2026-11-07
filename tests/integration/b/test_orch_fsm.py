"""Orchestrator FSM rows (contracts §6.2 session FSM, §6.3 presentation FSM, §6.5 playlist).

Every row is driven through the real A core (staircase, test bank, scorer, classifier) with
in-memory fakes and the FakeClock. Rule ids appear in the parametrize ids and test names; each
test also asserts the row id was recorded by ``BrowserSession.rule``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from orch_harness import Rig, build

from sevi.app.orchestrator import BreakerOpened, OperatorAbort
from sevi.contracts.adapters import (
    SttClosed,
    SttErrorMsg,
    SttProcessing,
    SttTranscript,
    TDtmf,
)
from sevi.contracts.enums import (
    AnswerStatus,
    Band,
    Condition,
    EndReason,
    EvonOutcome,
    KeyId,
    Mode,
    PresentationKind,
    PresentationState,
    SessionFlag,
    SessionState,
    SystemEvent,
    TruthRule,
    WindowCloseReason,
)
from sevi.contracts.errors import ErrorCode, SttError
from sevi.core.stimulus import BankStimulusProvider

pytestmark = pytest.mark.asyncio

S = SessionState
P = PresentationState


# ---------- drivers ----------
async def to_instructions(r: Rig) -> None:
    await r.to_consent_done()
    if r.sess.mode is Mode.VOICE:
        await r.media(35)  # first media arms T_FLOOR (600 ms) -> S27


async def to_test(r: Rig) -> None:
    await to_instructions(r)
    await r.through_instructions()
    assert r.sess.state is S.TEST


async def to_scored(r: Rig) -> None:
    """Answer volume + practice correctly until the first SCORED presentation begins."""
    await to_test(r)
    for _ in range(12):
        p = r.sess.pres
        assert p is not None
        if p.spec.kind is PresentationKind.SCORED:
            return
        await r.keypad_answer()
    raise AssertionError("no scored presentation")


def tx(r: Rig, text: str, *, bound: Any = "current", index: int = 1) -> SttTranscript:
    return SttTranscript(
        text=text,
        segment_id=f"g{index}",
        segment_index=index,
        audio_duration_ms=600,
        latency_ms=300,
        detected_language_code="ta-IN",
        received_mono_ms=r.clock.mono_ms(),
        bound=r.sess.current_tag if bound == "current" else bound,
    )


async def stt_push(r: Rig, ev: Any) -> None:
    r.stt.current.push(ev)
    await r.pump()


@pytest.fixture
def kp(test_bank: BankStimulusProvider) -> Callable[..., Rig]:
    return lambda **kw: build(test_bank, **kw)


@pytest.fixture
def vc(test_bank: BankStimulusProvider) -> Callable[..., Rig]:
    return lambda **kw: build(test_bank, mode=Mode.VOICE, **kw)


# =====================================================================================
# Session FSM (contracts §6.2), simple rows as a table: (row, mode, drive, checks)
# =====================================================================================
Drive = Callable[[Rig], Awaitable[None]]


async def d_accept(r: Rig) -> None:
    await r.sess.accept(r.t)  # type: ignore[arg-type]


async def d_bad_token(r: Rig) -> None:
    r.token = "x" * 22
    await r.start()


async def d_t_start(r: Rig) -> None:
    await d_accept(r)
    await r.advance(5_001)


async def d_closed_await(r: Rig) -> None:
    await d_accept(r)
    await r.hangup(clean=False)


async def d_rx_idle(r: Rig) -> None:
    await r.start()
    await r.advance(36_000)


async def d_declined(r: Rig) -> None:
    await r.start()
    await r.ui_event("consent_declined")
    await r.drain_to_end()


async def d_consent_timeout(r: Rig) -> None:
    await r.start()
    await r.finish_clip()
    await r.wait(300_001)
    await r.drain_to_end()


async def d_cant_hear(r: Rig) -> None:
    await to_test(r)
    await r.keypad_answer()
    await r.ui_event("cant_hear_confirmed")
    await r.drain_to_end()


async def d_hangup_test(r: Rig) -> None:
    await to_test(r)
    await r.hangup(clean=True)
    await r.drain_to_end()


async def d_lost_test(r: Rig) -> None:
    await to_test(r)
    await r.hangup(clean=False)
    await r.drain_to_end()


async def d_session_max(r: Rig) -> None:
    await to_test(r)
    assert r.sess.rec.started_at is not None
    armed = r.sess.timers.armed("T_SESSION_MAX")
    assert armed is not None
    assert armed - r.clock.mono_ms() > 700_000
    r.sess.timers.arm("T_SESSION_MAX", r.clock.mono_ms() + 1)  # skip ahead 720 s
    await r.advance(2)
    await r.drain_to_end()


async def d_abort_test(r: Rig) -> None:
    await to_test(r)
    await r.send(OperatorAbort("op"))
    await r.drain_to_end()


async def d_rx_idle_result(r: Rig) -> None:
    await r.start()
    await r.ui_event("consent_declined")  # RESULT, playlist z00 outstanding
    r.sess.timers.cancel("T_MARK_LOST")
    await r.advance(36_000)


def playlist(r: Rig) -> list[str]:
    return [k.value for k in r.sess.playlist]


SESSION_ROWS: list[tuple[str, Mode, Drive, Callable[[Rig], None]]] = [
    (
        "S13",
        Mode.KEYPAD,
        d_accept,
        lambda r: (
            r.sess.timers.armed("T_START") == r.clock.mono_ms() + 5_000
            and r.sess.timers.armed("T_RX_IDLE") == r.clock.mono_ms() + 35_000
        )
        or pytest.fail("timers"),
    ),
    (
        "S12",
        Mode.KEYPAD,
        d_bad_token,
        lambda r: (r.sess.state is S.AWAIT_START and r.t.closed == 4401) or pytest.fail("S12"),
    ),
    (
        "S15",
        Mode.VOICE,
        d_t_start,
        lambda r: (
            r.sess.rec.end_reason is EndReason.QUEUE_ABANDONED
            and r.t.closed == 4401
            and r.permit.released == 1
            and r.sess.is_ended()
        )
        or pytest.fail("S15"),
    ),
    (
        "S16",
        Mode.VOICE,
        d_closed_await,
        lambda r: (r.sess.rec.end_reason is EndReason.QUEUE_ABANDONED and r.permit.released == 1)
        or pytest.fail("S16"),
    ),
    (
        "S17",
        Mode.KEYPAD,
        d_rx_idle,
        lambda r: (
            r.sess.rec.end_reason is EndReason.SOCKET_LOST and r.decision().rule is TruthRule.T8
        )
        or pytest.fail("S17"),
    ),
    (
        "S23",
        Mode.KEYPAD,
        d_declined,
        lambda r: (
            r.decision().rule is TruthRule.T10
            and playlist(r) == ["WORDING_W_DECLINED", "GOODBYE"]
            and r.t.ui("result")[0].report_url == ""
            and r.sess.rec.report_token_sha256 is None
        )
        or pytest.fail("S23"),
    ),
    (
        "S26",
        Mode.KEYPAD,
        d_consent_timeout,
        lambda r: (
            r.decision().rule is TruthRule.T9
            and r.sess.rec.consent is not None
            and r.sess.rec.consent.method is None
            and playlist(r) == ["DISCLAIMER", "GOODBYE"]
            and r.t.ui("result")[0].page_text == ""
        )
        or pytest.fail("S26"),
    ),
    (
        "S52",
        Mode.KEYPAD,
        d_cant_hear,
        lambda r: (
            r.decision().rule is TruthRule.T4
            and r.decision().band is Band.REFER
            and playlist(r) == ["WORDING_W_REF_LOUD", "DISCLAIMER", "GOODBYE"]
            and ("clear", None) in r.t.sent
        )
        or pytest.fail("S52"),
    ),
    (
        "S53",
        Mode.KEYPAD,
        d_hangup_test,
        lambda r: (
            r.sess.rec.end_reason is EndReason.CALLER_HANGUP
            and r.decision().rule is TruthRule.T7
            and r.sess.playlist == []
            and r.playlist_marks() == []
        )
        or pytest.fail("S53"),
    ),
    (
        "S54",
        Mode.KEYPAD,
        d_lost_test,
        lambda r: (
            r.sess.rec.end_reason is EndReason.SOCKET_LOST and r.decision().band is Band.FAILED
        )
        or pytest.fail("S54"),
    ),
    (
        "S56",
        Mode.KEYPAD,
        d_session_max,
        lambda r: (
            r.sess.rec.end_reason is EndReason.SESSION_TIMEOUT
            and r.decision().rule is TruthRule.T7
            and playlist(r) == ["WORDING_W_INC", "DISCLAIMER", "GOODBYE_EARLY"]
        )
        or pytest.fail("S56"),
    ),
    (
        "S57",
        Mode.KEYPAD,
        d_abort_test,
        lambda r: (
            r.sess.rec.end_reason is EndReason.OPERATOR_ABORT
            and r.decision().rule is TruthRule.T7
            and playlist(r)[-1] == "GOODBYE_EARLY"
        )
        or pytest.fail("S57"),
    ),
    (
        "S65",
        Mode.KEYPAD,
        d_rx_idle_result,
        lambda r: (r.sess.is_ended() and r.decision().rule is TruthRule.T10) or pytest.fail("S65"),
    ),
]


@pytest.mark.parametrize(
    ("row", "mode", "drive", "check"), SESSION_ROWS, ids=[r[0] for r in SESSION_ROWS]
)
async def test_session_fsm_row(
    test_bank: BankStimulusProvider,
    row: str,
    mode: Mode,
    drive: Drive,
    check: Callable[[Rig], None],
) -> None:
    r = build(test_bank, mode=mode)
    await drive(r)
    assert row in r.sess.rules, r.sess.rules
    check(r)
    await r.close()


# ---------- session rows needing more setup ----------
async def test_S11_S20_start_then_consent(kp: Callable[..., Rig]) -> None:
    r = kp()
    await r.start()
    assert r.sess.rules[:3] == ["S13", "S11", "S20"]
    assert r.sess.state is S.CONSENT
    assert r.t.ui("consent")[0].script.value == "PUBLIC_V1"
    assert r.t.marks() == ["p0001"]
    assert r.sess.timers.armed("T_CONSENT_BROWSER") == r.clock.mono_ms() + 300_000
    # a second start (token already used) is rejected
    await r.start()
    assert "S12" not in r.sess.rules  # not in AWAIT_START any more: unexpected, ignored
    await r.close()


async def test_S18_crash_before_start(kp: Callable[..., Rig], monkeypatch: Any) -> None:
    r = kp()

    async def boom(_ev: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(r.sess, "_on_start", boom)
    await r.start()
    assert "S18" in r.sess.rules
    assert r.sess.is_ended()
    assert r.sess.rec.end_reason is EndReason.SERVER_ERROR
    assert r.decision().rule is TruthRule.T8
    assert "session.crash" in r.tm.names()


async def test_S22_keypad_and_voice(kp: Callable[..., Rig], vc: Callable[..., Rig]) -> None:
    k = kp()
    await k.to_consent_done()
    assert k.sess.state is S.INSTRUCTIONS
    assert "S22" in k.sess.rules
    assert k.sess.rec.consent is not None
    assert k.sess.rec.started_at is not None
    assert k.sess.timers.armed("T_SESSION_MAX") is not None
    assert k.t.ui("mic") == []
    v = vc()
    await v.to_consent_done()
    assert v.sess.state is S.CONSENT
    assert v.t.ui("mic")[0].start is True
    assert v.sess.stt is not None
    assert v.stt.current.cfg.key_id is KeyId.B
    assert v.stt.current.cfg.request_id == f"{v.sess.session_id}-s1"
    assert v.sess.timers.armed("T_MIC_FIRST") is not None
    await k.close()
    await v.close()


async def test_IT_ORC_03_consent_click_clears_prompt(kp: Callable[..., Rig]) -> None:
    r = kp()
    await r.start()
    await r.ui_event("consent_given")  # consent prompt still playing
    assert ("clear", None) in r.t.sent
    await r.send(_mark("p0001", r))  # the cleared mark echoes back: ignored
    assert r.sess.d_est_samples == []
    await r.close()


def _mark(name: str, r: Rig, cleared: bool = False) -> Any:
    from sevi.contracts.adapters import TMark

    return TMark(name=name, received_mono_ms=r.clock.mono_ms(), cleared=cleared)


async def test_S27_floor_timer_moves_voice_to_instructions(vc: Callable[..., Rig]) -> None:
    r = vc()
    await r.to_consent_done()
    await r.media(1)
    assert r.sess.timers.armed("T_FLOOR") is not None
    assert r.sess.state is S.CONSENT
    await r.advance(600)
    assert r.sess.state is S.INSTRUCTIONS
    assert "S27" in r.sess.rules
    await r.close()


@pytest.mark.parametrize("trigger", ["mic_denied", "mic_lost", "T_MIC_FIRST"])
async def test_S28_mic_failure_before_test_switches_to_keypad(
    vc: Callable[..., Rig], trigger: str
) -> None:
    r = vc()
    r.auto_media = False  # the page never delivers audio
    await r.to_consent_done()
    if trigger == "T_MIC_FIRST":
        await r.advance(15_001)
    else:
        await r.ui_event(trigger)
    assert "S28" in r.sess.rules
    assert r.sess.mode is Mode.KEYPAD
    assert r.sess.state is S.INSTRUCTIONS
    assert SessionFlag.MIC_LOST in r.sess.flags
    assert r.permit.released == 1
    assert r.sess.stt is None
    assert r.t.ui("mode")[-1].mode is Mode.KEYPAD
    await r.close()


async def test_S29_mic_settings(vc: Callable[..., Rig]) -> None:
    r = vc()
    await r.to_consent_done()
    await r.ui_event("mic_settings", ec=True, ns=True, agc=False, sample_rate_hz=48000)
    assert "S29" in r.sess.rules
    assert SessionFlag.MIC_PROCESSING in r.sess.flags
    assert r.sess.rec.client is not None
    assert r.sess.rec.client.mic_ns is True
    await r.close()


async def test_S40_S41_instruction_repeats(kp: Callable[..., Rig]) -> None:
    r = kp()
    await r.to_consent_done()
    assert "S40" in r.sess.rules
    for _ in range(3):
        await r.send(TDtmf(key="0", received_mono_ms=r.clock.mono_ms()))
    assert r.sess.rules.count("S41") == 2  # INSTRUCTION_REPEATS_MAX
    assert [k for k, _ in r.t.sent].count("clear") == 2
    await r.through_instructions()
    assert r.sess.state is S.TEST
    await r.close()


async def test_S42_IT_ORC_34_public_procedure_config(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    assert "S42" in r.sess.rules
    proc = r.sess.proc
    assert proc is not None
    assert proc.config.condition is Condition.NORMAL
    assert not proc.config.skip_volume_and_practice
    assert proc.config.seed_scope == r.sess.session_id
    assert r.sess.pres is not None
    assert r.sess.pres.spec.kind is PresentationKind.VOLUME
    await r.close()


async def test_IT_ORC_28_station_condition_skips_volume(kp: Callable[..., Rig]) -> None:
    r = kp(condition=Condition.LP1500)
    await to_test(r)
    proc = r.sess.proc
    assert proc is not None
    assert proc.config.skip_volume_and_practice
    await r.keypad_answer()
    assert all(t.spec.condition is Condition.LP1500 for t in r.store.trials)
    await r.close()


async def test_S50_trial_persisted_and_procedure_advanced(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    first = r.sess.pres
    assert first is not None
    await r.keypad_answer()
    assert "S50" in r.sess.rules
    (t,) = r.store.trials
    assert t.spec.presentation_id == first.spec.presentation_id
    assert t.result.status is AnswerStatus.OK
    assert t.result.triplet_correct
    assert t.d_est_ms == r.sess.d_est_ms
    assert t.t_guard_ms == 50
    assert r.sess.pres is not None
    assert r.sess.pres.spec.presentation_id == 2
    await r.close()


async def test_S51_IT_ORC_31_breaker_while_gated(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    # drive to the stimulus gate of presentation 1
    while r.sess.pres is not None and r.sess.pres.p is not P.GATED:
        await r.finish_clip()
    await r.send(BreakerOpened(KeyId.B))
    assert "S51" in r.sess.rules
    assert r.sess.mode is Mode.KEYPAD
    assert SessionFlag.BREAKER_TRIPPED in r.sess.flags
    assert SessionFlag.MIXED_MODE not in r.sess.flags
    mode_idx = next(i for i, (k, v) in enumerate(r.t.sent) if k == "ui" and v.type == "mode")
    await r.to_listen()
    kp_idx = max(
        i for i, (k, v) in enumerate(r.t.sent) if k == "ui" and v.type == "keypad" and v.enabled
    )
    assert mode_idx < kp_idx  # mode switch announced before the window opens
    assert r.sess.pres is not None
    assert r.sess.pres.window_mode is Mode.KEYPAD
    await r.close()


async def test_S55_IT_ORC_24_crash_in_test(kp: Callable[..., Rig], monkeypatch: Any) -> None:
    r = kp()
    await to_test(r)
    calls: list[Any] = []
    real = r.deps.classifier

    class Spy:
        def classify(self, inp: Any) -> Any:
            calls.append(inp)
            return real.classify(inp)

        def wording(self, *a: Any) -> Any:
            return real.wording(*a)

    r.deps.classifier = Spy()  # type: ignore[assignment]

    real_scorer = r.deps.scorer

    class Faulty:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_scorer, name)

        def score(self, *_a: Any) -> Any:
            raise RuntimeError("scorer fault")

    r.deps.scorer = Faulty()  # type: ignore[assignment]
    await r.keypad_answer()
    assert "S55" in r.sess.rules
    d = r.decision()
    assert d.rule is TruthRule.T8
    assert calls == []
    assert {s.value for s in d.surfaces} == {"SPOKEN", "PAGE", "REPORT", "OPERATOR"}
    assert playlist(r)[0] == "WORDING_W_SYSFAIL"
    await r.drain_to_end()
    assert r.sess.rec.end_reason is EndReason.SERVER_ERROR


async def test_S57_abort_in_result_goes_closing_without_reclassify(kp: Callable[..., Rig]) -> None:
    r = kp()
    await r.start()
    await r.ui_event("consent_declined")
    before = r.decision()
    await r.send(OperatorAbort("op"))
    assert r.decision() == before
    assert r.sess.rules.count("S60") == 1
    assert "S57" in r.sess.rules
    await r.drain_to_end()


async def test_S58_IT_ORC_32_mic_lost_in_test_rescored_in_keypad(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    pid = r.sess.pres.spec.presentation_id  # type: ignore[union-attr]
    r.auto_media = False
    # a 1 s arrival gap, then a frame with a new ts_ms -> nothing (stall is on ts_ms)
    await r.advance(1_000)
    await r.media(1)
    assert r.sess.mode is Mode.VOICE
    # no new ts_ms for 6 s -> S58
    await r.advance(6_100)
    assert "S58" in r.sess.rules
    assert r.sess.mode is Mode.KEYPAD
    assert SessionFlag.MIC_LOST in r.sess.flags
    t = r.store.trials[-1]
    assert t.spec.presentation_id == pid
    assert t.result.status is AnswerStatus.CAPTURE_MISS
    nxt = r.sess.pres
    assert nxt is not None
    assert nxt.spec.presentation_id > pid
    await r.keypad_answer()
    assert r.store.trials[-1].result.mode is Mode.KEYPAD
    await r.close()


async def test_S60_S61_S62_S70_S73_keypad_complete(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    while r.sess.state is S.TEST:
        await r.keypad_answer()
    await r.drain_to_end()
    for row in ("S60", "S61", "S62", "S70", "S73"):
        assert row in r.sess.rules, row
    assert r.sess.rec.end_reason is EndReason.COMPLETED
    assert r.playlist_marks() == ["z00", "z01", "z02"]
    assert r.t.ui("closed")
    assert r.t.closed == 1000
    assert r.sess.rec.evon is EvonOutcome.LIVE
    assert r.t.ui("explanation")[0].text == "A calm explanation."
    assert r.t.ui("mic") == []  # keypad never started the mic
    res = r.t.ui("result")[0]
    assert res.report_url.startswith("http://127.0.0.1:8000/sevi/r.html#")
    assert r.sess.rec.report_token_sha256 is not None


async def test_S63_IT_ORC_23_hangup_during_playlist(test_bank: BankStimulusProvider) -> None:
    from orch_harness import FakeExplainer

    ex = FakeExplainer(hang=True)
    r = build(test_bank, explainer=ex)
    await to_test(r)
    await r.ui_event("cant_hear_confirmed")
    token = r.sess.rec.report_token_sha256
    assert r.sess.explain_task is not None
    task = r.sess.explain_task
    await r.hangup(clean=True)
    await r.pump()
    assert "S63" in r.sess.rules
    assert r.sess.is_ended()
    assert task.cancelled()
    assert r.sess.rules.count("S60") == 1
    assert r.sess.rec.report_token_sha256 == token


async def test_S64_IT_ORC_25_crash_in_result(kp: Callable[..., Rig], monkeypatch: Any) -> None:
    r = kp()
    await r.start()
    await r.ui_event("consent_declined")
    before = r.decision()

    async def boom(**_k: Any) -> None:
        raise RuntimeError("playlist fault")

    monkeypatch.setattr(r.sess, "_next_playlist_item", boom)
    await r.finish_clip()
    assert "S64" in r.sess.rules
    assert r.sess.is_ended()
    assert r.decision() == before


async def test_S73_permit_released_and_record_persisted(vc: Callable[..., Rig]) -> None:
    r = vc()
    await r.start()
    await r.ui_event("consent_declined")
    await r.drain_to_end()
    assert r.permit.released == 1
    rec = r.store.sessions[r.sess.session_id]
    assert rec.state is S.ENDED
    assert rec.ended_at is not None
    assert rec.delete_after is not None
    assert (rec.delete_after - rec.created_at).total_seconds() == 72 * 3600


# =====================================================================================
# Presentation FSM (contracts §6.3)
# =====================================================================================
async def test_P01_P03_pre_prompt_fixes_d_est(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    p = r.sess.pres
    assert p is not None
    assert p.spec.pre_prompt is not None
    assert p.p is P.PRE_PROMPT
    assert "P01" in r.sess.rules
    assert not r.sess.d_est_fixed
    await r.finish_clip(extra_ms=200)  # mark rtt - duration = 200 ms
    assert "P03" in r.sess.rules
    assert r.sess.d_est_fixed
    # samples: consent (cleared? no: played out), instructions, VOLUME_INTRO -> 3 samples
    assert len(r.sess.d_est_samples) >= 2
    assert r.sess.rec.d_est_ms == r.sess.d_est_ms
    await r.close()


@pytest.mark.parametrize(
    ("samples", "expected", "defaulted"),
    [([180.0, 220.0], 200.0, False), ([180.0], 150.0, True), ([2000.0, 2000.0], 1500.0, False)],
)
async def test_IT_ORC_04_d_est(
    kp: Callable[..., Rig], samples: list[float], expected: float, defaulted: bool
) -> None:
    r = kp()
    r.sess.d_est_samples = [(0.0, s) for s in samples]
    r.sess._fix_d_est()
    assert r.sess.d_est_ms == expected
    assert (SessionFlag.D_EST_DEFAULTED in r.sess.flags) is defaulted


async def test_IT_ORC_04_voice_excludes_samples_before_first_media(vc: Callable[..., Rig]) -> None:
    r = vc()
    await r.start()
    await r.finish_clip()  # consent prompt: t_send before any media
    assert r.sess.d_est_samples == []
    await r.close()


async def test_P02_P04a_P04b_IT_ORC_05_gate(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    await r.finish_clip()  # VOLUME_INTRO -> SEND
    p = r.sess.pres
    assert p is not None
    assert p.p is P.GATED
    assert "P04a" in r.sess.rules
    assert "P04b" in r.sess.rules
    assert p.rendered is not None
    expected = p.t_send + r.sess.d_est_ms + p.rendered.timing.last_digit_offset_ms + 50
    assert r.sess.timers.armed("T_GATE") == expected == p.gate_end
    assert r.t.marks()[-1] == f"s{p.spec.presentation_id:03d}r0"
    assert not [u for u in r.t.ui("keypad") if u.enabled]
    await r.advance_to(expected - 1)
    assert p.p is P.GATED
    await r.advance_to(expected)
    assert p.p is P.LISTEN
    assert "P08" in r.sess.rules
    assert r.t.ui("keypad")[-1].enabled
    # the second volume presentation has no pre_prompt -> P02
    await r.keypad_answer()
    assert "P02" in r.sess.rules
    await r.close()


async def test_P05_P06_P07_gated_media_dtmf_transcript(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    while r.sess.pres is not None and r.sess.pres.p is not P.GATED:
        await r.finish_clip()
    p = r.sess.pres
    assert p is not None
    await r.media(1)
    assert "P05" in r.sess.rules
    assert r.stt.current.frames[-1] is None  # gated frames carry no window tag
    await r.send(TDtmf(key="4", received_mono_ms=r.clock.mono_ms()))
    assert "P06" in r.sess.rules
    assert p.dtmf == []
    await stt_push(r, tx(r, "4 7 2", bound=None))
    assert "P07" in r.sess.rules
    assert p.stale_segments == 1
    await r.close()


async def test_P09_P10_P11_P10b_voice_window(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    other = p.spec.presentation_id + 1
    from sevi.contracts.adapters import FrameTag

    await stt_push(r, SttProcessing())
    assert "P09" in r.sess.rules
    assert r.sess.outstanding == 1
    await stt_push(r, tx(r, ""))
    assert "P11" in r.sess.rules
    assert r.sess.outstanding == 0
    await stt_push(r, tx(r, "4 7 2", bound=FrameTag(presentation_id=other, window_index=1)))
    assert "P10b" in r.sess.rules
    assert p.stale_segments == 1
    assert p.p is P.LISTEN
    d1, d2, d3 = p.spec.triplet
    await stt_push(r, tx(r, f"{d1} {d2} {d3}", index=2))
    assert "P10" in r.sess.rules
    t = r.store.trials[-1]
    assert t.result.status is AnswerStatus.OK
    assert t.result.triplet_correct
    assert t.stale_segments == 1
    assert t.result.mode is Mode.VOICE
    assert r.stt.current.commits == 0  # IT-ORC-14: clean THREE_DIGITS -> no commit
    await r.close()


async def test_P12_IT_ORC_07_double_answer(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    for k in "4713":
        await r.send(TDtmf(key=k, received_mono_ms=r.clock.mono_ms()))  # type: ignore[arg-type]
    assert r.sess.rules.count("P12") == 3
    t = r.store.trials[-1]
    assert t.spec.presentation_id == p.spec.presentation_id  # type: ignore[union-attr]
    assert t.result.recognised == (4, 7, 1)
    await r.close()


async def test_P13_P15_IT_ORC_09_replay_once(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    await r.send(TDtmf(key="4", received_mono_ms=r.clock.mono_ms()))
    await r.send(TDtmf(key="*", received_mono_ms=r.clock.mono_ms()))
    assert "P13" in r.sess.rules
    assert p.replay_used
    assert p.dtmf == []
    await r.to_listen()
    assert f"s{p.spec.presentation_id:03d}r1" in r.t.marks()
    await r.send(TDtmf(key="*", received_mono_ms=r.clock.mono_ms()))
    assert "P15" in r.sess.rules
    assert p.p is P.LISTEN
    await r.keypad_answer()
    assert r.store.trials[-1].replays == 1
    await r.close()


async def test_P14_IT_ORC_10_repeat_instructions_in_practice(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    while r.sess.pres is not None and r.sess.pres.spec.kind is not PresentationKind.PRACTICE:
        await r.keypad_answer()
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    await r.send(TDtmf(key="0", received_mono_ms=r.clock.mono_ms()))
    assert "P14" in r.sess.rules
    clip = r.sess.player.deferred or r.sess.player.outstanding
    assert clip is not None
    assert clip.key is not None
    assert clip.key.value == "INSTRUCTIONS"
    await r.send(TDtmf(key="0", received_mono_ms=r.clock.mono_ms()))  # not LISTEN: ignored
    await r.to_listen()
    assert r.sess.pres is p
    assert not p.replay_used
    await r.close()


async def test_P16_P17_settle_and_timeout(vc: Callable[..., Rig], kp: Callable[..., Rig]) -> None:
    v = vc()
    await to_test(v)
    await v.to_listen()
    await stt_push(v, tx(v, "4 7"))
    settle = v.sess.settings.settle_window_ms  # max(900, 800 + 300) = 1100
    assert settle == 1100
    assert v.sess.timers.armed("T_SETTLE") == v.clock.mono_ms() + settle
    await v.advance(settle)
    assert "P16" in v.sess.rules
    assert v.store.trials[-1].result.status is AnswerStatus.PARTIAL
    await v.close()
    k = kp()
    await to_test(k)
    await k.to_listen()
    await k.advance(8_000)
    assert "P17" in k.sess.rules
    assert "P19" in k.sess.rules
    assert k.store.trials[-1].result.status is AnswerStatus.NO_RESPONSE
    await k.close()


async def test_P16_settle_outlasts_vad_silence_two_part_answer(vc: Callable[..., Rig]) -> None:
    """#92 follow-up to #94: "d1 d2 ... d3" with a 600 ms pause. The second part's final comes
    ~1,000 ms after the first (600 ms pause + 400 ms speech; each final follows its own
    min_silence endpointing), past t_settle_ms (900) but inside the effective settle window
    max(t_settle_ms, min_silence_ms + 300) = 1,100. Both parts score as one triplet, with no
    commit and no T_COMMIT_MAX."""
    v = vc()
    await to_test(v)
    await v.to_listen()
    p = v.sess.pres
    assert p is not None
    d1, d2, d3 = p.spec.triplet
    await stt_push(v, tx(v, f"{d1} {d2}", index=1))
    assert v.sess.settings.capture.t_settle_ms < 1000 < v.sess.settings.settle_window_ms
    await v.advance(1000)
    assert "P16" not in v.sess.rules
    assert p.p is P.LISTEN
    await stt_push(v, tx(v, f"{d3}", index=2))
    t = v.store.trials[-1]
    assert t.result.status is AnswerStatus.OK
    assert t.result.triplet_correct
    assert t.result.mode is Mode.VOICE
    assert v.stt.current.commits == 0
    assert "P16" not in v.sess.rules
    assert "P22" not in v.sess.rules
    assert v.sess.timers.armed("T_COMMIT_MAX") is None
    await v.close()


async def test_IT_ORC_15_intent_closes_at_once(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    await stt_push(r, tx(r, "theriyala"))
    t = r.store.trials[-1]
    assert t.result.status is AnswerStatus.DONT_KNOW
    await r.close()


async def test_P18_reconnect_then_capture_miss_and_held_send(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    first = r.stt.current
    r.stt.hold = True  # the reopen does not complete yet
    await stt_push(r, SttClosed(code=1006, reason="abnormal"))
    assert "P18" in r.sess.rules
    assert r.sess.stt_reopening
    t = r.store.trials[-1]
    assert t.result.status is AnswerStatus.CAPTURE_MISS
    assert first.closed
    nxt = r.sess.pres
    assert nxt is not None
    # SEND is held while the reconnect is in flight (CCR-v3-40)
    for _ in range(3):
        if r.sess.player.outstanding is not None:
            await r.finish_clip()
    assert r.sess.held_send is nxt
    r.stt.hold = False
    r.stt._hold_ev.set()
    await r.pump()
    assert not r.sess.stt_reopening
    assert r.sess.held_send is None
    assert r.sess.mode is Mode.VOICE
    assert len(r.stt.sessions) == 2
    await r.close()


async def test_P18_reconnect_failure_falls_back_to_keypad(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    r.stt.fail = [SttError(ErrorCode.STT_CONNECT_TIMEOUT, "x")]
    await stt_push(r, SttClosed(code=1006, reason="abnormal"))
    assert "S51" in r.sess.rules
    assert r.sess.mode is Mode.KEYPAD
    assert SessionFlag.BREAKER_TRIPPED in r.sess.flags
    assert SessionFlag.BUDGET_STOP not in r.sess.flags
    assert "ledger.budget_stop" not in r.tm.names()
    assert r.permit.released == 1
    await r.close()


async def test_P18b_stt_error_message(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    await stt_push(r, SttErrorMsg(message="utterance error"))
    assert "P18b" in r.sess.rules
    assert SystemEvent.STT_ERROR in p.system_events
    await r.close()


async def test_P20_P20b_P21_IT_ORC_16_timeout_with_speech_commits(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    now = r.clock.mono_ms()
    r.capture.speech.append((now + 7_000, now + 7_900))
    await r.advance(8_000)  # T_ANSWER with speech present
    p = r.sess.pres
    assert p is not None
    assert p.p is P.COMMIT
    assert "P20" in r.sess.rules
    assert r.stt.current.commits == 1
    await r.pump()  # SttCommitSent from the fake
    assert "P20b" in r.sess.rules
    assert p.t_commit is not None
    await stt_push(r, SttProcessing())
    assert "P21b" in r.sess.rules
    assert r.sess.outstanding == 1
    d1, d2, d3 = p.spec.triplet
    await stt_push(r, tx(r, f"{d1} {d2} {d3}"))
    assert "P21" in r.sess.rules
    assert r.sess.outstanding == 0
    t = r.store.trials[-1]
    assert t.result.status is AnswerStatus.OK
    await r.close()


async def test_P22_commit_max(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    now = r.clock.mono_ms()
    r.capture.speech.append((now + 7_000, now + 7_900))
    await r.advance(8_000)
    await r.advance(1_200)
    assert "P22" in r.sess.rules
    assert r.store.trials[-1].result.status is AnswerStatus.CAPTURE_MISS
    await r.close()


async def test_P31_P35_P36_IT_ORC_18_reask(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    await stt_push(r, tx(r, "hmm"))
    await r.advance(r.sess.settings.settle_window_ms)  # SETTLE -> NO_DIGITS -> REASK
    assert "P31" in r.sess.rules
    assert "P35" in r.sess.rules
    assert p.p is P.REASK_PROMPT
    assert p.reask_used
    while p.p is P.REASK_PROMPT:
        await r.finish_clip()  # the stimulus mark first, then the re-ask prompt
    assert "P36" in r.sess.rules
    assert p.window_index == 2
    assert p.p is P.LISTEN
    assert r.sess.current_tag is not None
    assert r.sess.current_tag.window_index == 2
    d1, d2, d3 = p.spec.triplet
    await stt_push(r, tx(r, f"{d1} {d2} {d3}"))
    t = r.store.trials[-1]
    assert t.reasks == 1
    assert t.result.status is AnswerStatus.OK
    await r.close()


async def test_P32_IT_ORC_19_repeat_request_then_reask(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    await stt_push(r, tx(r, "marupadiyum"))
    await r.advance(r.sess.settings.settle_window_ms)
    assert "P32" in r.sess.rules
    assert p.replay_used
    await r.to_listen()
    assert f"s{p.spec.presentation_id:03d}r1" in r.t.marks()
    await stt_push(r, tx(r, "marupadiyum"))
    await r.advance(r.sess.settings.settle_window_ms)
    assert "P31" in r.sess.rules  # REPEAT after replay -> NO_DIGITS -> REASK
    await r.close()


async def test_P33_P40_P41_practice_feedback(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    while r.sess.pres is not None and r.sess.pres.spec.kind is not PresentationKind.PRACTICE:
        await r.keypad_answer()
    p = r.sess.pres
    assert p is not None
    wrong = tuple(d for d in (1, 2, 3, 4, 5, 6, 7, 8) if d not in p.spec.triplet)[:3]
    await r.keypad_answer(wrong)
    assert "P33" in r.sess.rules
    assert "P40" in r.sess.rules
    assert r.t.ui("feedback")[-1].correct is False
    if r.sess.player.deferred is not None:
        await r.finish_clip()  # the stimulus mark returns; the feedback clip goes out
    assert r.t.marks()[-1] == f"f{p.spec.presentation_id:03d}"
    # PRACTICE_WRONG (400 ms) + 3 x (200 ms gap + 400 ms digit)
    clip = r.sess.player.outstanding
    assert clip is not None
    assert clip.duration_ms == 400 + 3 * 600
    await r.finish_clip()
    assert "P41" in r.sess.rules
    await r.close()


async def test_IT_ORC_08_dtmf_wins_over_speech(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    p = r.sess.pres
    assert p is not None
    await stt_push(r, tx(r, "4 7"))
    for k in "471":
        await r.send(TDtmf(key=k, received_mono_ms=r.clock.mono_ms()))  # type: ignore[arg-type]
    t = r.store.trials[-1]
    assert t.result.source.value == "DTMF"
    assert t.result.recognised == (4, 7, 1)
    await r.close()


async def test_IT_ORC_11_duplicate_and_unknown_marks(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    await r.finish_clip()  # pre-prompt
    p = r.sess.pres
    assert p is not None
    name = f"s{p.spec.presentation_id:03d}r0"
    await r.send(_mark("s004r0", r))
    await r.send(_mark(name, r))
    await r.send(_mark(name, r))
    unexpected = [f for e, _, f in r.tm.events if e == "mark.received" and f.get("unexpected")]
    assert len(unexpected) == 2
    await r.close()


async def test_IT_ORC_12_mark_lost(kp: Callable[..., Rig]) -> None:
    r = kp()
    await r.start()
    c = r.sess.player.outstanding
    assert c is not None
    assert c.t_send is not None
    assert r.sess.timers.armed("T_MARK_LOST") == c.t_send + c.duration_ms + 3_000
    await r.advance(c.duration_ms + 3_000)
    assert SessionFlag.MARK_LOST in r.sess.flags
    assert "mark.lost" in r.tm.names()
    assert r.sess.player.outstanding is None
    await r.close()


async def test_IT_ORC_13_deferred_stimulus_gate_from_player_sent(kp: Callable[..., Rig]) -> None:
    r = kp()
    await to_test(r)
    await r.keypad_answer()  # answered before the stimulus mark: next clip is deferred
    p = r.sess.pres
    assert p is not None
    assert p.p is P.SEND
    assert r.sess.player.deferred is not None
    requested = r.sess.player.deferred.requested_mono
    await r.advance(300)
    await r.finish_clip()  # previous stimulus mark returns -> deferred stimulus sent now
    assert p.p is P.GATED
    assert p.t_send > requested
    assert p.rendered is not None
    assert r.sess.timers.armed("T_GATE") == (
        p.t_send + r.sess.d_est_ms + p.rendered.timing.last_digit_offset_ms + r.sess.t_guard_ms
    )
    assert any(e == "span:turn.total" for e, _, _ in r.tm.events)
    await r.close()


async def test_IT_ORC_29_stt_open_pending_at_instructions_mark(vc: Callable[..., Rig]) -> None:
    r = vc()
    r.stt.hold = True
    await r.to_consent_done()
    await r.media(35)
    await r.through_instructions()
    assert r.sess.mode is Mode.KEYPAD
    assert r.permit.released == 1
    assert SessionFlag.MIXED_MODE not in r.sess.flags
    assert any(
        f.get("reason") == "start_timeout" for e, _, f in r.tm.events if e == "mode.switched"
    )
    await r.close()


async def test_start_failure_budget_exceeded_goes_keypad(vc: Callable[..., Rig]) -> None:
    """§6.2.1 step 2: the ledger refuses the open (BudgetExceeded) -> keypad for the session."""
    from sevi.contracts.errors import BudgetExceeded

    r = vc()
    r.stt.fail = [BudgetExceeded("budget")]
    await r.to_consent_done()
    assert r.sess.mode is Mode.KEYPAD
    assert r.sess.state is S.INSTRUCTIONS
    assert SessionFlag.MIC_LOST not in r.sess.flags
    assert SessionFlag.BUDGET_STOP in r.sess.flags  # #52
    assert SessionFlag.BREAKER_TRIPPED not in r.sess.flags
    assert "ledger.budget_stop" in r.tm.names()
    assert r.permit.released == 1
    await r.close()


async def test_P18_reconnect_refused_by_budget_flags_budget_stop(vc: Callable[..., Rig]) -> None:
    """#52: the ledger refuses the reconnect -> keypad with BUDGET_STOP, not BREAKER_TRIPPED."""
    from sevi.contracts.errors import BudgetExceeded

    r = vc()
    await to_test(r)
    await r.to_listen()
    r.stt.fail = [BudgetExceeded("budget")]
    await stt_push(r, SttClosed(code=1006, reason="abnormal"))
    assert "S51" in r.sess.rules
    assert r.sess.mode is Mode.KEYPAD
    assert SessionFlag.BUDGET_STOP in r.sess.flags
    assert SessionFlag.BREAKER_TRIPPED not in r.sess.flags
    assert r.tm.names().count("ledger.budget_stop") == 1
    await r.close()


async def test_T6_window_close_reason_breaker_only_from_capture_failures(
    vc: Callable[..., Rig],
) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    await r.send(BreakerOpened(KeyId.B))
    t = r.store.trials[-1]
    assert t.result.status is AnswerStatus.CAPTURE_MISS
    assert r.sess.rules.index("S51") < len(r.sess.rules)
    assert WindowCloseReason.BREAKER.value == "BREAKER"
    await r.close()


# ---------- review of #49 ----------
@pytest.mark.parametrize(
    ("browser", "earphones", "guard"),
    [("Safari", False, 250), ("Chrome", False, 50), ("Safari", True, 50)],
)
async def test_IT_ORC_43_safari_speaker_guard(
    test_bank: BankStimulusProvider, browser: str, earphones: bool, guard: int
) -> None:
    """B spec §7.6 (R-11): Safari + speaker adds 200 ms to T_GUARD_BROWSER_MS (50)."""
    r = build(test_bank, browser=browser, earphones=earphones)
    assert r.sess.rec.t_guard_ms == guard
    await to_test(r)
    await r.keypad_answer()
    await r.keypad_answer()
    assert r.store.trials
    assert all(t.t_guard_ms == guard for t in r.store.trials)
    p = r.sess.pres
    await r.finish_clip()
    if p is not None and p.p is P.GATED and p.rendered is not None:
        assert (
            p.gate_end
            == p.t_send + r.sess.d_est_ms + p.rendered.timing.last_digit_offset_ms + guard
        )
    await r.close()


@pytest.mark.parametrize("template", ["Template explanation.", None])
async def test_T_EXPLAIN_deadline_template_fallback(
    test_bank: BankStimulusProvider, template: str | None
) -> None:
    from orch_harness import FakeExplainer

    r = build(test_bank, explainer=FakeExplainer(hang=True))
    real = r.deps.copy_text

    def copy_text(key: str, lang: Any) -> str:
        if key == "evon.template.explain":
            if template is None:
                raise KeyError(key)
            return template
        return real(key, lang)

    r.deps.copy_text = copy_text
    await to_test(r)
    await r.ui_event("cant_hear_confirmed")  # T4 REFER: explainable band
    assert r.sess.explain_task is not None
    await r.drain_to_end()  # playlist done -> CLOSING waits for T_EXPLAIN (10 s)
    assert r.sess.rec.evon is EvonOutcome.TEMPLATE
    assert any(f.get("cause") == "deadline" for e, _, f in r.tm.events if e == "evon.template")
    texts = [u.text for u in r.t.ui("explanation")]
    assert texts == ([template] if template else [])
    assert r.t.ui("closed")


async def test_S57_abort_in_result_sends_clear(kp: Callable[..., Rig]) -> None:
    r = kp()
    await r.start()
    await r.ui_event("consent_declined")
    n = [k for k, _ in r.t.sent].count("clear")
    await r.send(OperatorAbort("op"))
    assert [k for k, _ in r.t.sent].count("clear") == n + 1
    await r.drain_to_end()


async def test_S58_scores_window_before_mode_message(vc: Callable[..., Rig]) -> None:
    r = vc()
    await to_test(r)
    await r.to_listen()
    n_trials = len(r.store.trials)
    seen: list[int] = []
    real_ui = r.sess.ui

    async def spy(msg: Any) -> None:
        if msg.type == "mode":
            seen.append(len(r.store.trials))
        await real_ui(msg)

    r.sess.ui = spy  # type: ignore[method-assign]
    await r.ui_event("mic_lost")
    assert seen == [n_trials + 1]  # CAPTURE_MISS trial persisted before ui mode {keypad}
    await r.close()


async def test_capture_playing_state_by_capture_time(vc: Callable[..., Rig]) -> None:
    r = vc()
    await r.to_consent_done()
    await r.media(3)
    r.sess.clip_intervals.append((100.0, 200.0))
    assert r.sess.playing_at(150.0)
    assert not r.sess.playing_at(250.0)
    await r.close()


async def test_U_ORC_34_queue_put_waits_then_overflows(kp: Callable[..., Rig]) -> None:
    import asyncio

    from sevi.app.orchestrator import BreakerOpened as BO

    r = kp()
    while not r.sess.queue.full():
        r.sess.queue.put_nowait(BO(KeyId.B))
    waiter = asyncio.create_task(r.sess.put(BO(KeyId.B)))
    await asyncio.sleep(0.05)
    assert not r.sess.overflowed  # still waiting for room
    r.sess.queue.get_nowait()
    await waiter
    assert not r.sess.overflowed
    assert r.sess.queue.full()
    await asyncio.wait_for(r.sess.put(BO(KeyId.B)), 2.0)
    assert r.sess.overflowed  # no room within 1,000 ms
