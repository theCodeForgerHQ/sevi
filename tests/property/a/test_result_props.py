# ruff: noqa: PT018, RUF001, E501
"""Property tests for core/result.py and the wording table (Person A spec §8.6: P-RES-01..06)."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from sevi.contracts.enums import (
    Band,
    Channel,
    EndReason,
    FriendsVariant,
    Lang,
    Mode,
    ProcedureTerminal,
    Surface,
    Transducer,
    TruthRule,
    WordingClass,
)
from sevi.contracts.procedure import SrtResult
from sevi.contracts.result import NormCellKey, ResultDecision, ResultInput
from sevi.core.result import classify, find_cell
from sevi.core.staircase import PROCEDURE
from sevi.core.wording import load_copy
from sevi.fakes.scenarios.result_inputs import BANK, OTHER_BANK, cell, inp, norms
from sevi.fakes.scenarios.staircase_driver import drive, make_config

TABLE = load_copy()
SYMS = st.sampled_from(["C", "W", "NR", "CM"])


@st.composite
def srts(draw: st.DrawFn) -> SrtResult | None:
    if draw(st.integers(0, 9)) == 0:
        return None
    skip = draw(st.booleans())
    answers = draw(st.lists(SYMS, max_size=80))
    state, _ = drive(make_config("prop-res", skip_volume_and_practice=skip), answers)
    return PROCEDURE.summarize(state)


@st.composite
def norms_files(draw: st.DrawFn) -> Any:
    kind = draw(st.sampled_from(["none", "empty", "cell"]))
    if kind == "none":
        return None
    if kind == "empty":
        return norms()
    n = draw(st.sampled_from([3, 6, 7, 10]))
    nh95 = draw(st.floats(-10.0, -4.0))
    mode = draw(st.sampled_from([Mode.KEYPAD, Mode.VOICE]))
    bank = draw(st.sampled_from([BANK, OTHER_BANK]))
    return norms(cell(n=n, nh95=nh95, mode=mode, bank_hash=bank))


@st.composite
def result_inputs(draw: st.DrawFn, s: Any = None) -> ResultInput:
    channel = draw(st.sampled_from(list(Channel)))
    lang = draw(st.sampled_from(list(Lang)))
    return inp(
        srt=draw(srts()) if s is None else s,
        end_reason=draw(st.sampled_from(list(EndReason))),
        channel=channel,
        lang=lang,
        mode=draw(st.sampled_from(list(Mode))),
        mode_requested=draw(st.sampled_from([Mode.KEYPAD, Mode.VOICE])),
        transducer=draw(st.sampled_from(list(Transducer))),
        friends=draw(st.sampled_from(list(FriendsVariant))),
        experimental_language=draw(st.booleans()),
        norms=draw(norms_files()),
        hhie_total=draw(st.one_of(st.none(), st.integers(0, 40))),
    )


def oracle(i: ResultInput) -> TruthRule:
    """Independent restatement of contracts §4.7.1 (rule only)."""
    s, e = i.srt, i.end_reason
    n24 = bool(s and s.n_scored == 24)
    t = s.terminal if s else None
    rules: list[tuple[bool, TruthRule]] = [
        (t == ProcedureTerminal.VOLUME_FAILED, TruthRule.T1),
        (t == ProcedureTerminal.PRACTICE_FAILED, TruthRule.T2),
        (bool(s and (t == ProcedureTerminal.LOUD_FAILED or s.loud_wrong_total >= 2)), TruthRule.T3),
        (e == EndReason.CANT_HEAR, TruthRule.T4),
        (bool(s and s.ceiling_fail), TruthRule.T5),
        (bool(s and s.flags > 3), TruthRule.T6),
        (
            (not n24 and e.value in {"CALLER_HANGUP", "OPERATOR_ABORT", "SESSION_TIMEOUT"})
            or t == ProcedureTerminal.NO_RESPONSE_END,
            TruthRule.T7,
        ),
        (not n24 and e.value in {"SERVER_ERROR", "SOCKET_LOST"}, TruthRule.T8),
        (e.value in {"CONSENT_NO_REPLY", "DIAL_FAILED", "QUEUE_ABANDONED"}, TruthRule.T9),
        (e == EndReason.CONSENT_DECLINED, TruthRule.T10),
    ]
    for cond, rule in rules:
        if cond:
            return rule
    if not (s and n24 and s.srt_db is not None):
        return TruthRule.T8
    x = s.srt_db
    if x >= 0.0:
        return TruthRule.T10B
    c = None
    if not (
        i.experimental_language
        or i.friends == FriendsVariant.DISALLOWED
        or i.mode == Mode.MIXED
        or i.transducer == Transducer.SPEAKER
        or i.channel == Channel.PHONE
    ):
        c = find_cell(
            i.norms,
            NormCellKey(lang=i.lang, channel=i.channel, mode=i.mode, transducer=i.transducer),  # type: ignore[arg-type]
        )
    if c is None or c.n < 6 or c.bank_hash != i.bank_hash or c.nh95_db is None:
        return TruthRule.T11
    h = c.nh95_db
    if x > h + 3:
        return TruthRule.T12
    if (
        h < x <= h + 3
        or (c.cutoff_version is not None and c.cutoff_version.value == "v0" and abs(x - h) <= 1)
        or (s.reversals is not None and s.reversals < 4)
        or s.censor.value == "HIGH"
    ):
        return TruthRule.T13
    if i.hhie_total is not None and i.hhie_total >= 10:
        return TruthRule.T14
    return TruthRule.T15


@settings(deadline=None, max_examples=150)
@given(result_inputs())
def test_P_RES_05_oracle_never_raises_validates(i: ResultInput) -> None:
    d = classify(i)
    assert d.rule is oracle(i)
    assert ResultDecision.model_validate(d.model_dump()) == d


@settings(deadline=None, max_examples=100)
@given(result_inputs())
def test_P_RES_01_ceiling_fail_refers(i: ResultInput) -> None:
    if i.srt is not None and i.srt.ceiling_fail:
        assert classify(i).band is Band.REFER


@settings(deadline=None, max_examples=60)
@given(st.booleans(), st.integers(0, 60))
def test_P_RES_02_silence_never_inconclusive(skip: bool, n: int) -> None:
    state, _ = drive(make_config("prop-silence", skip_volume_and_practice=skip), ["NR"] * n)
    s = PROCEDURE.summarize(state)
    for end in EndReason:
        assert classify(inp(srt=s, end_reason=end)).band is not Band.INCONCLUSIVE


@settings(deadline=None, max_examples=150)
@given(result_inputs())
def test_P_RES_03_refer_conditions(i: ResultInput) -> None:
    s = i.srt
    d = classify(i)
    if s is not None and (
        s.volume_failed
        or s.practice_failed
        or s.loud_wrong_total >= 2
        or s.ceiling_fail
        or s.terminal in (ProcedureTerminal.VOLUME_FAILED, ProcedureTerminal.PRACTICE_FAILED)
    ):
        assert d.band is Band.REFER
    if i.end_reason is EndReason.CANT_HEAR:
        assert d.band is Band.REFER


@settings(deadline=None, max_examples=30)
@given(st.booleans())
def test_P_RES_03_silent_at_plus10_never_reassures(skip: bool) -> None:
    # every answer wrong: the track climbs to +10 and stays there
    state, _ = drive(make_config("prop-loud", skip_volume_and_practice=skip), ["NR"] * 80)
    d = classify(inp(srt=PROCEDURE.summarize(state)))
    assert d.band not in (Band.PASS, Band.BORDERLINE, Band.NO_BAND, Band.INCOMPLETE)


REFERRAL = {
    "en": ("audiologist", "hearing check"),
    "ta": ("செவித்திறன் பரிசோதனை", "செவித்திறன் நிபுணர"),
}
DISCLAIMER = {"en": "not a diagnosis", "ta": "நோயறிதல் அல்ல"}
LIMITS = {
    WordingClass.W_PASS,
    WordingClass.W_PASS_NOTE,
    WordingClass.W_BORDER,
    WordingClass.W_NOBAND_LT6,
    WordingClass.W_NOBAND_RANGE,
    WordingClass.W_NOBAND_UNCOMPARABLE,
}


def _in_order(text: str, phrases: tuple[str, ...]) -> bool:
    pos = 0
    for p in phrases:
        k = text.find(p, pos)
        if k < 0:
            return False
        pos = k + len(p)
    return True


@settings(deadline=None, max_examples=150)
@given(result_inputs())
def test_P_RES_04_wording_referral_and_limits(i: ResultInput) -> None:
    d = classify(i)
    for surf in d.surfaces:
        for lang in Lang:
            w = TABLE.wording(d, lang, surf)
            entry = TABLE.entry(w.copy_key)
            if d.wording_class is not WordingClass.W_DECLINED:
                assert (
                    any(k in w.text for k in REFERRAL[lang.value])
                    or DISCLAIMER[lang.value] in w.text
                )
            if d.wording_class.value.startswith("W_REF_"):
                assert all(m in w.text for m in entry.must_contain[lang.value])
            if d.wording_class in LIMITS and surf is not Surface.OPERATOR:
                assert _in_order(w.text, entry.must_contain[lang.value])


@settings(deadline=None, max_examples=150)
@given(
    result_inputs(),
    st.floats(0.0, 10.0),
)
def test_P_RES_06_abs_refer_every_context(i: ResultInput, x: float) -> None:
    from sevi.fakes.scenarios.result_inputs import srt

    j = i.model_copy(
        update={"srt": srt(srt_db=x), "end_reason": EndReason.COMPLETED, "hhie_total": None}
    )
    d = classify(j)
    assert (d.band, d.rule, d.wording_class) == (Band.REFER, TruthRule.T10B, WordingClass.W_REF_ABS)
    k = i.model_copy(update={"srt": srt(srt_db=-abs(x) - 0.01), "end_reason": EndReason.COMPLETED})
    assert classify(k).rule is not TruthRule.T10B
