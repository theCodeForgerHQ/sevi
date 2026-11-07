"""Property tests for core/scoring.py (spec §7.7 P-SC-01..07)."""

from __future__ import annotations

from typing import Any, Literal

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sevi.contracts.enums import (
    AnswerSource,
    AnswerStatus,
    Lang,
    Mode,
    PresentationKind,
    PromptKey,
    ScoreAction,
    SystemEvent,
    WindowCloseReason,
)
from sevi.contracts.trial import (
    AnswerEvidence,
    DtmfPress,
    PresentationSpec,
    ScoringResult,
    SttSegment,
)
from sevi.core.scoring import align, load_lexicons

LEX = load_lexicons()
SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])

DIGIT_FORMS = {d: LEX.forms_all_langs(d) for d in range(10)}
INTENTS = ["theriyala", "don't know", "marupadiyum", "one more time", "come again"]
GROUPINGS = [
    "four seventy-two",
    "forty-seven two",
    "four seventy",
    "four hundred and ninety two",
    "நாலு எழுபத்தி ரெண்டு",
    "seventy two",
    "double five two",
    "double",
    "triple four",
]
EN_TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}
TA_TENS = {
    2: ["இருபத்தி", "irubathu"],
    3: ["முப்பத்து", "muppathu"],
    4: ["நாற்பத்து", "naapathu"],
    5: ["ஐம்பத்தி", "aimbathu"],
    6: ["அறுபத்து", "arubathu"],
    7: ["எழுபத்தி", "ezhubathu"],
    8: ["எண்பத்து", "enbathu"],
    9: ["தொண்ணூற்று", "thonnooru"],
}
EN_TEENS = [
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TA_TEENS = [
    "பதினொன்று",
    "பன்னிரண்டு",
    "பதிமூன்று",
    "பதினான்கு",
    "பதினைந்து",
    "பதினாறு",
    "பதினேழு",
    "பதினெட்டு",
    "பத்தொன்பது",
]

triplets = st.permutations(list(range(1, 10))).map(lambda p: (p[0], p[1], p[2]))
fillers = st.sampled_from(LEX.fillers)


def form(d: int) -> st.SearchStrategy[str]:
    return st.sampled_from(DIGIT_FORMS[d])


def spec(triplet: tuple[int, int, int]) -> PresentationSpec:
    return PresentationSpec(
        presentation_id=1,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=0.0,
        triplet=triplet,
        pool_index=0,
        noise_seed=0,
    )


segment_text = st.one_of(
    st.lists(st.integers(0, 9).flatmap(form), max_size=5).map(" ".join),
    fillers,
    st.sampled_from(INTENTS),
    st.sampled_from(GROUPINGS),
    st.text(alphabet="abcdefg ", max_size=12),
    st.just(""),
    st.lists(st.integers(0, 9), min_size=1, max_size=14).map(lambda ds: "".join(map(str, ds))),
)

KEYS: list[Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]] = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


@st.composite
def evidences(draw: st.DrawFn) -> AnswerEvidence:
    texts = draw(st.lists(segment_text, max_size=4))
    dtmf = draw(st.lists(st.sampled_from(KEYS), max_size=4))
    return AnswerEvidence(
        presentation_id=1,
        lang=draw(st.sampled_from(list(Lang))),
        mode=draw(st.sampled_from([Mode.VOICE, Mode.KEYPAD])),
        window_index=draw(st.sampled_from([1, 2])),
        window_open_mono_ms=0.0,
        window_close_mono_ms=3000.0,
        close_reason=WindowCloseReason.SETTLE,
        segments=tuple(
            SttSegment(text=t, received_ms=float(i), after_commit=False)
            for i, t in enumerate(texts)
        ),
        dtmf=tuple(DtmfPress(key=k, at_ms=float(i)) for i, k in enumerate(dtmf)),
        speech_present=draw(st.booleans()),
        speech_ms=0,
        early_speech=draw(st.booleans()),
        echo_mode=draw(st.booleans()),
        system_events=tuple(draw(st.sets(st.sampled_from(list(SystemEvent))))),
        reask_used=draw(st.booleans()),
        replay_used=draw(st.booleans()),
    )


@SETTINGS
@given(triplets, st.data())
def test_P_SC_01_any_forms_parse(t: tuple[int, int, int], data: st.DataObject) -> None:
    text = " ".join(data.draw(form(d)) for d in t)
    res = LEX.score(spec(t), _voice(text))
    assert res.recognised == t
    assert res.status is AnswerStatus.OK


def _voice(*texts: str) -> AnswerEvidence:
    return AnswerEvidence(
        presentation_id=1,
        lang=Lang.TA,
        mode=Mode.VOICE,
        window_index=1,
        window_open_mono_ms=0.0,
        window_close_mono_ms=1.0,
        close_reason=WindowCloseReason.SETTLE,
        segments=tuple(SttSegment(text=t, received_ms=0.0, after_commit=False) for t in texts),
        speech_present=True,
        speech_ms=500,
        early_speech=False,
        echo_mode=False,
        reask_used=False,
        replay_used=False,
    )


@SETTINGS
@given(triplets, st.data())
def test_P_SC_02_fillers_do_not_change_parse(t: tuple[int, int, int], data: st.DataObject) -> None:
    words = [data.draw(form(d)) for d in t]
    plain = LEX.parse([" ".join(words)])
    mixed: list[str] = []
    for w in words:
        mixed.extend(data.draw(st.lists(fillers, max_size=2)))
        mixed.append(w)
    mixed.extend(data.draw(st.lists(fillers, max_size=2)))
    p = LEX.parse([" ".join(mixed)])
    assert (p.digits, p.intent, p.number_form, p.unparsed) == (
        plain.digits,
        plain.intent,
        plain.number_form,
        plain.unparsed,
    )


@SETTINGS
@given(evidences(), triplets)
def test_P_SC_03_status_invariants(ev: AnswerEvidence, t: tuple[int, int, int]) -> None:
    res = LEX.score(spec(t), ev)
    assert len(res.recognised) <= 12
    assert res.position_correct == align(res.recognised, t)
    if res.source is AnswerSource.SPEECH:
        p = LEX.parse([s.text for s in ev.segments])
        n = len(p.digits)
        if n > 3:
            assert res.status is AnswerStatus.AMBIGUOUS
        elif 1 <= n <= 2:
            want = AnswerStatus.NO_DIGITS if p.unparsed else AnswerStatus.PARTIAL
            assert res.status is want


def oracle(ev: AnswerEvidence, t: tuple[int, int, int]) -> tuple[Any, ...]:
    """Independent literal reading of contracts §4.5.1 (status, action, prompt, source)."""
    accept = (ScoreAction.ACCEPT, None)
    if ev.dtmf:
        keys = [int(k.key) for k in ev.dtmf][:3]
        return (
            AnswerStatus.OK if len(keys) == 3 else AnswerStatus.PARTIAL,
            *accept,
            AnswerSource.DTMF,
        )
    if ev.mode == Mode.KEYPAD:
        return (AnswerStatus.NO_RESPONSE, *accept, AnswerSource.NONE)
    ev_set = set(ev.system_events)
    if ev_set & {SystemEvent.BREAKER_TRIP, SystemEvent.RECONNECT, SystemEvent.MIC_LOST}:
        return (AnswerStatus.CAPTURE_MISS, *accept, AnswerSource.NONE)
    empty = [s.text.strip() == "" for s in ev.segments]
    if SystemEvent.STT_ERROR in ev_set and all(empty) and ev.speech_present:
        return (AnswerStatus.CAPTURE_MISS, *accept, AnswerSource.NONE)
    if ev.early_speech and not ev.echo_mode and not ev.reask_used:
        return (
            AnswerStatus.EARLY_ANSWER,
            ScoreAction.REASK,
            PromptKey.REASK_SAY_AGAIN,
            AnswerSource.NONE,
        )
    if all(empty):
        st_ = AnswerStatus.CAPTURE_MISS if ev.speech_present else AnswerStatus.NO_RESPONSE
        return (st_, *accept, AnswerSource.NONE)
    p = LEX.parse([s.text for s in ev.segments])
    n = len(p.digits)
    if n == 3:
        status = AnswerStatus.OK
    elif n > 3:
        status = AnswerStatus.AMBIGUOUS
    elif n >= 1:
        status = AnswerStatus.NO_DIGITS if p.unparsed else AnswerStatus.PARTIAL
    elif p.intent == "REPEAT":
        status = AnswerStatus.REPEAT_REQUEST
    elif p.intent == "DONT_KNOW":
        status = AnswerStatus.DONT_KNOW
    else:
        status = AnswerStatus.NO_DIGITS
    if status == AnswerStatus.REPEAT_REQUEST:
        if not ev.replay_used:
            return (status, ScoreAction.REPLAY, None, AnswerSource.SPEECH)
        status = AnswerStatus.NO_DIGITS
    if status in (AnswerStatus.NO_DIGITS, AnswerStatus.AMBIGUOUS) and not ev.reask_used:
        return (status, ScoreAction.REASK, PromptKey.REASK_THREE, AnswerSource.SPEECH)
    return (status, *accept, AnswerSource.SPEECH)


@settings(max_examples=1000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(evidences(), triplets)
def test_P_SC_04_oracle_agrees(ev: AnswerEvidence, t: tuple[int, int, int]) -> None:
    res = LEX.score(spec(t), ev)
    assert (res.status, res.action, res.reask_prompt, res.source) == oracle(ev, t)
    ScoringResult.model_validate(res.model_dump())  # validators never fire


@SETTINGS
@given(st.lists(segment_text, max_size=4))
def test_P_SC_05_digits_so_far_matches_parse(texts: list[str]) -> None:
    assert LEX.digits_so_far(Lang.TA, texts) == min(len(LEX.parse(texts).digits), 12)


@SETTINGS
@given(triplets, st.lists(st.integers(0, 9), max_size=8), st.integers(0, 3))
def test_P_SC_06_align_properties(t: tuple[int, int, int], r: list[int], k: int) -> None:
    assert align(t, t) == (True, True, True)
    assert align((), t) == (False, False, False)
    assert align(t[:k], t)[:k] == (True,) * k
    assert len(align(r, t)) == 3


@SETTINGS
@given(st.integers(1, 9), st.integers(1, 9), st.integers(1, 9), st.data())
def test_P_SC_07_grouping(d: int, a: int, b: int, data: st.DataObject) -> None:
    if a == 1:
        word = data.draw(st.sampled_from([EN_TEENS[b - 1], TA_TEENS[b - 1]]))
    else:
        tens = data.draw(st.sampled_from([EN_TENS[a], *TA_TENS[a]]))
        word = f"{tens} {data.draw(form(b))}"
    p = LEX.parse([f"{data.draw(form(d))} {word}"])
    assert p.digits == (d, a, b)
    assert p.number_form
