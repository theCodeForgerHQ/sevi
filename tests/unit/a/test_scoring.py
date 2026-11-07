"""Unit tests for core/scoring.py (spec §7.4-§7.7; contracts §4.5.1, §4.13 C1-C5)."""

from __future__ import annotations

import random
import time
import unicodedata
from typing import Any

import pytest

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
from sevi.contracts.ports import AnswerScorer
from sevi.contracts.trial import (
    AnswerEvidence,
    DtmfPress,
    PresentationSpec,
    ScoringResult,
    SttSegment,
)
from sevi.core.scoring import LexiconScorer, align, load_lexicons

LEX = load_lexicons()
OK, PARTIAL, AMB = AnswerStatus.OK, AnswerStatus.PARTIAL, AnswerStatus.AMBIGUOUS
NOD, DK, RR = AnswerStatus.NO_DIGITS, AnswerStatus.DONT_KNOW, AnswerStatus.REPEAT_REQUEST
CM, NR, EA = AnswerStatus.CAPTURE_MISS, AnswerStatus.NO_RESPONSE, AnswerStatus.EARLY_ANSWER
ACC, REASK, REPLAY = ScoreAction.ACCEPT, ScoreAction.REASK, ScoreAction.REPLAY
THREE, SAY_AGAIN = PromptKey.REASK_THREE, PromptKey.REASK_SAY_AGAIN


def spec(triplet: tuple[int, int, int] = (4, 7, 2), pid: int = 1) -> PresentationSpec:
    return PresentationSpec(
        presentation_id=pid,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=-6.0,
        triplet=triplet,
        pool_index=0,
        noise_seed=1,
    )


def evidence(*texts: str, pid: int = 1, **kw: Any) -> AnswerEvidence:
    dtmf = tuple(
        DtmfPress(key=k, at_ms=float(i * 100))  # type: ignore[arg-type]
        for i, k in enumerate(kw.pop("dtmf", ()))
    )
    base: dict[str, Any] = {
        "presentation_id": pid,
        "lang": Lang.TA,
        "mode": Mode.VOICE,
        "window_index": 1,
        "window_open_mono_ms": 0.0,
        "window_close_mono_ms": 3000.0,
        "close_reason": WindowCloseReason.SETTLE,
        "segments": tuple(
            SttSegment(text=t, received_ms=float(100 * i), after_commit=False)
            for i, t in enumerate(texts)
        ),
        "dtmf": dtmf,
        "speech_present": any(t.strip() for t in texts),
        "speech_ms": 900,
        "early_speech": False,
        "echo_mode": False,
        "system_events": (),
        "reask_used": False,
        "replay_used": False,
    }
    base.update(kw)
    return AnswerEvidence(**base)


# Row: (row id, segment texts, evidence overrides, triplet, status, action, prompt,
#       recognised, triplet_correct, extra assertions)
R472 = (4, 7, 2)
ROWS: list[
    tuple[
        str,
        tuple[str, ...],
        dict[str, Any],
        tuple[int, int, int],
        AnswerStatus,
        ScoreAction,
        PromptKey | None,
        tuple[int, ...],
        bool,
        dict[str, Any],
    ]
] = []


def row(
    rid: str,
    texts: tuple[str, ...],
    status: AnswerStatus,
    action: ScoreAction,
    recognised: tuple[int, ...] = (),
    correct: bool = False,
    *,
    prompt: PromptKey | None = None,
    triplet: tuple[int, int, int] = R472,
    ev: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    if action is REASK and prompt is None:
        prompt = THREE
    ROWS.append((rid, texts, ev or {}, triplet, status, action, prompt, recognised, correct, extra))


# fmt: off
row("01", ("472",), OK, ACC, R472, True, source=AnswerSource.SPEECH)
row("02", ("47", "2"), OK, ACC, R472, True)
for i, t in enumerate(["4 7 2.", "4-7-2", "4,7,2"]):
    row(f"03{'abc'[i]}", (t,), OK, ACC, R472, True)
for i, t in enumerate(["naalu ezhu rendu", "nalu yezhu randu"]):
    row(f"04{'ab'[i]}", (t,), OK, ACC, R472, True)
for i, t in enumerate(["நான்கு ஏழு இரண்டு", "நாலு ஏழு ரெண்டு"]):
    row(f"05{'ab'[i]}", (t,), OK, ACC, R472, True)
row("06", ("நாலுஏழுரெண்டு",), OK, ACC, R472, True)
for i, t in enumerate(["four seven two", "FOUR Seven TWO", "for seven to"]):
    row(f"07{'abc'[i]}", (t,), OK, ACC, R472, True)
row("08", ("four 7 ரெண்டு",), OK, ACC, R472, True)
for i, t in enumerate(["௪௭௨", "४७२"]):
    row(f"09{'ab'[i]}", (t,), OK, ACC, R472, True)
row("10", ("umm naalu, ezhu... rendu sari",), OK, ACC, R472, True)
row("11", ("the numbers are four seven two",), OK, ACC, R472, True)
row("12", ("naalu ezhu rendu nu ninaikiren",), OK, ACC, R472, True)
for i, t in enumerate(["நானூற்று எழுபத்து இரண்டு", "four hundred seventy two"]):
    row(f"13{'ab'[i]}", (t,), OK, ACC, R472, True, number_form=True)
for i, t in enumerate(["foar sevun tu", "fore sevan too"]):
    row(f"14{'ab'[i]}", (t,), OK, ACC, R472, True)
row("15", ("four hundred and seventy two",), OK, ACC, R472, True, number_form=True)
row("16", ("47",), PARTIAL, ACC, (4, 7), position_correct=(True, True, False))
row("17", ("72",), PARTIAL, ACC, (7, 2), position_correct=(False, True, True))
row("18a", ("742",), OK, ACC, (7, 4, 2))
row("18b", ("447",), OK, ACC, (4, 4, 7))
row("18c", ("947",), OK, ACC, (9, 4, 7))
row("19a", ("4721",), AMB, REASK, (4, 7, 2, 1))
row("19b", ("1,000",), AMB, REASK, (1, 0, 0, 0))
row("19c", ("naalu anju illa ezhu rendu",), AMB, REASK, (4, 5, 7, 2))
row("20a", ("4721",), AMB, ACC, (4, 7, 2, 1), ev={"reask_used": True}, valid=True)
row("20b", ("1,000",), AMB, ACC, (1, 0, 0, 0), ev={"reask_used": True}, valid=True)
row("20c", ("naalu anju illa ezhu rendu",), AMB, ACC, (4, 5, 7, 2), ev={"reask_used": True})
for i, t in enumerate(["theriyala", "don't know", "not sure"]):
    row(f"21{'abc'[i]}", (t,), DK, ACC, valid=True)
for i, t in enumerate(["marupadiyum", "come again"]):
    row(f"22{'ab'[i]}", (t,), RR, REPLAY)
row("23", ("marupadiyum",), NOD, REASK, ev={"replay_used": True})
for i, t in enumerate(["hello", "sari sari"]):
    row(f"24{'ab'[i]}", (t,), NOD, REASK)
row("25", ("hello",), NOD, ACC, ev={"reask_used": True}, valid=True)
row("26", ("",), CM, ACC, ev={"speech_present": True}, valid=False, source=AnswerSource.NONE)
row("27", ("",), NR, ACC, ev={"speech_present": False}, no_response=True)
row("28", ("472",), CM, ACC, ev={"system_events": (SystemEvent.BREAKER_TRIP,)}, valid=False)
row("29", ("",), CM, ACC, ev={"system_events": (SystemEvent.STT_ERROR,), "speech_present": True},
    valid=False)
row("30", ("472",), OK, ACC, R472, True, ev={"system_events": (SystemEvent.STT_ERROR,)})
row("31", ("472",), EA, REASK, prompt=SAY_AGAIN, ev={"early_speech": True, "echo_mode": False})
row("32", ("472",), OK, ACC, R472, True, ev={"early_speech": True, "echo_mode": True})
row("33", ("472",), PARTIAL, ACC, (4, 7), ev={"dtmf": ("4", "7")}, source=AnswerSource.DTMF)
row("34", (), NR, ACC, ev={"mode": Mode.KEYPAD}, source=AnswerSource.NONE)
row("35", (), OK, ACC, R472, True, ev={"mode": Mode.KEYPAD, "dtmf": ("4", "7", "2")},
    source=AnswerSource.DTMF, normalised_tokens=("4", "7", "2"))
row("36", ("4 7 2 4 7 2",), AMB, REASK, (4, 7, 2, 4, 7, 2))
row("37", ("4 7 2 4 7 2 4 7 2 4 7 2 4",), AMB, REASK, (4, 7, 2) * 4)
row("38", ("oh four seven two",), OK, ACC, R472, True)
row("39a", ("one",), PARTIAL, ACC, (1,))
row("39b", ("one more time",), RR, REPLAY)
row("40", ("faiv siks ait",), OK, ACC, (5, 6, 8), True, triplet=(5, 6, 8), ev={"lang": Lang.EN})
row("41", (unicodedata.normalize("NFD", "நான்கு ஏழு இரண்டு"),), OK, ACC, R472, True)
row("42", ("theriyala 4",), PARTIAL, ACC, (4,))
row("43", ("four seventy-two",), OK, ACC, R472, True, number_form=True)
row("44", ("நாலு எழுபத்தி ரெண்டு",), OK, ACC, R472, True, number_form=True)
row("45a", ("four ninety two",), OK, ACC, (4, 9, 2), True, triplet=(4, 9, 2), number_form=True)
row("45b", ("four hundred and ninety two",), OK, ACC, (4, 9, 2), True, triplet=(4, 9, 2),
    number_form=True)
row("46", ("forty-seven two",), NOD, REASK)
row("47", ("four seventy",), NOD, REASK)
row("48", ("forty-seven two",), NOD, ACC, ev={"reask_used": True}, valid=True)
row("49", ("\uff14\uff17\uff12",), OK, ACC, R472, True)
row("50", ("நாலேழு ரெண்டு",), OK, ACC, R472, True)
row("51", ("ஃபோர் செவன் டூ",), OK, ACC, R472, True)
row("52", ("நாலு எழு ரண்டு",), OK, ACC, R472, True)
row("53", ("naanku ezu erandu",), OK, ACC, R472, True)
row("54", ("fore fiv nain",), OK, ACC, (4, 5, 9), True, triplet=(4, 5, 9))
row("55", ("naalu ezhu illa",), NOD, REASK)
row("56", ("I want to say 472",), AMB, REASK, (2, 4, 7, 2))
row("57", ("seventy two",), NOD, REASK)
row("58", ("double five two",), OK, ACC, (5, 5, 2), number_form=False)
row("59", ("triple four",), OK, ACC, (4, 4, 4))
row("60", ("double",), NOD, REASK)
row("61", ("won tu ait",), OK, ACC, (1, 2, 8), True, triplet=(1, 2, 8))
row("62", ("oh",), NOD, REASK)
row("63", ("டபுள் அஞ்சு ரெண்டு",), OK, ACC, (5, 5, 2))
# fmt: on

ROW_IDS = [r[0] for r in ROWS]
U_SC_41_48 = {"43", "44", "45a", "45b", "46", "47", "49", "50", "51"}
U_SC_49 = {"58", "59", "60", "61", "62"}


def test_U_SC_01_table_covers_all_63_rows() -> None:
    assert sorted({int(rid[:2]) for rid in ROW_IDS}) == list(range(1, 64))


def _check_row(
    texts: tuple[str, ...],
    ev: dict[str, Any],
    triplet: tuple[int, int, int],
    status: AnswerStatus,
    action: ScoreAction,
    prompt: PromptKey | None,
    recognised: tuple[int, ...],
    correct: bool,
    extra: dict[str, Any],
) -> ScoringResult:
    res = LEX.score(spec(triplet), evidence(*texts, **ev))
    assert res.status is status
    assert res.action is action
    assert res.reask_prompt is prompt
    assert res.recognised == recognised
    assert res.triplet_correct is correct
    assert res.position_correct == align(recognised, triplet)
    assert res.valid is (status is not CM)
    assert res.no_response is (status is NR)
    if "number_form" not in extra:
        assert res.number_form is False
    for field, want in extra.items():
        assert getattr(res, field) == want, field
    return res


@pytest.mark.parametrize(
    ("texts", "ev", "triplet", "status", "action", "prompt", "recognised", "correct", "extra"),
    [r[1:] for r in ROWS],
    ids=ROW_IDS,
)
def test_U_SC_01_edge_case_table(
    texts: tuple[str, ...],
    ev: dict[str, Any],
    triplet: tuple[int, int, int],
    status: AnswerStatus,
    action: ScoreAction,
    prompt: PromptKey | None,
    recognised: tuple[int, ...],
    correct: bool,
    extra: dict[str, Any],
) -> None:
    _check_row(texts, ev, triplet, status, action, prompt, recognised, correct, extra)


@pytest.mark.parametrize("rid", sorted(U_SC_41_48))
def test_U_SC_41_48_v4_rows(rid: str) -> None:
    r = ROWS[ROW_IDS.index(rid)]
    _check_row(*r[1:])


@pytest.mark.parametrize("rid", sorted(U_SC_49))
def test_U_SC_49_rule_C5_rows(rid: str) -> None:
    r = ROWS[ROW_IDS.index(rid)]
    _check_row(*r[1:])


@pytest.mark.parametrize(
    ("recognised", "want"),
    [
        ((4, 7, 2), (True, True, True)),
        ((4, 7), (True, True, False)),
        ((7, 2), (False, True, True)),
        ((2, 7, 4), (False, True, False)),
        ((4, 4, 7, 2), (True, True, True)),
        ((), (False, False, False)),
        ((9, 9, 9), (False, False, False)),
    ],
)
def test_U_SC_02_align_examples(recognised: tuple[int, ...], want: tuple[bool, ...]) -> None:
    assert align(recognised, (4, 7, 2)) == want


def test_U_SC_03_mismatched_presentation_id_raises() -> None:
    with pytest.raises(ValueError, match="presentation_id"):
        LEX.score(spec(pid=2), evidence("472", pid=1))


def test_U_SC_04_digits_and_intent_so_far() -> None:
    ta = Lang.TA
    assert LEX.digits_so_far(ta, ["47"]) == 2
    assert LEX.digits_so_far(ta, ["47", "2"]) == 3
    assert LEX.digits_so_far(ta, ["1" * 15]) == 12
    assert LEX.intent_so_far(ta, ["theriyala"]) == "DONT_KNOW"
    assert LEX.intent_so_far(ta, ["theriyala", "marupadiyum"]) == "REPEAT"
    assert LEX.intent_so_far(ta, ["4 theriyala"]) is None


def test_U_SC_05_perf_budgets() -> None:
    rng = random.Random(5)  # noqa: S311 - deterministic test data
    words = [*(w for d in range(10) for w in LEX.forms_all_langs(d)), *LEX.fillers, "hello"]
    cases = []
    for _ in range(10_000):
        texts = tuple(" ".join(rng.choices(words, k=rng.randint(1, 6))) for _ in range(2))
        cases.append(texts)
    sp = spec()
    score_t: list[float] = []
    dsf_t: list[float] = []
    for texts in cases:
        ev = evidence(*texts)
        t0 = time.perf_counter()
        LEX.score(sp, ev)
        t1 = time.perf_counter()
        LEX.digits_so_far(Lang.TA, texts)
        t2 = time.perf_counter()
        score_t.append(t1 - t0)
        dsf_t.append(t2 - t1)
    score_t.sort()
    dsf_t.sort()
    p99 = int(0.99 * len(cases))
    assert score_t[p99] <= 0.005
    assert dsf_t[p99] <= 0.002


@pytest.mark.skip(reason="U-SC-06: no VT-PRS-05 transcripts recorded yet (needs a live run)")
def test_U_SC_06_vt_prs_05_replay() -> None:  # pragma: no cover
    raise AssertionError


def test_scorer_satisfies_port() -> None:
    assert isinstance(LEX, AnswerScorer)
    assert isinstance(LEX, LexiconScorer)


def test_dtmf_ignores_speech_and_uses_first_three_keys() -> None:
    res = LEX.score(spec(), evidence("one two three", dtmf=("4", "7", "2", "9")))
    assert res.source is AnswerSource.DTMF
    assert res.recognised == R472
    assert res.triplet_correct
    assert res.normalised_tokens == ("4", "7", "2")


def test_mic_lost_and_reconnect_are_capture_miss() -> None:
    for e in (SystemEvent.MIC_LOST, SystemEvent.RECONNECT):
        res = LEX.score(spec(), evidence("472", system_events=(e,)))
        assert res.status is CM
        assert not res.valid


def test_early_speech_after_reask_is_scored() -> None:
    res = LEX.score(spec(), evidence("472", early_speech=True, reask_used=True))
    assert res.status is OK


def test_repeat_request_after_replay_and_reask_accepts() -> None:
    res = LEX.score(spec(), evidence("marupadiyum", replay_used=True, reask_used=True))
    assert (res.status, res.action) == (NOD, ACC)
