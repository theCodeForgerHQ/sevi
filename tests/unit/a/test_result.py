# ruff: noqa: PT018, RUF001, E501
"""Unit tests for core/result.py (Person A spec §8.6: U-RES-01..23, U-RES-30)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from sevi.contracts.enums import (
    Band,
    Censor,
    Channel,
    EndReason,
    FriendsVariant,
    Lang,
    Mode,
    NoBandReason,
    ProcedureTerminal,
    ReferReason,
    Surface,
    Transducer,
    TruthRule,
    WordingClass,
)
from sevi.contracts.ports import ResultClassifier
from sevi.contracts.result import ResultDecision
from sevi.core.result import (
    ALL4,
    PH3,
    TruthTableClassifier,
    build_classifier,
    classify,
    find_cell,
    srt_display,
    t8_decision,
)
from sevi.fakes.scenarios.result_inputs import OTHER_BANK, cell, inp, norms, srt

S, P, R, OP = Surface.SPOKEN, Surface.PAGE, Surface.REPORT, Surface.OPERATOR
PARTIAL = srt(n_scored=10, srt_db=None, terminal=None)

# (test id, ResultInput overrides, rule, band, refer/no-band reason, wording, surfaces)
ROWS: list[tuple[str, dict[str, Any], TruthRule, Band, Any, WordingClass, tuple[Surface, ...]]] = [
    ("U_RES_01_T1", {"srt": srt(n_scored=0, srt_db=None, terminal=ProcedureTerminal.VOLUME_FAILED, volume_failed=True)},
     TruthRule.T1, Band.REFER, ReferReason.LOUD_NOT_HEARD, WordingClass.W_REF_LOUD, ALL4),
    ("U_RES_02_T2", {"srt": srt(n_scored=0, srt_db=None, terminal=ProcedureTerminal.PRACTICE_FAILED, practice_failed=True)},
     TruthRule.T2, Band.REFER, ReferReason.PRACTICE_FAILED, WordingClass.W_REF_PRACTICE, ALL4),
    ("U_RES_03_T3", {"srt": srt(n_scored=12, srt_db=None, terminal=ProcedureTerminal.LOUD_FAILED, loud_wrong_total=2)},
     TruthRule.T3, Band.REFER, ReferReason.NO_RESPONSE_LOUD, WordingClass.W_REF_LOUD, ALL4),
    ("U_RES_04_T4", {"srt": None, "end_reason": EndReason.CANT_HEAR},
     TruthRule.T4, Band.REFER, ReferReason.CANT_HEAR, WordingClass.W_REF_LOUD, ALL4),
    ("U_RES_05_T5", {"srt": srt(ceiling_fail=True, flags=5), "end_reason": EndReason.CALLER_HANGUP},
     TruthRule.T5, Band.REFER, ReferReason.CEILING, WordingClass.W_REF_CEILING, ALL4),
    ("U_RES_06_T6", {"srt": srt(flags=4)},
     TruthRule.T6, Band.INCONCLUSIVE, None, WordingClass.W_INCONC, ALL4),
    ("U_RES_07_T7", {"srt": PARTIAL, "end_reason": EndReason.OPERATOR_ABORT},
     TruthRule.T7, Band.INCOMPLETE, None, WordingClass.W_INC, ALL4),
    ("U_RES_08_T8", {"srt": PARTIAL, "end_reason": EndReason.SERVER_ERROR},
     TruthRule.T8, Band.FAILED, None, WordingClass.W_SYSFAIL, ALL4),
    ("U_RES_09_T9", {"srt": None, "end_reason": EndReason.CONSENT_NO_REPLY},
     TruthRule.T9, Band.NOT_STARTED, None, WordingClass.W_NOREPLY, (OP, R)),
    ("U_RES_10_T10", {"srt": None, "end_reason": EndReason.CONSENT_DECLINED},
     TruthRule.T10, Band.DECLINED, None, WordingClass.W_DECLINED, (S, P)),
    ("U_RES_11_T11", {"lang": Lang.EN, "experimental_language": True, "norms": None},
     TruthRule.T11, Band.NO_BAND, NoBandReason.EXPERIMENTAL_LANGUAGE, WordingClass.W_NOBAND_LT6, ALL4),
    ("U_RES_12_T12", {"srt": srt(srt_db=-3.5)},
     TruthRule.T12, Band.REFER, ReferReason.SRT, WordingClass.W_REF_SRT, ALL4),
    ("U_RES_13_T13", {"srt": srt(srt_db=-5.0)},
     TruthRule.T13, Band.BORDERLINE, None, WordingClass.W_BORDER, ALL4),
    ("U_RES_14_T14", {"hhie_total": 12},
     TruthRule.T14, Band.PASS_WITH_NOTE, None, WordingClass.W_PASS_NOTE, ALL4),
    ("U_RES_15_T15", {},
     TruthRule.T15, Band.PASS, None, WordingClass.W_PASS, ALL4),
    ("U_RES_22_T10b", {"srt": srt(srt_db=0.5)},
     TruthRule.T10B, Band.REFER, ReferReason.SRT_ABSOLUTE, WordingClass.W_REF_ABS, ALL4),
]  # fmt: skip


@pytest.mark.parametrize(
    ("over", "rule", "band", "reason", "wording", "surfaces"),
    [r[1:] for r in ROWS],
    ids=[r[0] for r in ROWS],
)
def test_U_RES_01_to_15_truth_table_rows(
    over: dict[str, Any],
    rule: TruthRule,
    band: Band,
    reason: Any,
    wording: WordingClass,
    surfaces: tuple[Surface, ...],
) -> None:
    d = classify(inp(**over))
    assert (d.rule, d.band, d.wording_class, d.surfaces) == (rule, band, wording, surfaces)
    if isinstance(reason, ReferReason):
        assert (d.refer_reason, d.no_band_reason) == (reason, None)
    elif isinstance(reason, NoBandReason):
        assert (d.refer_reason, d.no_band_reason) == (None, reason)
    else:
        assert (d.refer_reason, d.no_band_reason) == (None, None)


def test_U_RES_15_pass_carries_norm_fields() -> None:
    d = classify(inp())
    assert d.norm_cell is not None and d.n_norm == 7 and d.ages == "18-35"
    assert d.nh95_db == pytest.approx(-6.800144)
    assert d.srt_display == "−8.0"


def test_U_RES_16_bug_path_23_trials_is_t8() -> None:
    s = srt(n_scored=23, srt_db=None, terminal=None)
    d = classify(inp(srt=s))
    assert d == t8_decision(s)
    assert d.rule is TruthRule.T8 and d.band is Band.FAILED


WORKED: list[tuple[str, float, dict[str, Any], TruthRule]] = [
    ("m8.2_pass", -8.2, {}, TruthRule.T15),
    ("m7.5_v0_zone", -7.5, {}, TruthRule.T13),
    ("m5.0_border", -5.0, {}, TruthRule.T13),
    ("m3.5_refer", -3.5, {}, TruthRule.T12),
    ("p0.5_abs", 0.5, {}, TruthRule.T10B),
    ("p0.5_en_abs", 0.5, {"lang": Lang.EN, "experimental_language": True, "norms": None}, TruthRule.T10B),
    ("m0.6_en_noband", -0.6, {"lang": Lang.EN, "experimental_language": True, "norms": None}, TruthRule.T11),
    ("m8.2_rev3", -8.2, {"reversals": 3}, TruthRule.T13),
    ("m8.2_v1", -8.2, {"cell_n": 10}, TruthRule.T15),
]  # fmt: skip


@pytest.mark.parametrize(
    ("x", "extra", "rule"), [w[1:] for w in WORKED], ids=[w[0] for w in WORKED]
)
def test_U_RES_17_worked_band_table(x: float, extra: dict[str, Any], rule: TruthRule) -> None:
    extra = dict(extra)
    rev = extra.pop("reversals", 8)
    n = extra.pop("cell_n", 7)
    d = classify(
        inp(srt=srt(srt_db=x, reversals=rev), norms=extra.pop("norms", norms(cell(n=n))), **extra)
    )
    assert d.rule is rule
    if rule is TruthRule.T11:
        assert d.no_band_reason is NoBandReason.EXPERIMENTAL_LANGUAGE


@pytest.mark.parametrize(
    ("x", "out"),
    [(-5.43, "−5.5"), (-8.2, "−8.0"), (2.24, "+2.0"), (-0.24, "0.0"), (-0.26, "−0.5"),
     (-0.25, "0.0"), (0.25, "+0.5"), (0.0, "0.0")],
)  # fmt: skip
def test_U_RES_17_srt_display(x: float, out: str) -> None:
    assert srt_display(x) == out


def _ref_mixed(n: int) -> dict[str, Any]:
    return {
        "mode": Mode.MIXED,
        "mode_requested": Mode.VOICE,
        "srt": srt(mode=Mode.MIXED),
        "norms": norms(cell(n=n, mode=Mode.VOICE)),
    }


def test_U_RES_18_t11_reason_order_and_wording() -> None:
    d = classify(inp(experimental_language=True, friends=FriendsVariant.DISALLOWED))
    assert d.no_band_reason is NoBandReason.EXPERIMENTAL_LANGUAGE

    d = classify(inp(**_ref_mixed(7)))
    assert (d.no_band_reason, d.wording_class) == (
        NoBandReason.MIXED_MODE,
        WordingClass.W_NOBAND_RANGE,
    )
    assert (d.n_norm, d.ages) == (7, "18-35")

    d = classify(inp(**_ref_mixed(5)))
    assert d.wording_class is WordingClass.W_NOBAND_LT6 and d.n_norm is None

    d = classify(inp(transducer=Transducer.SPEAKER))
    assert (d.no_band_reason, d.wording_class) == (
        NoBandReason.SPEAKER_TRANSDUCER,
        WordingClass.W_NOBAND_UNCOMPARABLE,
    )

    d = classify(inp(channel=Channel.PHONE, transducer=Transducer.HANDSET))
    assert d.no_band_reason is NoBandReason.NO_CELL and d.surfaces == PH3 == (S, R, OP)

    d = classify(inp(norms=None))
    assert d.no_band_reason is NoBandReason.NO_CELL and d.wording_class is WordingClass.W_NOBAND_LT6

    d = classify(inp(norms=norms(cell(bank_hash=OTHER_BANK))))
    assert (d.no_band_reason, d.wording_class) == (
        NoBandReason.NORMS_BANK_MISMATCH,
        WordingClass.W_NOBAND_UNCOMPARABLE,
    )
    assert d.n_norm == 7  # MISMATCH keeps the matched cell's fields

    d = classify(inp(norms=norms(cell(n=4))))
    assert d.no_band_reason is NoBandReason.CELL_LT6 and d.n_norm == 4

    d = classify(inp(friends=FriendsVariant.DISALLOWED))
    assert d.wording_class is WordingClass.W_NOBAND_UNCOMPARABLE and d.norm_cell is None


@pytest.mark.parametrize(
    ("end", "terminal", "spoken"),
    [
        (EndReason.CALLER_HANGUP, None, False),
        (EndReason.SESSION_TIMEOUT, None, True),
        (EndReason.OPERATOR_ABORT, None, True),
        (EndReason.COMPLETED, ProcedureTerminal.NO_RESPONSE_END, True),
    ],
)
def test_U_RES_19_t7_surfaces(
    end: EndReason, terminal: ProcedureTerminal | None, spoken: bool
) -> None:
    d = classify(inp(end_reason=end, srt=srt(n_scored=14, srt_db=None, terminal=terminal)))
    assert d.rule is TruthRule.T7
    assert d.surfaces == (ALL4 if spoken else (P, R, OP))


def test_U_RES_19_hangup_after_24_keeps_band() -> None:
    d = classify(inp(end_reason=EndReason.CALLER_HANGUP))
    assert d.rule is TruthRule.T15


def test_U_RES_20_censor_high_never_pass() -> None:
    d = classify(inp(srt=srt(srt_db=-12.0, censor=Censor.HIGH)))
    assert d.rule is TruthRule.T13


def test_U_RES_21_perf() -> None:
    clf = build_classifier()
    inputs = [inp(srt=srt(srt_db=-12 + (i % 130) / 10)) for i in range(2000)]
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        clf.classify(x)
        times.append(time.perf_counter() - t0)
    times.sort()
    assert times[int(0.99 * len(times))] <= 0.005
    d = clf.classify(inputs[0])
    wt = []
    for _ in range(2000):
        t0 = time.perf_counter()
        clf.wording(d, Lang.TA, Surface.PAGE)
        wt.append(time.perf_counter() - t0)
    wt.sort()
    assert wt[int(0.99 * len(wt))] <= 0.001


@pytest.mark.parametrize(
    ("x", "over", "rule"),
    [
        (0.5, {}, TruthRule.T10B),
        (0.0, {}, TruthRule.T10B),
        (-0.01, {"norms": None}, TruthRule.T11),
        (-0.01, {}, TruthRule.T12),
    ],
)
def test_U_RES_22_t10b_boundary(x: float, over: dict[str, Any], rule: TruthRule) -> None:
    assert classify(inp(srt=srt(srt_db=x), **over)).rule is rule


def test_U_RES_22_t10b_phone_surfaces_and_no_norm_fields() -> None:
    d = classify(inp(srt=srt(srt_db=1.0), channel=Channel.PHONE, transducer=Transducer.HANDSET))
    assert d.rule is TruthRule.T10B and d.surfaces == (S, R, OP)
    d = classify(inp(srt=srt(srt_db=1.0)))
    assert d.norm_cell is None and d.n_norm is None and d.ages is None and d.nh95_db is None
    assert d.srt_display == "+1.0"


def test_U_RES_23_build_classifier_on_repo_files() -> None:
    clf = build_classifier()
    assert isinstance(clf, ResultClassifier)
    assert isinstance(clf, TruthTableClassifier)
    w = clf.wording(clf.classify(inp()), Lang.EN, Surface.PAGE)
    assert "−8.0" not in w.text  # PASS page cites N and AGES, not SRT
    assert "n = 7" in w.text and "18-35" in w.text


@pytest.mark.parametrize("s", [None, PARTIAL], ids=["none", "partial"])
@pytest.mark.parametrize("end", [EndReason.SERVER_ERROR, EndReason.SOCKET_LOST])
def test_U_RES_30_t8_decision_equals_classify(s: Any, end: EndReason) -> None:
    d = classify(inp(srt=s, end_reason=end))
    assert d == t8_decision(s)
    assert d.surfaces == ALL4 and d.wording_class is WordingClass.W_SYSFAIL
    assert ResultDecision.model_validate(d.model_dump()) == d


def test_find_cell_exact_match_last_wins() -> None:
    a, b = cell(n=7), cell(n=9)
    assert find_cell(norms(a, b), a.key) == b
    assert find_cell(None, a.key) is None
    assert find_cell(norms(cell(mode=Mode.VOICE)), a.key) is None
