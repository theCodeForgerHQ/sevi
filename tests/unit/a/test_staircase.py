"""Unit tests for core/staircase.py (Person A spec §6.5, contracts §4.3.1, §4.6.1)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sevi.contracts.enums import (
    AnswerSource,
    AnswerStatus,
    Censor,
    Condition,
    Mode,
    PresentationKind,
    ProcedurePhase,
    ProcedureTerminal,
    PromptKey,
    ScoreAction,
)
from sevi.contracts.errors import ErrorCode, ProcedureProtocolError
from sevi.contracts.ports import AdaptiveProcedure
from sevi.contracts.procedure import ProcedureState, ScoredEntry
from sevi.contracts.rng import derive_seed
from sevi.contracts.trial import ScoringResult
from sevi.core.staircase import (
    PROCEDURE,
    Staircase,
    _next_snr,
    _track,
    count_reversals,
    max_consec_valid_wrong_at_ceiling,
)
from sevi.fakes.scenarios.staircase_driver import drive, make_config, result_for

EX_A = "CCCWCWWCCWCWCCWWCWCCWWCW"
EX_A_SNRS = [
    0,
    -4,
    -8,
    -12,
    -10,
    -12,
    -10,
    -8,
    -10,
    -12,
    -10,
    -12,
    -10,
    -12,
    -14,
    -12,
    -10,
    -12,
    -10,
    -12,
    -14,
    -12,
    -10,
    -12,
]


def scored_cfg(**kw: object):  # type: ignore[no-untyped-def]
    return make_config(skip_volume_and_practice=True, **kw)


def run_scored(answers: list[str], **kw: object) -> ProcedureState:
    s, _ = drive(scored_cfg(**kw), answers)
    return s


def test_port_conformance() -> None:
    assert isinstance(PROCEDURE, AdaptiveProcedure)
    assert isinstance(PROCEDURE, Staircase)


def test_U_ST_010_contract_worked_example_trials_1_to_8() -> None:
    """contracts §4.3.1 table, against the real procedure and the pure rule."""
    s = run_scored(list("CCCWCWWC"))
    assert [e.snr_db for e in s.scored] == [0, -4, -8, -12, -10, -12, -10, -8]
    assert s.current_snr_db == -10
    assert s.pending is not None
    assert s.pending.snr_db == -10
    assert s.pending.scored_index == 9
    fws = [False, False, False, True, True, True, True, True]
    assert _track([c == "C" for c in "CCCWCWWC"]) == (0, -4, -8, -12, -10, -12, -10, -8, -10)
    # first_wrong_seen flips exactly at trial 4
    for k in range(1, 9):
        assert run_scored(list("CCCWCWWC"[:k])).first_wrong_seen is fws[k - 1]


def test_U_ST_010_silent_caller_trace() -> None:
    s, specs = drive(scored_cfg(), ["NR", "NR", "NR"])
    assert [e.snr_db for e in s.scored] == [0, 2, 4]
    assert s.current_snr_db == 6
    assert s.phase is ProcedurePhase.LOUD
    assert specs[-1].kind is PresentationKind.LOUD
    assert specs[-1].pre_prompt is PromptKey.LOUD_CHECK


def test_U_ST_011_example_A_full_24() -> None:
    s = run_scored(list(EX_A))
    r = PROCEDURE.summarize(s)
    assert s.terminal is ProcedureTerminal.COMPLETE
    assert s.pending is None
    assert s.phase is ProcedurePhase.DONE
    assert list(r.track_db[:24]) == EX_A_SNRS
    assert r.virtual_25_db == -10
    assert r.srt_db == pytest.approx(-234 / 21, abs=1e-9)
    assert r.srt_db == pytest.approx(-11.142857, abs=1e-6)
    assert r.reversals == 13
    assert r.censor is Censor.NONE
    assert r.flags == 0
    assert r.ceiling_fail is False
    assert r.complete
    assert r.n_scored == 24
    assert len(r.track_db) == 25


def test_U_ST_012_example_B_all_correct() -> None:
    r = PROCEDURE.summarize(run_scored(["C"] * 24))
    assert r.track_db[:6] == (0, -4, -8, -12, -16, -20)
    assert r.virtual_25_db == -20
    assert r.srt_db == pytest.approx(-19.809524, abs=1e-6)
    assert r.srt_db == pytest.approx((-16 + 20 * -20) / 21, abs=1e-9)
    assert r.censor is Censor.LOW
    assert r.reversals == 0


def test_U_ST_012_example_C_alternating() -> None:
    r = PROCEDURE.summarize(run_scored(["C", "W"] * 12))
    assert r.track_db[:5] == (0, -4, -2, -4, -2)
    assert r.virtual_25_db == -2
    assert r.srt_db == pytest.approx(-62 / 21, abs=1e-9)
    assert r.srt_db == pytest.approx(-2.952381, abs=1e-6)
    assert r.reversals == 19


def test_U_ST_013_example_D_loud_failed() -> None:
    s, specs = drive(scored_cfg(), ["NR", "NR", "NR", "NR", "NR"])
    assert s.terminal is ProcedureTerminal.LOUD_FAILED
    assert s.loud_wrong_total == 2
    assert s.pending is None
    loud = [p for p in specs if p.kind is PresentationKind.LOUD]
    assert [p.pre_prompt for p in loud] == [PromptKey.LOUD_CHECK, None]
    assert all(p.snr_db == 10 for p in loud)
    assert s.current_snr_db == 6  # loud never moves the SNR
    assert PROCEDURE.summarize(s).terminal is ProcedureTerminal.LOUD_FAILED


def test_U_ST_013_example_D_no_response_end_ceiling_fail() -> None:
    ans = ["NR"] * 3 + ["C", "C"] + ["NR"] * 3 + ["C", "C"] + ["NR"] * 3
    s, specs = drive(scored_cfg(), ans)
    assert s.terminal is ProcedureTerminal.NO_RESPONSE_END
    assert [e.snr_db for e in s.scored] == [0, 2, 4, 6, 8, 10, 10, 10, 10]
    loud = [p for p in specs if p.kind is PresentationKind.LOUD]
    assert [p.pre_prompt for p in loud] == [PromptKey.LOUD_CHECK, None] * 2
    # after loud check #1 the scored trial resumes at the unchanged +6
    first_after = next(
        p for p in specs if p.kind is PresentationKind.SCORED and p.scored_index == 4
    )
    assert first_after.snr_db == 6
    assert first_after.pre_prompt is None
    r = PROCEDURE.summarize(s)
    assert r.valid_at_ceiling == 4
    assert r.correct_at_ceiling == 0
    assert r.ceiling_fail is True
    assert r.srt_db is None
    assert s.loud_checks_started == 2
    assert s.loud_wrong_total == 0


def test_U_ST_014_loud_wrong_is_cumulative() -> None:
    ans = ["NR"] * 3 + ["W", "C"] + ["NR"] * 3 + ["W"]
    s, _ = drive(scored_cfg(), ans)
    assert s.terminal is ProcedureTerminal.LOUD_FAILED
    assert s.loud_wrong_total == 2
    assert s.loud_checks_started == 2
    assert s.loud_results == (False,)


def test_U_ST_015_example_E_ceiling_stop() -> None:
    s = run_scored(["W"] * 11)
    assert [e.snr_db for e in s.scored] == [0, 2, 4, 6, 8, 10, 10, 10, 10, 10, 10]
    assert s.terminal is ProcedureTerminal.CEILING_STOP
    r = PROCEDURE.summarize(s)
    assert r.n_scored == 11
    assert r.srt_db is None
    assert r.ceiling_fail is True
    assert r.max_consec_valid_wrong_at_ceiling == 6


def test_U_ST_015_example_F_flag_removed_from_run() -> None:
    # reach +10 with five W (0,2,4,6,8 -> 10), then at +10: W, W, CM+CM(flag), W, W, W, W
    s = run_scored(["W"] * 5 + ["W", "W", "CM", "CM", "W", "W", "W"])
    assert s.terminal is None
    s = run_scored(["W"] * 5 + ["W", "W", "CM", "CM", "W", "W", "W", "W"])
    assert s.terminal is ProcedureTerminal.CEILING_STOP
    r = PROCEDURE.summarize(s)
    assert r.flags == 1
    assert r.max_consec_valid_wrong_at_ceiling == 6


def test_U_ST_016_ceiling_fail_without_stop() -> None:
    ans = ["W"] * 5 + ["W", "W", "C", "W", "W", "W"]
    # +10:W, +10:W, +10:C -> +8:W -> +10:W, +10:W
    s = run_scored(ans)
    snrs = [e.snr_db for e in s.scored]
    assert snrs[5:] == [10, 10, 10, 8, 10, 10]
    r = PROCEDURE.summarize(s)
    assert r.valid_at_ceiling == 5
    assert r.correct_at_ceiling == 1
    assert r.ceiling_fail is True
    assert s.terminal is None


def test_U_ST_017_volume_rules() -> None:
    cfg = make_config()
    s0 = PROCEDURE.start(cfg)
    assert s0.phase is ProcedurePhase.VOLUME
    assert s0.pending is not None
    assert s0.pending.pre_prompt is PromptKey.VOLUME_INTRO
    assert s0.pending.snr_db == 10
    s, specs = drive(cfg, ["W", "W"])
    assert s.volume_round == 2
    assert specs[-1].pre_prompt is PromptKey.VOLUME_UP
    assert specs[-1].kind is PresentationKind.VOLUME
    assert specs[-1].snr_db == 10
    s, specs = drive(cfg, ["W", "C"])
    assert s.phase is ProcedurePhase.PRACTICE
    assert specs[-1].pre_prompt is PromptKey.PRACTICE_INTRO
    assert specs[-1].feedback is True
    s, _ = drive(cfg, ["W", "W", "W", "W"])
    assert s.terminal is ProcedureTerminal.VOLUME_FAILED
    assert PROCEDURE.summarize(s).volume_failed is True


def test_U_ST_018_practice_rules() -> None:
    cfg = make_config()
    _, specs = drive(cfg, ["C", "C", "C", "W"])
    assert [p.snr_db for p in specs[2:5]] == [6, 2, -2]
    assert all(p.feedback for p in specs[2:5])
    s, specs = drive(cfg, ["C", "C", "C", "W", "W"])
    assert s.practice_round == 2
    assert specs[-1].pre_prompt is PromptKey.PRACTICE_RETRY
    assert specs[-1].snr_db == 6
    s, specs = drive(cfg, ["C", "C", "C", "C", "W"])
    assert s.phase is ProcedurePhase.SCORED
    p = specs[-1]
    assert p.kind is PresentationKind.SCORED
    assert p.snr_db == 0
    assert p.scored_index == 1
    assert p.pre_prompt is PromptKey.TEST_INTRO
    assert p.feedback is False
    s, _ = drive(cfg, ["C", "C", "W", "W", "W", "W", "W", "C"])
    assert s.terminal is ProcedureTerminal.PRACTICE_FAILED
    assert PROCEDURE.summarize(s).practice_failed is True


def test_U_ST_019_protocol_errors() -> None:
    s = PROCEDURE.start(scored_cfg())
    assert s.pending is not None
    good = result_for(s.pending, "C")
    reask = ScoringResult.model_validate(
        {
            **good.model_dump(),
            "action": ScoreAction.REASK,
            "reask_prompt": PromptKey.REASK_THREE,
            "triplet_correct": False,
            "status": AnswerStatus.NO_DIGITS,
        }
    )
    other = ScoringResult.model_validate({**good.model_dump(), "presentation_id": 99})
    for bad in (reask, other):
        with pytest.raises(ProcedureProtocolError) as ei:
            PROCEDURE.record(s, bad)
        assert ei.value.code is ErrorCode.PROCEDURE_PROTOCOL
    done = run_scored(["C"] * 24)
    with pytest.raises(ProcedureProtocolError) as ei:
        PROCEDURE.record(done, good)
    assert ei.value.code is ErrorCode.PROCEDURE_PROTOCOL


def test_U_ST_020_skip_volume_and_practice_and_condition() -> None:
    s = PROCEDURE.start(scored_cfg())
    p = s.pending
    assert p is not None
    assert s.phase is ProcedurePhase.SCORED
    assert p.kind is PresentationKind.SCORED
    assert p.scored_index == 1
    assert p.snr_db == 0
    assert p.pre_prompt is PromptKey.TEST_INTRO
    _, specs = drive(scored_cfg(condition=Condition.N3), ["NR"] * 3 + ["C", "CM", "W"] + ["C"] * 20)
    assert specs
    assert all(sp.condition is Condition.N3 for sp in specs)


def test_U_ST_021_partial_session_summary() -> None:
    s = run_scored(list(EX_A[:10]))
    r = PROCEDURE.summarize(s)
    assert r.complete is False
    assert r.srt_db is None
    assert r.reversals is None
    assert r.virtual_25_db is None
    assert len(r.track_db) == 11
    assert r.n_scored == 10
    r0 = PROCEDURE.summarize(PROCEDURE.start(make_config()))
    assert r0.track_db == ()
    assert r0.mode is Mode.KEYPAD
    assert r0.n_scored == 0


def _entry(i: int, snr: float, correct: bool, valid: bool = True) -> ScoredEntry:
    return ScoredEntry(
        scored_index=i,
        presentation_id=i,
        snr_db=snr,
        correct=correct,
        valid=valid,
        no_response=False,
        status=AnswerStatus.OK if valid else AnswerStatus.CAPTURE_MISS,
        mode=Mode.KEYPAD,
    )


def _with_entries(entries: list[ScoredEntry]) -> ProcedureState:
    s = PROCEDURE.start(scored_cfg())
    return ProcedureState.model_validate({**s.model_dump(), "scored": entries})


def test_U_ST_022_censor() -> None:
    base = [_entry(i, -6, i % 2 == 0) for i in range(1, 13)]

    def censor(changes: dict[int, tuple[float, bool]]) -> Censor:
        es = [
            _entry(e.scored_index, *changes[e.scored_index]) if e.scored_index in changes else e
            for e in base
        ]
        return PROCEDURE.summarize(_with_entries(es)).censor

    assert censor({7: (10, False)}) is Censor.HIGH
    assert censor({12: (-20, True)}) is Censor.LOW
    assert censor({7: (10, False), 12: (-20, True)}) is Censor.HIGH
    assert censor({3: (10, False)}) is Censor.NONE
    assert censor({}) is Censor.NONE


def test_U_ST_023_attempt2_capture_miss_flags() -> None:
    s, specs = drive(scored_cfg(), ["C", "CM", "CM"])
    e = s.scored[-1]
    assert e.valid is False
    assert e.correct is False
    assert e.status is AnswerStatus.CAPTURE_MISS
    assert e.snr_db == -4
    assert s.current_snr_db == -2  # first wrong: +2
    assert specs[2].attempt == 2
    assert specs[2].pre_prompt is None
    assert PROCEDURE.summarize(s).flags == 1
    s, _ = drive(scored_cfg(), ["CM", "CM"] * 4)
    assert PROCEDURE.summarize(s).flags == 4


def test_U_ST_024_mixed_modes() -> None:
    s, _ = drive(scored_cfg(), ["C", "W", "C"], modes=[Mode.VOICE, Mode.KEYPAD, Mode.VOICE])
    assert PROCEDURE.summarize(s).mode is Mode.MIXED
    s, _ = drive(scored_cfg(), ["C", "W"], modes=[Mode.VOICE, Mode.VOICE])
    assert PROCEDURE.summarize(s).mode is Mode.VOICE


def test_U_ST_025_seeds_and_permutation() -> None:
    scope = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b"
    cfg = make_config(scope, digit_set=(1, 2, 3, 4, 5, 6, 7, 8), skip_volume_and_practice=True)
    s, specs = drive(cfg, ["C"] * 7)
    assert specs[6].presentation_id == 7
    assert specs[6].noise_seed == derive_seed(scope, "noise:7")
    expected = np.random.Generator(np.random.PCG64(derive_seed(scope, "triplets"))).permutation(120)
    assert s.permutation == tuple(int(i) for i in expected)
    assert s.permutation[:5] == PINNED_PERM_5
    assert [p.pool_index for p in specs] == list(s.permutation[:8])


PINNED_PERM_5 = (23, 107, 2, 117, 64)  # pinned (U-ST-025)


def test_U_ST_026_record_perf() -> None:
    cfg = scored_cfg()
    times: list[float] = []
    n = 0
    while n < 10_000:
        s = PROCEDURE.start(cfg)
        k = 0
        while s.pending is not None and n < 10_000:
            r = result_for(s.pending, EX_A[k % 24])
            t0 = time.perf_counter()
            s = PROCEDURE.record(s, r)
            times.append(time.perf_counter() - t0)
            k += 1
            n += 1
    arr = np.array(times) * 1000
    assert arr.mean() <= 0.3, arr.mean()
    assert np.percentile(arr, 99) <= 1.0


def test_start_rejects_mismatched_design() -> None:
    cfg = make_config()
    bad = cfg.model_copy(
        update={"design": make_config(digit_set=(1, 2, 3, 4, 5, 6, 7, 8, 9)).design}
    )
    with pytest.raises(ValueError, match="design"):
        PROCEDURE.start(bad)


def test_pure_helpers() -> None:
    assert _next_snr(0, True, False) == -4
    assert _next_snr(0, False, True) == 2
    assert _next_snr(-18, True, False) == -20
    assert _next_snr(9, False, False) == 10
    assert count_reversals([0, -2, 2, 2, -2]) == 2
    es = [
        _entry(1, 10, False),
        _entry(2, 10, False, valid=False),
        _entry(3, 10, False),
        _entry(4, 8, False),
    ]
    assert max_consec_valid_wrong_at_ceiling(es) == 2


def test_R05_attempt1_capture_miss_changes_only_pending() -> None:
    s0 = PROCEDURE.start(scored_cfg())
    assert s0.pending is not None
    s1 = PROCEDURE.record(s0, result_for(s0.pending, "CM"))
    assert s1.pending is not None
    assert s1.pending.attempt == 2
    assert s1.pending.scored_index == 1
    assert s1.pending.snr_db == 0
    assert s1.cursor == s0.cursor + 1
    assert s1.scored == ()
    assert result_for(s1.pending, "C").source is AnswerSource.DTMF
