"""Property tests for core/staircase.py (Person A spec §6.5, P-ST-01..10, P-ST-20)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from sevi.contracts.enums import (
    Condition,
    PresentationKind,
    ProcedurePhase,
    ProcedureTerminal,
)
from sevi.contracts.procedure import ProcedureConfig
from sevi.contracts.rng import derive_seed
from sevi.core.staircase import PROCEDURE, _track
from sevi.core.triplets import design
from sevi.fakes.scenarios.staircase_driver import drive, make_config, result_for

answers_st = st.lists(st.sampled_from(["C", "W", "NR", "CM"]), max_size=90)
cond_st = st.sampled_from(list(Condition))


def _cfg(skip: bool, cond: Condition = Condition.NORMAL, scope: str = "prop") -> ProcedureConfig:
    if cond is not Condition.NORMAL:
        skip = True
    return make_config(scope, skip_volume_and_practice=skip, condition=cond)


def _states(cfg: ProcedureConfig, answers: list[str]):  # type: ignore[no-untyped-def]
    s = PROCEDURE.start(cfg)
    out = [s]
    for a in answers:
        if s.pending is None:
            break
        s = PROCEDURE.record(s, result_for(s.pending, a))
        out.append(s)
    return out


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_01_snr_bounds(answers: list[str], skip: bool) -> None:
    for s in _states(_cfg(skip), answers):
        assert -20 <= s.current_snr_db <= 10
        if s.pending is not None:
            assert -20 <= s.pending.snr_db <= 10


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_02_direction_and_step_sizes(answers: list[str], skip: bool) -> None:
    s, _ = drive(_cfg(skip), answers)
    fws = False
    for a, b in zip(s.scored, s.scored[1:], strict=False):
        fws = fws or not a.correct
        if a.correct:
            assert b.snr_db <= a.snr_db
        else:
            assert b.snr_db >= a.snr_db
        step = abs(b.snr_db - a.snr_db)
        assert step in (0.0, 2.0 if fws else 4.0) or b.snr_db in (-20, 10)


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_03_scored_count(answers: list[str], skip: bool) -> None:
    states = _states(_cfg(skip), answers)
    for prev, s in itertools.pairwise(states):
        assert len(s.scored) <= 24
        assert (s.terminal is ProcedureTerminal.COMPLETE) == (len(s.scored) == 24)
        if prev.pending is not None and s.pending is not None and s.pending.attempt == 2:
            assert len(s.scored) == len(prev.scored)


@pytest.mark.slow
@settings(max_examples=20, deadline=None)
@given(st.floats(-14, -2), st.floats(0.10, 0.25), st.integers(0, 2**32 - 1))
def test_P_ST_04_logistic_listener_converges(theta: float, slope: float, seed: int) -> None:
    rng = np.random.Generator(np.random.PCG64(seed))
    errs = []
    for run in range(200):
        s = PROCEDURE.start(_cfg(True, scope=f"p4:{seed}:{run}"))
        while s.pending is not None:
            p = 1 / (1 + math.exp(-4 * slope * (s.pending.snr_db - theta)))
            s = PROCEDURE.record(s, result_for(s.pending, "C" if rng.random() < p else "W"))
        srt = PROCEDURE.summarize(s).srt_db
        assert srt is not None
        errs.append(abs(srt - theta))
    assert float(np.mean(errs)) <= 1.5


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_05_determinism(answers: list[str], skip: bool) -> None:
    a, sa = drive(_cfg(skip), answers)
    b, sb = drive(_cfg(skip), answers)
    assert sa == sb
    assert PROCEDURE.summarize(a) == PROCEDURE.summarize(b)


@settings(deadline=None)
@given(st.lists(st.booleans(), max_size=24), st.sampled_from([-4.0, -2.0, 2.0, 4.0]))
def test_P_ST_06_shift_equivariance(seq: list[bool], c: float) -> None:
    t0 = _track(seq, 0.0)
    tc = _track(seq, c)
    if all(-20 < v < 10 for v in (*t0, *tc)):
        assert tc == tuple(v + c for v in t0)
        if len(seq) == 24:
            assert np.mean(tc[4:]) == pytest.approx(np.mean(t0[4:]) + c)


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_07_attempt1_capture_miss_only_changes_pending(answers: list[str], skip: bool) -> None:
    s, _ = drive(_cfg(skip), answers)
    if s.pending is None or s.pending.attempt != 1:
        return
    p = s.pending
    s2 = PROCEDURE.record(s, result_for(p, "CM"))
    q = s2.pending
    assert q is not None
    assert (q.kind, q.snr_db, q.scored_index, q.attempt, q.pre_prompt) == (
        p.kind,
        p.snr_db,
        p.scored_index,
        2,
        None,
    )
    assert q.pool_index == s.permutation[s.cursor]
    assert s2.cursor == s.cursor + 1
    assert s2.next_presentation_id == s.next_presentation_id + 1
    ignore = {"pending", "cursor", "next_presentation_id"}
    assert s2.model_dump(exclude=ignore) == s.model_dump(exclude=ignore)


@settings(max_examples=200, deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_08_cursor_and_unique_pool(answers: list[str], skip: bool) -> None:
    s, specs = drive(_cfg(skip), answers)
    assert s.cursor <= 76
    idx = [p.pool_index for p in specs]
    assert len(idx) == len(set(idx))


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_09_loud_invariants(answers: list[str], skip: bool) -> None:
    for s in _states(_cfg(skip), answers):
        if s.loud_wrong_total >= 2:
            assert s.terminal is not None
        if s.phase is ProcedurePhase.LOUD:
            assert s.no_response_run == 0


@settings(deadline=None)
@given(answers_st, st.booleans())
def test_P_ST_10_summarize_total(answers: list[str], skip: bool) -> None:
    for s in _states(_cfg(skip), answers):
        r = PROCEDURE.summarize(s)
        assert (r.srt_db is not None) == (
            r.n_scored == 24 and s.terminal is ProcedureTerminal.COMPLETE
        )


@settings(deadline=None)
@given(answers_st, cond_st)
def test_P_ST_20_condition_propagates(answers: list[str], cond: Condition) -> None:
    _, specs = drive(_cfg(True, cond), answers)
    assert all(p.condition is cond for p in specs)
    assert all(p.feedback == (p.kind is PresentationKind.PRACTICE) for p in specs)


def test_P_ST_20_condition_requires_skip() -> None:
    d = (1, 2, 3, 4, 5, 6, 7, 8)
    with pytest.raises(ValidationError):
        ProcedureConfig(seed_scope="x", design=design(d), digit_set=d, condition=Condition.N3)


def test_seed_scope_changes_permutation() -> None:
    a = PROCEDURE.start(_cfg(True, scope="a")).permutation
    b = PROCEDURE.start(_cfg(True, scope="b")).permutation
    assert a != b
    assert sorted(a) == sorted(b) == list(range(120))
    assert derive_seed("a", "triplets") != derive_seed("b", "triplets")
