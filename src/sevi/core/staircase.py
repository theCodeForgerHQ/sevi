"""Adaptive SRT procedure (contracts §4.3, §4.6, §4.6.1; Person A spec §6).

Pure and functional: every call returns a new ``ProcedureState``. The only SNR rule is
:func:`_next_snr`. ``PROCEDURE`` is the stateless singleton B's startup uses.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Final

import numpy as np

from sevi.contracts.constants import (
    CAL_SUBSET_SIZE,
    CEILING_CONSECUTIVE,
    CEILING_MAX_CORRECT,
    CEILING_MIN_VALID,
    LOUD_N,
    LOUD_SNR_DB,
    LOUD_WRONG_REFER,
    MAX_LOUD_CHECKS,
    N_SCORED,
    NO_RESPONSE_RUN,
    OFFSET_PAIRS,
    POOL_SIZE,
    PRACTICE_PASS_MIN,
    PRACTICE_ROUNDS,
    PRACTICE_SNRS_DB,
    SNR_MAX_DB,
    SNR_MIN_DB,
    SRT_FIRST_TRIAL,
    START_SNR_DB,
    STEP_DB,
    STEP_INITIAL_DB,
    VOLUME_N,
    VOLUME_PASS_MIN,
    VOLUME_ROUNDS,
    VOLUME_SNR_DB,
)
from sevi.contracts.enums import (
    AnswerStatus,
    Censor,
    Mode,
    PresentationKind,
    ProcedurePhase,
    ProcedureTerminal,
    PromptKey,
    ScoreAction,
)
from sevi.contracts.errors import ErrorCode, ProcedureProtocolError
from sevi.contracts.procedure import ProcedureConfig, ProcedureState, ScoredEntry, SrtResult
from sevi.contracts.rng import derive_seed
from sevi.contracts.stimulus import TripletDesign
from sevi.contracts.trial import PresentationSpec, ScoringResult

__all__ = [
    "PROCEDURE",
    "Staircase",
    "build_pool",
    "count_reversals",
    "design",
    "max_consec_valid_wrong_at_ceiling",
]

Triplet3 = tuple[int, int, int]


# ---------- triplet pool (contracts §4.3 formula) ----------
# NOTE: core/triplets.py (issue #4) owns the canonical build_pool/design. These private copies
# follow the same normative formula so the staircase does not block on that PR; swap to the
# triplets import once it lands on main.


def build_pool(digit_set: Sequence[int]) -> tuple[Triplet3, ...]:
    return _pool(tuple(digit_set))


@lru_cache(maxsize=8)
def _pool(digit_set: tuple[int, ...]) -> tuple[Triplet3, ...]:
    n = len(digit_set)
    out: list[Triplet3] = []
    for d1, d2 in OFFSET_PAIRS[n]:
        for a in range(n):
            out.append((digit_set[a], digit_set[(a + d1) % n], digit_set[(a + d2) % n]))
    return tuple(out)


def design(digit_set: Sequence[int]) -> TripletDesign:
    n = len(digit_set)
    return TripletDesign.model_validate(
        {
            "n_digits": n,
            "offset_pairs": OFFSET_PAIRS[n],
            "pool_size": POOL_SIZE[n],
            "calibration_subset_size": CAL_SUBSET_SIZE[n],
        }
    )


# ---------- pure helpers ----------


def _next_snr(snr: float, correct: bool, first_wrong_seen: bool) -> float:
    """The only SNR rule: clamp(snr -/+ step, -20, +10), step 2 once any wrong seen, else 4."""
    step = STEP_DB if first_wrong_seen else STEP_INITIAL_DB
    nxt = snr - step if correct else snr + step
    return float(min(SNR_MAX_DB, max(SNR_MIN_DB, nxt)))


def _track(correct: Sequence[bool], start_db: float = START_SNR_DB) -> tuple[float, ...]:
    """SNRs presented for a correctness sequence, then the virtual next value (test helper)."""
    snr = start_db
    fws = False
    out = [snr]
    for c in correct:
        if not c:
            fws = True
        snr = _next_snr(snr, c, fws)
        out.append(snr)
    return tuple(out)


def max_consec_valid_wrong_at_ceiling(entries: Sequence[ScoredEntry]) -> int:
    best = run = 0
    for e in entries:
        if not e.valid:
            continue
        if e.snr_db == SNR_MAX_DB and not e.correct:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def count_reversals(track_with_virtual: Sequence[float]) -> int:
    """Sign changes between consecutive non-zero moves (R-06); caller passes the slice."""
    moves = [b - a for a, b in itertools.pairwise(track_with_virtual) if b != a]
    return sum(1 for m0, m1 in itertools.pairwise(moves) if (m0 > 0) != (m1 > 0))


# ---------- the procedure ----------


def _evolve(s: ProcedureState, **changes: Any) -> ProcedureState:
    return ProcedureState.model_validate({**s.model_dump(), **changes})


def _protocol(msg: str) -> ProcedureProtocolError:
    return ProcedureProtocolError(ErrorCode.PROCEDURE_PROTOCOL, msg)


class Staircase:
    """Implements ``contracts.ports.AdaptiveProcedure``."""

    def start(self, config: ProcedureConfig) -> ProcedureState:
        if design(config.digit_set) != config.design:
            raise ValueError("config.design does not match the digit_set design")
        n_pool = len(build_pool(config.digit_set))
        perm = np.random.Generator(
            np.random.PCG64(derive_seed(config.seed_scope, "triplets"))
        ).permutation(n_pool)
        skip = config.skip_volume_and_practice
        s = ProcedureState(
            config=config,
            permutation=tuple(int(i) for i in perm),
            cursor=0,
            next_presentation_id=1,
            phase=ProcedurePhase.SCORED if skip else ProcedurePhase.VOLUME,
            volume_round=1,
            practice_round=1,
            current_snr_db=START_SNR_DB,
            pending=None,
        )
        if skip:
            return self._new(s, PresentationKind.SCORED, START_SNR_DB, PromptKey.TEST_INTRO, 1)
        return self._new(s, PresentationKind.VOLUME, VOLUME_SNR_DB, PromptKey.VOLUME_INTRO)

    @staticmethod
    def _new(
        s: ProcedureState,
        kind: PresentationKind,
        snr: float,
        pre_prompt: PromptKey | None,
        scored_index: int | None = None,
        attempt: int = 1,
        **changes: Any,
    ) -> ProcedureState:
        cfg = s.config
        idx = s.permutation[s.cursor]
        pid = s.next_presentation_id
        spec = PresentationSpec(
            presentation_id=pid,
            kind=kind,
            scored_index=scored_index,
            attempt=1 if attempt == 1 else 2,
            snr_db=snr,
            triplet=build_pool(cfg.digit_set)[idx],
            pool_index=idx,
            noise_seed=derive_seed(cfg.seed_scope, f"noise:{pid}"),
            pre_prompt=pre_prompt,
            feedback=kind is PresentationKind.PRACTICE,
            condition=cfg.condition,
        )
        return _evolve(
            s,
            cursor=s.cursor + 1,
            next_presentation_id=pid + 1,
            pending=spec.model_dump(),
            **changes,
        )

    @staticmethod
    def _end(s: ProcedureState, terminal: ProcedureTerminal, **changes: Any) -> ProcedureState:
        return _evolve(s, terminal=terminal, phase=ProcedurePhase.DONE, pending=None, **changes)

    def record(self, state: ProcedureState, result: ScoringResult) -> ProcedureState:
        s = state
        if s.terminal is not None:
            raise _protocol("procedure already terminal")
        if result.action is not ScoreAction.ACCEPT:
            raise _protocol(f"result.action must be ACCEPT, got {result.action}")
        p = s.pending
        if p is None or result.presentation_id != p.presentation_id:
            raise _protocol("result.presentation_id does not match pending")

        # R-05: attempt-1 CAPTURE_MISS changes nothing but `pending`.
        if result.status is AnswerStatus.CAPTURE_MISS and p.attempt == 1:
            return self._new(s, p.kind, p.snr_db, None, p.scored_index, attempt=2)

        correct = result.triplet_correct
        K = PresentationKind
        if p.kind is K.VOLUME:
            res = (*s.volume_results, correct)
            if len(res) < VOLUME_N:
                return self._new(s, K.VOLUME, VOLUME_SNR_DB, None, volume_results=res)
            if sum(res) >= VOLUME_PASS_MIN:
                return self._new(
                    s,
                    K.PRACTICE,
                    PRACTICE_SNRS_DB[0],
                    PromptKey.PRACTICE_INTRO,
                    volume_results=res,
                    phase=ProcedurePhase.PRACTICE,
                    practice_results=(),
                )
            if s.volume_round < VOLUME_ROUNDS:
                return self._new(
                    s,
                    K.VOLUME,
                    VOLUME_SNR_DB,
                    PromptKey.VOLUME_UP,
                    volume_round=s.volume_round + 1,
                    volume_results=(),
                )
            return self._end(s, ProcedureTerminal.VOLUME_FAILED, volume_results=res)

        if p.kind is K.PRACTICE:
            res = (*s.practice_results, correct)
            n = len(res)
            if n < len(PRACTICE_SNRS_DB):
                return self._new(s, K.PRACTICE, PRACTICE_SNRS_DB[n], None, practice_results=res)
            if sum(res) >= PRACTICE_PASS_MIN:
                return self._new(
                    s,
                    K.SCORED,
                    START_SNR_DB,
                    PromptKey.TEST_INTRO,
                    1,
                    practice_results=res,
                    phase=ProcedurePhase.SCORED,
                    current_snr_db=START_SNR_DB,
                )
            if s.practice_round < PRACTICE_ROUNDS:
                return self._new(
                    s,
                    K.PRACTICE,
                    PRACTICE_SNRS_DB[0],
                    PromptKey.PRACTICE_RETRY,
                    practice_round=s.practice_round + 1,
                    practice_results=(),
                )
            return self._end(s, ProcedureTerminal.PRACTICE_FAILED, practice_results=res)

        if p.kind is K.SCORED:
            assert p.scored_index is not None
            entry = ScoredEntry(
                scored_index=p.scored_index,
                presentation_id=p.presentation_id,
                snr_db=p.snr_db,
                correct=correct,
                valid=result.valid,
                no_response=result.no_response,
                status=result.status,
                mode=result.mode,
            )
            scored = (*s.scored, entry)
            fws = s.first_wrong_seen or not correct
            cur = _next_snr(p.snr_db, correct, fws)
            nrr = s.no_response_run + 1 if result.no_response else 0
            upd: dict[str, Any] = {
                "scored": scored,
                "first_wrong_seen": fws,
                "current_snr_db": cur,
                "no_response_run": nrr,
            }
            if max_consec_valid_wrong_at_ceiling(scored) >= CEILING_CONSECUTIVE:
                return self._end(s, ProcedureTerminal.CEILING_STOP, **upd)
            if len(scored) == N_SCORED:
                return self._end(s, ProcedureTerminal.COMPLETE, **upd)
            if nrr == NO_RESPONSE_RUN:
                if s.loud_checks_started < MAX_LOUD_CHECKS:
                    upd.update(
                        phase=ProcedurePhase.LOUD,
                        loud_checks_started=s.loud_checks_started + 1,
                        loud_results=(),
                        no_response_run=0,
                    )
                    return self._new(s, K.LOUD, LOUD_SNR_DB, PromptKey.LOUD_CHECK, **upd)
                return self._end(s, ProcedureTerminal.NO_RESPONSE_END, **upd)
            return self._new(s, K.SCORED, cur, None, len(scored) + 1, **upd)

        # LOUD: not scored, no SNR change.
        res = (*s.loud_results, correct)
        lwt = s.loud_wrong_total + (0 if correct else 1)
        if lwt >= LOUD_WRONG_REFER:
            return self._end(
                s, ProcedureTerminal.LOUD_FAILED, loud_results=res, loud_wrong_total=lwt
            )
        if len(res) == LOUD_N:
            return self._new(
                s,
                K.SCORED,
                s.current_snr_db,
                None,
                len(s.scored) + 1,
                loud_results=res,
                loud_wrong_total=lwt,
                phase=ProcedurePhase.SCORED,
            )
        return self._new(s, K.LOUD, LOUD_SNR_DB, None, loud_results=res, loud_wrong_total=lwt)

    def summarize(self, state: ProcedureState) -> SrtResult:
        sc = state.scored
        n = len(sc)
        complete = n == N_SCORED
        track = (*(e.snr_db for e in sc), state.current_snr_db) if n else ()
        first = SRT_FIRST_TRIAL - 1
        srt = (
            float(np.mean(track[first:]))
            if complete and state.terminal is not ProcedureTerminal.CEILING_STOP
            else None
        )
        tail = [e for e in sc if e.scored_index >= SRT_FIRST_TRIAL]
        if any(e.snr_db == SNR_MAX_DB and not e.correct for e in tail):
            censor = Censor.HIGH
        elif any(e.snr_db == SNR_MIN_DB and e.correct for e in tail):
            censor = Censor.LOW
        else:
            censor = Censor.NONE
        at_ceil = [e for e in sc if e.valid and e.snr_db == SNR_MAX_DB]
        vac = len(at_ceil)
        cac = sum(1 for e in at_ceil if e.correct)
        mc = max_consec_valid_wrong_at_ceiling(sc)
        modes = {e.mode for e in sc}
        mode = Mode.KEYPAD if not modes else (Mode.MIXED if len(modes) > 1 else modes.pop())
        return SrtResult(
            n_scored=n,
            complete=complete,
            srt_db=srt,
            track_db=track,
            virtual_25_db=state.current_snr_db if complete else None,
            reversals=count_reversals(track[first:]) if complete else None,
            censor=censor,
            flags=sum(1 for e in sc if not e.valid),
            valid_at_ceiling=vac,
            correct_at_ceiling=cac,
            max_consec_valid_wrong_at_ceiling=mc,
            ceiling_fail=(vac >= CEILING_MIN_VALID and cac <= CEILING_MAX_CORRECT)
            or mc >= CEILING_CONSECUTIVE,
            mode=mode,
            terminal=state.terminal,
            loud_wrong_total=state.loud_wrong_total,
            volume_failed=state.terminal is ProcedureTerminal.VOLUME_FAILED,
            practice_failed=state.terminal is ProcedureTerminal.PRACTICE_FAILED,
        )


PROCEDURE: Final = Staircase()
