"""Drive the real staircase with symbolic answers (Person A spec §6.5).

Symbols: ``C`` (OK, correct), ``W`` (OK, wrong), ``NR`` (NO_RESPONSE), ``CM`` (CAPTURE_MISS).
"""

from __future__ import annotations

from collections.abc import Iterable

from sevi.contracts.enums import AnswerSource, AnswerStatus, Lang, Mode, ScoreAction
from sevi.contracts.languages import DIGIT_SETS
from sevi.contracts.procedure import ProcedureConfig, ProcedureState
from sevi.contracts.trial import PresentationSpec, ScoringResult
from sevi.core.staircase import PROCEDURE, design

TA_DIGITS = DIGIT_SETS[Lang.TA]


def make_config(seed_scope: str = "test-scope", **kw: object) -> ProcedureConfig:
    digits = tuple(kw.pop("digit_set", TA_DIGITS))  # type: ignore[call-overload]
    return ProcedureConfig.model_validate(
        {"seed_scope": seed_scope, "design": design(digits), "digit_set": digits, **kw}
    )


def result_for(spec: PresentationSpec, sym: str, mode: Mode = Mode.KEYPAD) -> ScoringResult:
    status = {
        "C": AnswerStatus.OK,
        "W": AnswerStatus.OK,
        "NR": AnswerStatus.NO_RESPONSE,
        "CM": AnswerStatus.CAPTURE_MISS,
    }[sym]
    correct = sym == "C"
    if correct:
        rec = spec.triplet
    elif sym == "W":
        rec = tuple(9 if d != 9 else 0 for d in spec.triplet)
    else:
        rec = ()
    return ScoringResult(
        presentation_id=spec.presentation_id,
        mode=mode,  # type: ignore[arg-type]
        status=status,
        action=ScoreAction.ACCEPT,
        reask_prompt=None,
        source=AnswerSource.DTMF if rec else AnswerSource.NONE,
        recognised=rec,
        triplet_correct=correct,
        position_correct=(correct, correct, correct),
        valid=sym != "CM",
        no_response=sym == "NR",
        number_form=False,
    )


def drive(
    cfg_or_state: ProcedureConfig | ProcedureState,
    answers: Iterable[str],
    modes: Iterable[Mode] | None = None,
) -> tuple[ProcedureState, list[PresentationSpec]]:
    """Feed answers until terminal or the list ends. Returns final state and every spec shown."""
    s = PROCEDURE.start(cfg_or_state) if isinstance(cfg_or_state, ProcedureConfig) else cfg_or_state
    specs: list[PresentationSpec] = [s.pending] if s.pending is not None else []
    mode_it = iter(modes) if modes is not None else None
    for sym in answers:
        if s.terminal is not None or s.pending is None:
            break
        mode = next(mode_it) if mode_it is not None else Mode.KEYPAD
        s = PROCEDURE.record(s, result_for(s.pending, sym, mode))
        if s.pending is not None:
            specs.append(s.pending)
    return s, specs
