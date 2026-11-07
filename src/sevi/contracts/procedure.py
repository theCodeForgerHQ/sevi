"""Adaptive-procedure config/state and SRT result (contracts §4.6; rules §4.6.1)."""

from pydantic import Field, model_validator

from .enums import AnswerStatus, Censor, Condition, Mode, ProcedurePhase, ProcedureTerminal
from .stimulus import TripletDesign
from .trial import PresentationSpec
from .types import Digit, SeviModel


class ProcedureConfig(SeviModel):
    seed_scope: str = Field(min_length=1, max_length=64)  # session_id (live), run id (sim)
    design: TripletDesign
    digit_set: tuple[Digit, ...] = Field(min_length=8, max_length=9)
    # True for simulated-loss station runs and sim/calibration tooling; public sessions never
    skip_volume_and_practice: bool = False
    condition: Condition = Condition.NORMAL  # STUDY station sessions only; public always NORMAL

    @model_validator(mode="after")
    def _condition(self) -> "ProcedureConfig":
        # A review B-1: a simulated-loss run never runs volume/practice under the condition.
        if self.condition is not Condition.NORMAL and not self.skip_volume_and_practice:
            raise ValueError("condition != NORMAL requires skip_volume_and_practice=True")
        return self


class ScoredEntry(SeviModel):
    scored_index: int = Field(ge=1, le=24)
    presentation_id: int = Field(ge=1)
    snr_db: float
    correct: bool
    valid: bool  # False iff system-flagged (second CAPTURE_MISS)
    no_response: bool
    status: AnswerStatus
    mode: Mode


class ProcedureState(SeviModel):
    config: ProcedureConfig
    # rng(derive_seed(seed_scope, "triplets")).permutation(pool_size)
    permutation: tuple[int, ...]
    cursor: int = Field(ge=0)  # next permutation index to consume
    next_presentation_id: int = Field(ge=1)
    phase: ProcedurePhase
    volume_round: int = Field(ge=1, le=2)
    volume_results: tuple[bool, ...] = ()  # this round
    practice_round: int = Field(ge=1, le=2)
    practice_results: tuple[bool, ...] = ()  # this round
    scored: tuple[ScoredEntry, ...] = ()
    current_snr_db: float = 0.0
    first_wrong_seen: bool = False
    no_response_run: int = Field(0, ge=0)
    loud_checks_started: int = Field(0, ge=0, le=2)
    loud_results: tuple[bool, ...] = ()  # current loud check
    loud_wrong_total: int = Field(0, ge=0)
    pending: PresentationSpec | None
    terminal: ProcedureTerminal | None = None


class SrtResult(SeviModel):
    n_scored: int = Field(ge=0, le=24)
    complete: bool  # n_scored == 24
    srt_db: float | None  # None unless complete (and not CEILING_STOP)
    track_db: tuple[float, ...]  # SNRs of scored trials 1..n, then the virtual next value
    virtual_25_db: float | None
    # R-06: moves=[t[k+1]-t[k] for k in 5..24] (1-based, t[25]=virtual); drop zeros;
    # count sign changes
    reversals: int | None
    censor: Censor  # HIGH: a wrong at +10 in trials 5-24; LOW: a correct at -20 in 5-24
    flags: int = Field(ge=0)  # scored trials with valid == False
    valid_at_ceiling: int = Field(ge=0)
    correct_at_ceiling: int = Field(ge=0)
    max_consec_valid_wrong_at_ceiling: int = Field(ge=0)  # flagged trials removed first
    # (valid_at_ceiling >= 4 and correct_at_ceiling <= 1) or max_consec >= 6
    ceiling_fail: bool
    # MIXED if scored trials used both modes; KEYPAD when n_scored == 0
    # (use SessionRecord.mode_effective)
    mode: Mode
    terminal: ProcedureTerminal | None
    loud_wrong_total: int = Field(ge=0)
    volume_failed: bool
    practice_failed: bool
