"""Trial request, answer capture, scoring (contracts §4.5; scorer rules §4.5.1)."""

from typing import Literal

from pydantic import Field, model_validator

from .enums import (
    AnswerSource,
    AnswerStatus,
    Condition,
    Lang,
    Mode,
    PresentationKind,
    PromptKey,
    ScoreAction,
    SystemEvent,
    WindowCloseReason,
)
from .types import Digit, NonNegMs, SessionId, SeviModel, SnrDb, Triplet, UInt64


class PresentationSpec(SeviModel):
    """'Trial request': produced by AdaptiveProcedure, consumed by the orchestrator."""

    presentation_id: int = Field(ge=1)  # 1,2,3... per session; replays reuse it
    kind: PresentationKind
    scored_index: int | None = Field(None, ge=1, le=24)  # set iff kind == SCORED
    # 2 = re-presentation after CAPTURE_MISS (new triplet, same SNR)
    attempt: Literal[1, 2] = 1
    snr_db: SnrDb
    triplet: Triplet
    pool_index: int = Field(ge=0, le=125)
    noise_seed: UInt64  # derive_seed(seed_scope, f"noise:{presentation_id}")
    pre_prompt: PromptKey | None = None  # orchestrator plays this before the stimulus
    feedback: bool = False  # True only for PRACTICE
    condition: Condition = Condition.NORMAL

    @model_validator(mode="after")
    def _scored(self) -> "PresentationSpec":
        if (self.kind is PresentationKind.SCORED) != (self.scored_index is not None):
            raise ValueError("scored_index set iff kind == SCORED")
        if self.feedback and self.kind is not PresentationKind.PRACTICE:
            raise ValueError("feedback only for PRACTICE")
        return self


class SttSegment(SeviModel):
    text: str = Field(max_length=500)
    segment_id: str | None = None
    segment_index: int | None = Field(None, ge=1)  # R-19: Prisma segment_index, merge order
    received_ms: NonNegMs  # relative to window open (monotonic)
    audio_duration_ms: int | None = Field(None, ge=0)
    latency_ms: int | None = Field(None, ge=0)
    after_commit: bool


class DtmfPress(SeviModel):
    # R-21: "0", "*", "#" are controls, never answer digits
    key: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    at_ms: NonNegMs  # relative to window open


class AnswerEvidence(SeviModel):
    """Answer capture result: produced by B (capture + Prisma barrier), consumed by AnswerScorer."""

    presentation_id: int = Field(ge=1)
    lang: Lang
    mode: Literal[Mode.VOICE, Mode.KEYPAD]
    window_index: Literal[1, 2]  # 2 = the window after a REASK
    window_open_mono_ms: NonNegMs
    window_close_mono_ms: NonNegMs
    close_reason: WindowCloseReason
    segments: tuple[SttSegment, ...] = Field(default=(), max_length=20)
    dtmf: tuple[DtmfPress, ...] = Field(default=(), max_length=20)
    # >= SPEECH_MIN_MS frames above floor + SPEECH_DB_ABOVE_FLOOR in the window
    speech_present: bool
    speech_ms: int = Field(ge=0)
    early_speech: bool  # speech onset detected inside the gate (always False in keypad)
    echo_mode: bool  # session echo coupling >= ECHO_HIGH_DB
    system_events: tuple[SystemEvent, ...] = ()
    reask_used: bool
    replay_used: bool


class ScoringResult(SeviModel):
    presentation_id: int = Field(ge=1)
    mode: Literal[Mode.VOICE, Mode.KEYPAD]
    status: AnswerStatus
    action: ScoreAction
    # REASK_SAY_AGAIN (EARLY_ANSWER) | REASK_THREE (NO_DIGITS, AMBIGUOUS)
    reask_prompt: PromptKey | None
    source: AnswerSource
    # all digits parsed in order (AMBIGUOUS may have > 3)
    recognised: tuple[Digit, ...] = Field(max_length=12)
    triplet_correct: bool  # recognised == presented triplet, exactly, in order
    position_correct: tuple[bool, bool, bool]  # Levenshtein alignment (shared with calibration)
    valid: bool  # False iff status == CAPTURE_MISS
    no_response: bool  # status == NO_RESPONSE
    # a compound number (100-999) or a digit+tens grouping (§4.13 rule C2) was converted
    number_form: bool
    normalised_tokens: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def _consistent(self) -> "ScoringResult":
        if self.triplet_correct and self.status is not AnswerStatus.OK:
            raise ValueError("triplet_correct requires status OK")
        if (self.action is ScoreAction.REASK) != (self.reask_prompt is not None):
            raise ValueError("reask_prompt set iff action REASK")
        if self.valid != (self.status is not AnswerStatus.CAPTURE_MISS):
            raise ValueError("valid iff not CAPTURE_MISS")
        return self


class TrialRecord(SeviModel):
    """One presentation as stored (every kind, every attempt)."""

    session_id: SessionId
    spec: PresentationSpec
    result: ScoringResult  # the final ACCEPTed result
    system_flag: bool  # True iff attempt == 2 and result.status == CAPTURE_MISS
    reasks: int = Field(ge=0, le=1)
    replays: int = Field(ge=0, le=1)
    early_answer: bool
    echo_mode: bool
    rx_floor_dbfs: float | None
    speech_ms: int = Field(ge=0)
    t_send_mono_ms: NonNegMs
    gate_end_mono_ms: NonNegMs  # t_send + d_est + last_digit_offset + t_guard
    d_est_ms: NonNegMs
    t_guard_ms: int = Field(ge=0)
    last_digit_offset_ms: NonNegMs
    transcript: str | None = Field(
        None, max_length=1000
    )  # joined segment texts; deleted with session
    stale_segments: int = Field(ge=0)
    turn_ms: float | None = Field(None, ge=0)  # span turn.total ending at this first media
    level_limited: bool
