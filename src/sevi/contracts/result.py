"""Norm cells, result input and decision (contracts §4.7; truth table §4.7.1).

Reduced scope: no norms are produced in this build (human panel and validation are cut), so
``ResultInput.norms`` is ``None`` in practice and banded languages fall to T11 NO_CELL. The norm
models stay because the truth table (T11-T13) is defined over them.
"""

from typing import Literal

from pydantic import Field, model_validator

from .enums import (
    Band,
    Channel,
    CutoffVersion,
    EndReason,
    FriendsVariant,
    Lang,
    Mode,
    NoBandReason,
    ParticipantType,
    ReferReason,
    Surface,
    Transducer,
    TruthRule,
    WordingClass,
)
from .evals import ResultFileMeta
from .procedure import SrtResult
from .types import SessionId, SeviModel, Sha256Hex, UtcDatetime


class NormCellKey(SeviModel):
    lang: Lang
    channel: Channel
    mode: Literal[Mode.VOICE, Mode.KEYPAD]
    transducer: Literal[Transducer.HANDSET, Transducer.EARPHONES]
    # string form "ta/browser/keypad/earphones"


class NormCell(SeviModel):
    key: NormCellKey
    n: int = Field(ge=0)
    mean_db: float | None
    sd_db: float | None  # sample SD, ddof=1
    nh95_db: float | None  # mean + 1.645 * sd; None if n < 6
    cutoff_version: CutoffVersion | None  # None if n < 6
    ages: str = Field(pattern=r"^\d{2}-\d{2}$")  # e.g. "18-35"
    bank_hash: Sha256Hex
    computed_at: UtcDatetime


class NormsFile(SeviModel):
    meta: ResultFileMeta
    cells: tuple[NormCell, ...]

    @model_validator(mode="after")
    def _no_experimental(self) -> "NormsFile":
        # Scope v3 (b): English never has human norms.
        if any(c.key.lang is Lang.EN for c in self.cells):
            raise ValueError("norms.json must not contain en cells (Scope v3 (b))")
        return self


class ResultInput(SeviModel):
    session_id: SessionId
    lang: Lang
    channel: Channel
    mode: Mode
    # R-03: reference cell for W_NOBAND_RANGE when MIXED
    mode_requested: Literal[Mode.VOICE, Mode.KEYPAD]
    transducer: Transducer
    participant_type: ParticipantType
    end_reason: EndReason
    srt: SrtResult | None  # None if the procedure never started
    friends: FriendsVariant
    experimental_language: bool
    bank_hash: Sha256Hex
    norms: NormsFile | None
    hhie_total: int | None = Field(None, ge=0, le=40)  # Could; None when not administered


class ResultDecision(SeviModel):
    rule: TruthRule
    band: Band
    refer_reason: ReferReason | None
    no_band_reason: NoBandReason | None
    wording_class: WordingClass
    surfaces: tuple[Surface, ...]
    norm_cell: NormCellKey | None
    cutoff_version: CutoffVersion | None
    nh95_db: float | None
    n_norm: int | None
    srt_db: float | None
    # U+2212 minus sign, e.g. "\u22128.5"
    srt_display: str | None = Field(None, pattern=r"^[+−]?\d+\.(0|5)$")  # noqa: RUF001
    ages: str | None = Field(None, pattern=r"^\d{2}-\d{2}$")  # R-02: from the matched NormCell
