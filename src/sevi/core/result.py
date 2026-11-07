"""Result truth table (contracts §4.7.1; Person A spec §8.1-§8.2).

First match wins. Every condition that reads ``srt`` is False when ``srt is None`` (R-04), and
``classify`` never raises for a valid :class:`ResultInput`: if evaluation reaches T10b/T11 without
24 scored trials and an SRT, the T8 decision is returned (the "bug path", CCR-v3-20).
"""

from __future__ import annotations

import math
from typing import Final

from sevi.contracts.constants import (
    ABS_REFER_DB,
    FLAG_LIMIT,
    NORM_MIN_N,
    REFER_MARGIN_DB,
    REVERSALS_MIN,
    V0_BORDER_HALFWIDTH_DB,
)
from sevi.contracts.enums import (
    Band,
    Censor,
    Channel,
    CutoffVersion,
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
from sevi.contracts.ports import ResultClassifier, WordingText
from sevi.contracts.procedure import SrtResult
from sevi.contracts.result import NormCell, NormCellKey, NormsFile, ResultDecision, ResultInput
from sevi.core.wording import DEFAULT_COPY_PATH, WordingTable, load_copy

ALL4: Final = (Surface.SPOKEN, Surface.PAGE, Surface.REPORT, Surface.OPERATOR)
PH3: Final = (Surface.SPOKEN, Surface.REPORT, Surface.OPERATOR)
T7_NO_SPOKEN: Final = (Surface.PAGE, Surface.REPORT, Surface.OPERATOR)
T7_ENDS: Final = frozenset(
    {EndReason.CALLER_HANGUP, EndReason.OPERATOR_ABORT, EndReason.SESSION_TIMEOUT}
)
T7_SPOKEN_ENDS: Final = frozenset({EndReason.SESSION_TIMEOUT, EndReason.OPERATOR_ABORT})
T8_ENDS: Final = frozenset({EndReason.SERVER_ERROR, EndReason.SOCKET_LOST})
T9_ENDS: Final = frozenset(
    {EndReason.CONSENT_NO_REPLY, EndReason.DIAL_FAILED, EndReason.QUEUE_ABANDONED}
)
UNCOMPARABLE_REASONS: Final = frozenset(
    {
        NoBandReason.FRIENDS_DISALLOWED,
        NoBandReason.SPEAKER_TRANSDUCER,
        NoBandReason.NORMS_BANK_MISMATCH,
    }
)
RANGE_REASONS: Final = frozenset({NoBandReason.EXPERIMENTAL_LANGUAGE, NoBandReason.MIXED_MODE})
MINUS: Final = chr(0x2212)  # U+2212 MINUS SIGN


def srt_display(srt_db: float) -> str:
    """Round half up to 0.5 dB; U+2212 minus; "+" for positive, no sign for zero (§8.2)."""
    v = math.floor(2.0 * srt_db + 0.5) / 2.0
    sign = MINUS if v < 0 else ("+" if v > 0 else "")
    return f"{sign}{abs(v):.1f}"


def find_cell(norms: NormsFile | None, key: NormCellKey) -> NormCell | None:
    """Exact match on all four key fields; the last match wins (a malformed file)."""
    if norms is None:
        return None
    found: NormCell | None = None
    for cell in norms.cells:
        if cell.key == key:
            found = cell
    return found


def _decision(
    rule: TruthRule,
    band: Band,
    wording: WordingClass,
    surfaces: tuple[Surface, ...],
    srt: SrtResult | None,
    *,
    refer: ReferReason | None = None,
    no_band: NoBandReason | None = None,
    cell: NormCell | None = None,
) -> ResultDecision:
    srt_db = srt.srt_db if srt is not None else None
    return ResultDecision(
        rule=rule,
        band=band,
        refer_reason=refer,
        no_band_reason=no_band,
        wording_class=wording,
        surfaces=surfaces,
        norm_cell=cell.key if cell is not None else None,
        cutoff_version=cell.cutoff_version if cell is not None else None,
        nh95_db=cell.nh95_db if cell is not None else None,
        n_norm=cell.n if cell is not None else None,
        srt_db=srt_db,
        srt_display=srt_display(srt_db) if srt_db is not None else None,
        ages=cell.ages if cell is not None else None,
    )


def t8_decision(srt: SrtResult | None) -> ResultDecision:
    """The exact T8 decision (FAILED, W_SYSFAIL, ALL4). Pure; never raises (CCR-v3-20)."""
    return _decision(TruthRule.T8, Band.FAILED, WordingClass.W_SYSFAIL, ALL4, srt)


def _cell_key(inp: ResultInput, mode: Mode) -> NormCellKey | None:
    if mode not in (Mode.VOICE, Mode.KEYPAD) or inp.transducer is Transducer.SPEAKER:
        return None
    return NormCellKey(lang=inp.lang, channel=inp.channel, mode=mode, transducer=inp.transducer)


def _lookup(inp: ResultInput, mode: Mode) -> NormCell | None:
    key = _cell_key(inp, mode)
    return find_cell(inp.norms, key) if key is not None else None


def _no_band(inp: ResultInput) -> tuple[NoBandReason | None, NormCell | None]:
    """T11 reason (first of, in order) and the matched cell (None if no cell was matched)."""
    if inp.experimental_language:
        return NoBandReason.EXPERIMENTAL_LANGUAGE, None
    if inp.friends is FriendsVariant.DISALLOWED:
        return NoBandReason.FRIENDS_DISALLOWED, None
    if inp.mode is Mode.MIXED:
        return NoBandReason.MIXED_MODE, None
    if inp.transducer is Transducer.SPEAKER:
        return NoBandReason.SPEAKER_TRANSDUCER, None
    if inp.channel is Channel.PHONE:
        return NoBandReason.NO_CELL, None
    cell = _lookup(inp, inp.mode)
    if cell is None:
        return NoBandReason.NO_CELL, None
    if cell.n < NORM_MIN_N:
        return NoBandReason.CELL_LT6, cell
    if cell.bank_hash != inp.bank_hash:
        return NoBandReason.NORMS_BANK_MISMATCH, cell
    return None, cell


def _t11(
    inp: ResultInput, srt: SrtResult, reason: NoBandReason, cell: NormCell | None
) -> ResultDecision:
    surfaces = PH3 if inp.channel is Channel.PHONE else ALL4
    if reason in UNCOMPARABLE_REASONS:
        return _decision(
            TruthRule.T11, Band.NO_BAND, WordingClass.W_NOBAND_UNCOMPARABLE, surfaces, srt,
            no_band=reason, cell=cell,
        )  # fmt: skip
    if reason in RANGE_REASONS:
        ref = _lookup(inp, inp.mode_requested)
        if ref is not None and ref.n >= NORM_MIN_N and ref.bank_hash == inp.bank_hash:
            return _decision(
                TruthRule.T11, Band.NO_BAND, WordingClass.W_NOBAND_RANGE, surfaces, srt,
                no_band=reason, cell=ref,
            )  # fmt: skip
    return _decision(
        TruthRule.T11, Band.NO_BAND, WordingClass.W_NOBAND_LT6, surfaces, srt,
        no_band=reason, cell=cell,
    )  # fmt: skip


def classify(inp: ResultInput) -> ResultDecision:
    """Contracts §4.7.1, first match wins. Pure; never raises for a valid ResultInput."""
    s = inp.srt
    end = inp.end_reason
    n24 = s is not None and s.n_scored == 24
    term = s.terminal if s is not None else None

    if term is ProcedureTerminal.VOLUME_FAILED:  # T1
        return _decision(
            TruthRule.T1, Band.REFER, WordingClass.W_REF_LOUD, ALL4, s,
            refer=ReferReason.LOUD_NOT_HEARD,
        )  # fmt: skip
    if term is ProcedureTerminal.PRACTICE_FAILED:  # T2
        return _decision(
            TruthRule.T2, Band.REFER, WordingClass.W_REF_PRACTICE, ALL4, s,
            refer=ReferReason.PRACTICE_FAILED,
        )  # fmt: skip
    if s is not None and (term is ProcedureTerminal.LOUD_FAILED or s.loud_wrong_total >= 2):
        return _decision(
            TruthRule.T3, Band.REFER, WordingClass.W_REF_LOUD, ALL4, s,
            refer=ReferReason.NO_RESPONSE_LOUD,
        )  # fmt: skip
    if end is EndReason.CANT_HEAR:  # T4
        return _decision(
            TruthRule.T4, Band.REFER, WordingClass.W_REF_LOUD, ALL4, s,
            refer=ReferReason.CANT_HEAR,
        )  # fmt: skip
    if s is not None and s.ceiling_fail:  # T5
        return _decision(
            TruthRule.T5, Band.REFER, WordingClass.W_REF_CEILING, ALL4, s,
            refer=ReferReason.CEILING,
        )  # fmt: skip
    if s is not None and s.flags > FLAG_LIMIT:  # T6
        return _decision(TruthRule.T6, Band.INCONCLUSIVE, WordingClass.W_INCONC, ALL4, s)
    if (not n24 and end in T7_ENDS) or term is ProcedureTerminal.NO_RESPONSE_END:  # T7
        # never SPOKEN for CALLER_HANGUP (the caller is gone), whatever the terminal
        spoken = end is not EndReason.CALLER_HANGUP and (
            term is ProcedureTerminal.NO_RESPONSE_END or end in T7_SPOKEN_ENDS
        )
        surfaces = ALL4 if spoken else T7_NO_SPOKEN
        return _decision(TruthRule.T7, Band.INCOMPLETE, WordingClass.W_INC, surfaces, s)
    if not n24 and end in T8_ENDS:  # T8
        return t8_decision(s)
    if end in T9_ENDS:  # T9
        return _decision(
            TruthRule.T9, Band.NOT_STARTED, WordingClass.W_NOREPLY,
            (Surface.OPERATOR, Surface.REPORT), s,
        )  # fmt: skip
    if end is EndReason.CONSENT_DECLINED:  # T10
        return _decision(
            TruthRule.T10, Band.DECLINED, WordingClass.W_DECLINED, (Surface.SPOKEN, Surface.PAGE), s
        )
    if s is None or not n24 or s.srt_db is None:  # bug path (R-04)
        return t8_decision(s)
    x = s.srt_db
    if x >= ABS_REFER_DB:  # T10b
        return _decision(
            TruthRule.T10B, Band.REFER, WordingClass.W_REF_ABS,
            PH3 if inp.channel is Channel.PHONE else ALL4, s,
            refer=ReferReason.SRT_ABSOLUTE,
        )  # fmt: skip
    reason, cell = _no_band(inp)
    if reason is not None or cell is None or cell.nh95_db is None:  # T11
        return _t11(inp, s, reason or NoBandReason.CELL_LT6, cell)
    h = cell.nh95_db
    if x > h + REFER_MARGIN_DB:  # T12
        return _decision(
            TruthRule.T12, Band.REFER, WordingClass.W_REF_SRT, ALL4, s,
            refer=ReferReason.SRT, cell=cell,
        )  # fmt: skip
    if (
        h < x <= h + REFER_MARGIN_DB
        or (cell.cutoff_version is CutoffVersion.V0 and abs(x - h) <= V0_BORDER_HALFWIDTH_DB)
        or (s.reversals is not None and s.reversals < REVERSALS_MIN)
        or s.censor is Censor.HIGH
    ):  # T13
        return _decision(TruthRule.T13, Band.BORDERLINE, WordingClass.W_BORDER, ALL4, s, cell=cell)
    if inp.hhie_total is not None and inp.hhie_total >= 10:  # T14
        return _decision(
            TruthRule.T14, Band.PASS_WITH_NOTE, WordingClass.W_PASS_NOTE, ALL4, s, cell=cell
        )
    return _decision(TruthRule.T15, Band.PASS, WordingClass.W_PASS, ALL4, s, cell=cell)  # T15


class TruthTableClassifier:
    """Implements :class:`sevi.contracts.ports.ResultClassifier`. Immutable after construction."""

    def __init__(self, wording: WordingTable) -> None:
        self._wording = wording

    def classify(self, inp: ResultInput) -> ResultDecision:
        return classify(inp)

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> WordingText:
        return self._wording.wording(decision, lang, surface)


def build_classifier(
    copy_path: str = DEFAULT_COPY_PATH, selection_path: str | None = None
) -> ResultClassifier:
    """Contracts §5.1.1: ``TruthTableClassifier(load_copy(copy_path, selection_path))``."""
    return TruthTableClassifier(load_copy(copy_path, selection_path))
