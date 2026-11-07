"""Minimal ResultInput / SrtResult / norms builders for result tests (Person A spec §8.6)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.enums import (
    Censor,
    Channel,
    CutoffVersion,
    EndReason,
    FriendsVariant,
    Lang,
    Mode,
    Owner,
    ParticipantType,
    ProcedureTerminal,
    Transducer,
)
from sevi.contracts.evals import ResultFileMeta
from sevi.contracts.procedure import SrtResult
from sevi.contracts.result import NormCell, NormCellKey, NormsFile, ResultInput

BANK: str = "a" * 64
OTHER_BANK: str = "b" * 64
SESSION: str = "01890a5d-ac96-774b-bcce-b302099a8057"
NOW = datetime(2026, 11, 7, tzinfo=UTC)


def srt(**kw: Any) -> SrtResult:
    """A complete, clean 24-trial SRT at -8.2 dB unless overridden."""
    base: dict[str, Any] = {
        "n_scored": 24,
        "complete": True,
        "srt_db": -8.2,
        "track_db": (),
        "virtual_25_db": None,
        "reversals": 8,
        "censor": Censor.NONE,
        "flags": 0,
        "valid_at_ceiling": 0,
        "correct_at_ceiling": 0,
        "max_consec_valid_wrong_at_ceiling": 0,
        "ceiling_fail": False,
        "mode": Mode.KEYPAD,
        "terminal": ProcedureTerminal.COMPLETE,
        "loud_wrong_total": 0,
        "volume_failed": False,
        "practice_failed": False,
    }
    base.update(kw)
    if base["n_scored"] != 24 and "complete" not in kw:
        base["complete"] = False
    return SrtResult(**base)


def cell(
    n: int = 7,
    nh95: float | None = -6.800144,
    *,
    lang: Lang = Lang.TA,
    mode: Mode = Mode.KEYPAD,
    transducer: Transducer = Transducer.EARPHONES,
    bank_hash: str = BANK,
    ages: str = "18-35",
) -> NormCell:
    cv = None if n < 6 else (CutoffVersion.V0 if n < 10 else CutoffVersion.V1)
    return NormCell(
        key=NormCellKey(lang=lang, channel=Channel.BROWSER, mode=mode, transducer=transducer),  # type: ignore[arg-type]
        n=n,
        mean_db=-8.014286,
        sd_db=0.73808,
        nh95_db=nh95 if n >= 6 else None,
        cutoff_version=cv,
        ages=ages,
        bank_hash=bank_hash,
        computed_at=NOW,
    )


def norms(*cells: NormCell) -> NormsFile:
    return NormsFile(
        meta=ResultFileMeta(
            schema_id="sevi.results.norms/1",
            generated_at=NOW,
            git_sha="0" * 40,
            generator="test",
            owner=Owner.A,
            contracts_version=CONTRACTS_VERSION,
        ),
        cells=cells,
    )


def inp(**kw: Any) -> ResultInput:
    """Banded ta/browser/keypad/earphones, completed, with the worked-example cell (n = 7)."""
    base: dict[str, Any] = {
        "session_id": SESSION,
        "lang": Lang.TA,
        "channel": Channel.BROWSER,
        "mode": Mode.KEYPAD,
        "mode_requested": Mode.KEYPAD,
        "transducer": Transducer.EARPHONES,
        "participant_type": ParticipantType.PUBLIC,
        "end_reason": EndReason.COMPLETED,
        "srt": srt(),
        "friends": FriendsVariant.ALLOWED,
        "experimental_language": False,
        "bank_hash": BANK,
        "norms": norms(cell()),
        "hhie_total": None,
    }
    base.update(kw)
    return ResultInput(**base)
