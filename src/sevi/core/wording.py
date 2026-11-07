"""Wording class -> copy key, variant and filled slots (Person A spec §8.3; contracts §4.11).

The copy table is ``src/sevi/copy/copy.yaml`` (:class:`CopyFile`). For each entry the selected
variant is the one recorded in ``results/copy_selection.json`` (if given), else the first variant
whose ``when`` all match the outcome values (the ``when == {}`` default is last).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import yaml

from sevi.contracts.copy import CopyEntry, CopyFile, CopyVariant, OutcomeVar
from sevi.contracts.enums import Lang, Surface, WordingClass
from sevi.contracts.evals import CopySelectionFile
from sevi.contracts.ports import WordingText
from sevi.contracts.result import ResultDecision

DEFAULT_COPY_PATH: Final = str(Path(__file__).resolve().parent.parent / "copy" / "copy.yaml")
MISSING_SLOT: Final = "—"  # em dash for a None slot (U-WD-04 proves it is unreachable)
SLOT_RE: Final = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Reduced scope (BUILD-BRIEF): no human panel, no Prisma grid, no carrier check, no validation,
# browser only. These are the outcome values used when no selection file is given.
REDUCED_SCOPE_OUTCOMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        OutcomeVar.SHIP.value: "NO_PANEL",
        OutcomeVar.GRID.value: "NOT_RUN",
        OutcomeVar.CARRIER.value: "NOT_RUN",
        OutcomeVar.LP_SHIFT.value: "WEAK",
        OutcomeVar.PHONE.value: "BROWSER_ONLY",
        OutcomeVar.EVON.value: "TEMPLATE",
        OutcomeVar.FRIENDS.value: "ALLOWED",
        OutcomeVar.VERSION.value: "v3",
    }
)

_ALL4: Final = (Surface.SPOKEN, Surface.PAGE, Surface.REPORT, Surface.OPERATOR)
# Surfaces per wording class = union over the §4.7.1 rows that use it.
CLASS_SURFACES: Final[Mapping[WordingClass, tuple[Surface, ...]]] = MappingProxyType(
    {
        **{wc: _ALL4 for wc in WordingClass},
        WordingClass.W_NOREPLY: (Surface.OPERATOR, Surface.REPORT),
        WordingClass.W_DECLINED: (Surface.SPOKEN, Surface.PAGE),
    }
)
# Wording slots and the decision field that fills each (§8.3).
WORDING_SLOTS: Final = frozenset({"SRT", "N", "AGES"})


def wording_key(wording_class: WordingClass, surface: Surface) -> str:
    return f"wording.{wording_class.value}.{surface.value.lower()}"


def fill(text: str, slots: Mapping[str, str | None]) -> str:
    """Replace ``{NAME}`` with ``slots[NAME]``; a None value renders as an em dash. Unknown
    placeholders are left untouched."""

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in slots:
            return m.group(0)
        value = slots[name]
        return MISSING_SLOT if value is None else value

    return SLOT_RE.sub(_sub, text)


def placeholders(text: str) -> frozenset[str]:
    return frozenset(SLOT_RE.findall(text))


def select_variant(entry: CopyEntry, outcomes: Mapping[str, str]) -> CopyVariant:
    """First variant whose ``when`` all match; the default (``when == {}``) is last."""
    for v in entry.variants:
        if all(outcomes.get(var.value) == val for var, val in v.when.items()):
            return v
    raise LookupError(f"{entry.key}: no variant matches and no default variant")


def select_all(copy: CopyFile, outcomes: Mapping[str, str]) -> dict[str, str]:
    """Copy key -> selected variant id, for every entry."""
    return {e.key: select_variant(e, outcomes).variant_id for e in copy.entries}


def parse_copy(raw: str) -> CopyFile:
    return CopyFile.model_validate(yaml.safe_load(raw))


@dataclass(frozen=True)
class WordingTable:
    """Immutable: copy entries, the selected variant per key and the outcome values."""

    copy: CopyFile
    selected: Mapping[str, str]
    outcomes: Mapping[str, str]

    def entry(self, key: str) -> CopyEntry:
        for e in self.copy.entries:
            if e.key == key:
                return e
        raise KeyError(key)

    def variant(self, key: str) -> CopyVariant:
        entry = self.entry(key)
        vid = self.selected.get(key)
        if vid is not None:
            for v in entry.variants:
                if v.variant_id == vid:
                    return v
        return select_variant(entry, self.outcomes)

    def text(self, key: str, lang: Lang, slots: Mapping[str, str | None] | None = None) -> str:
        return fill(self.variant(key).text[lang.value], slots or {})

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> WordingText:
        key = wording_key(decision.wording_class, surface)
        variant = self.variant(key)
        slots: dict[str, str | None] = {
            "SRT": decision.srt_display,
            "N": str(decision.n_norm) if decision.n_norm is not None else None,
            "AGES": decision.ages,
        }
        return WordingText(
            wording_class=decision.wording_class.value,
            surface=surface,
            lang=lang.value,
            text=fill(variant.text[lang.value], slots),
            copy_key=key,
            variant_id=variant.variant_id,
        )


def load_copy(
    copy_path: str = DEFAULT_COPY_PATH,
    selection_path: str | None = None,
    outcomes: Mapping[str, str] = REDUCED_SCOPE_OUTCOMES,
) -> WordingTable:
    """Construction I/O. A missing selection file means "first matching variant" (§8.5)."""
    copy = parse_copy(Path(copy_path).read_text(encoding="utf-8"))
    selected: dict[str, str] = {}
    values = dict(outcomes)
    if selection_path is not None and Path(selection_path).is_file():
        sel = CopySelectionFile.model_validate(
            json.loads(Path(selection_path).read_text(encoding="utf-8"))
        )
        selected = dict(sel.selected)
        values.update(sel.outcome_values)
    else:
        selected = select_all(copy, values)
    return WordingTable(
        copy=copy, selected=MappingProxyType(selected), outcomes=MappingProxyType(values)
    )
