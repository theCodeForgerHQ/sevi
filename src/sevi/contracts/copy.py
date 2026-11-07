"""Copy-table entry and outcome variables (contracts §4.11).

The copy table lives at ``src/sevi/copy/copy.yaml``. Selection rule: for each entry the selected
variant is the **first** whose ``when`` all match the outcome values; a ``when == {}`` variant
MUST be last (validated below).
"""

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal

from pydantic import Field, model_validator

from .types import SeviModel, UtcDatetime


class OutcomeVar(StrEnum):
    SHIP = "SHIP"
    EVON = "EVON"
    FRIENDS = "FRIENDS"
    PHONE = "PHONE"
    LP_SHIFT = "LP_SHIFT"
    VERSION = "VERSION"
    CARRIER = "CARRIER"  # real carrier channel check (cut in this build: always NOT_RUN)
    GRID = "GRID"  # did the ta 1 dB Prisma grid run (cut in this build: always NOT_RUN)


OUTCOME_VALUES: Final[MappingProxyType[str, tuple[str, ...]]] = MappingProxyType(
    {
        # outcome-complete; reduced scope ships NO_PANEL (or PRISMA_UNUSABLE) with an RMS bank
        "SHIP": ("PRISMA", "HUMAN", "RMS", "NO_PANEL", "PRISMA_UNUSABLE"),
        "EVON": ("LIVE", "TEMPLATE"),
        "FRIENDS": ("ALLOWED", "DISALLOWED"),
        "PHONE": ("OK", "BROWSER_ONLY"),
        "LP_SHIFT": ("OK", "WEAK"),
        "VERSION": ("v3", "v2.5"),
        "CARRIER": ("MATCH", "DIFFERS", "NOT_RUN"),
        "GRID": ("RAN", "NOT_RUN"),
    }
)
# Sources: SHIP, first match: gates.json g1b == FAIL -> PRISMA_UNUSABLE; panel_outcome.json
#   missing -> NO_PANEL (always, in this build: the panel is cut); else panel_outcome.json.ship.
#   CARRIER <- NOT_RUN (channel check cut). GRID <- NOT_RUN (Prisma grid cut).
#   EVON, FRIENDS, PHONE, VERSION <- gates.json (PHONE = OK iff g3t PASS).
#   LP_SHIFT <- WEAK (validation cut: no observed shift).


class CopySurface(StrEnum):
    SPOKEN = "SPOKEN"
    PAGE = "PAGE"
    REPORT = "REPORT"
    LANDING = "LANDING"
    README = "README"
    POST_LINKEDIN = "POST_LINKEDIN"
    POST_X = "POST_X"
    SHARE_CARD = "SHARE_CARD"
    VIDEO_CAPTION = "VIDEO_CAPTION"
    CONSENT = "CONSENT"
    PROMPT = "PROMPT"
    EVON_SYSTEM = "EVON_SYSTEM"
    OPERATOR = "OPERATOR"  # wording.<CLASS>.operator entries


class CopyVariant(SeviModel):
    variant_id: str = Field(pattern=r"^[a-z0-9_]+$")
    when: dict[OutcomeVar, str] = Field(default_factory=dict)  # all must match; {} = default
    text: dict[str, str]  # lang ("ta","en") -> text
    slots: tuple[str, ...] = ()  # e.g. ("N_pre_panel",); filled only from results/*
    # SPOKEN/PROMPT/CONSENT texts MUST NOT contain slots or digits (U-WD-02)

    @model_validator(mode="after")
    def _when_values(self) -> "CopyVariant":
        for var, value in self.when.items():
            if value not in OUTCOME_VALUES[var.value]:
                raise ValueError(f"when[{var.value}] = {value!r} is not an outcome value")
        return self


class CopyEntry(SeviModel):
    key: str = Field(pattern=r"^[A-Za-z0-9_.]+$")  # e.g. "wording.W_REF_LOUD.spoken"
    surface: CopySurface
    variants: tuple[CopyVariant, ...] = Field(min_length=1)
    # lang -> required substrings (P-RES-04)
    must_contain: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    banned_check: bool = True  # scanned for the banned list

    @model_validator(mode="after")
    def _default_last(self) -> "CopyEntry":
        if any(not v.when for v in self.variants[:-1]):
            raise ValueError("a variant with when == {} must be last")
        return self


class CopyFile(SeviModel):  # src/sevi/copy/copy.yaml
    schema_id: Literal["sevi.copy/1"] = "sevi.copy/1"
    frozen_at: UtcDatetime | None
    # banned phrases (L1 §1.1 + §4.11.2); case-insensitive, whole-phrase (\b...\b) match
    banned: tuple[str, ...]
    # lang -> words that count as "audiologist"/"hearing check"
    referral_keywords: dict[str, tuple[str, ...]]
    entries: tuple[CopyEntry, ...]
