"""Zero-numeral output guard for Evon text (contracts §8.5.4; B spec §11.1). Pure.

Every Evon text passes ``check`` before it can reach a page or TTS. A rejected text is never
shown: the caller regenerates once, then falls back to the template.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import Literal

from sevi.contracts.enums import Lang
from sevi.contracts.types import SeviModel

Purpose = Literal["EXPLAIN_SPOKEN", "REPORT_PROSE"]

MAX_LEN: dict[str, int] = {"EXPLAIN_SPOKEN": 300, "REPORT_PROSE": 600}
SCRIPT_MIN_SHARE = 0.80
TAMIL_PREFIX_MIN = 3  # Tamil-script entries this long also match as token prefixes

UNIT_TOKENS = frozenset({"db", "snr", "percent", "decibel", "டெசிபல்", "சதவீதம்"})
_APOSTROPHES = ("'", "’")  # noqa: RUF001


class GuardReason(StrEnum):
    EMPTY = "EMPTY"
    LENGTH = "LENGTH"
    NUMERAL = "NUMERAL"
    NUMBER_WORD = "NUMBER_WORD"
    UNIT = "UNIT"
    MEDICAL = "MEDICAL"
    SCRIPT = "SCRIPT"
    BANNED = "BANNED"


class GuardVerdict(SeviModel):
    ok: bool
    reasons: tuple[GuardReason, ...]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def _is_word_char(c: str) -> bool:
    return unicodedata.category(c)[0] in ("L", "M")


def _is_latin(c: str) -> bool:
    o = ord(c)
    return 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or 0xC0 <= o <= 0x24F


def _is_tamil(c: str) -> bool:
    return 0x0B80 <= ord(c) <= 0x0BFF


def tokenize(text: str) -> list[str]:
    """Maximal runs of L/M characters; an apostrophe between two Latin letters joins the run."""
    tokens: list[str] = []
    cur: list[str] = []
    n = len(text)
    for i, c in enumerate(text):
        if _is_word_char(c):
            cur.append(c)
        elif (
            c in _APOSTROPHES
            and cur
            and _is_latin(cur[-1])
            and i + 1 < n
            and _is_latin(text[i + 1])
        ):
            cur.append("'")
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def read_word_list(name: str) -> tuple[str, ...]:
    """UTF-8, one entry per line, ``#`` comments, blank lines ignored; entries NFC + casefolded."""
    raw = files("sevi.core").joinpath(name).read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            out.append(_norm(entry))
    return tuple(out)


@cache
def number_entries() -> tuple[tuple[str, ...], ...]:
    """Both number lists (ta + en, regardless of lang), each entry as its token sequence."""
    seqs: list[tuple[str, ...]] = []
    for name in ("guard_numbers/ta.txt", "guard_numbers/en.txt"):
        for entry in read_word_list(name):
            toks = tuple(tokenize(entry))
            if toks:
                seqs.append(toks)
    return tuple(seqs)


@cache
def medical_entries() -> tuple[str, ...]:
    return read_word_list("guard_medical.txt")


def _token_matches(token: str, entry: str) -> bool:
    if token == entry:
        return True
    return (
        len(entry) >= TAMIL_PREFIX_MIN
        and all(_is_tamil(c) for c in entry)
        and token.startswith(entry)
    )


def _has_number_word(tokens: Sequence[str]) -> bool:
    for seq in number_entries():
        k = len(seq)
        for i in range(len(tokens) - k + 1):
            if all(_token_matches(tokens[i + j], seq[j]) for j in range(k)):
                return True
    return False


def _has_numeral(n: str) -> bool:
    return any(unicodedata.category(c) in ("Nd", "No") or "௰" <= c <= "௲" for c in n)


def _script_ok(t: str, lang: Lang) -> bool:
    letters = [c for c in t if unicodedata.category(c).startswith("L")]
    if not letters:
        return True
    in_range = _is_tamil if lang == Lang.TA else _is_latin
    return sum(1 for c in letters if in_range(c)) / len(letters) >= SCRIPT_MIN_SHARE


def check(text: str, lang: Lang, purpose: Purpose, banned: Sequence[str] = ()) -> GuardVerdict:
    """All failing reasons, in enum order; ``ok = not reasons`` (B spec §11.1 algorithm)."""
    n = unicodedata.normalize("NFC", text)
    t = n.casefold()
    tokens = tokenize(t)
    failed: set[GuardReason] = set()
    if t.strip() == "":
        failed.add(GuardReason.EMPTY)
    if len(n) > MAX_LEN[purpose]:
        failed.add(GuardReason.LENGTH)
    if _has_numeral(n):
        failed.add(GuardReason.NUMERAL)
    if _has_number_word(tokens):
        failed.add(GuardReason.NUMBER_WORD)
    if "%" in t or any(tok in UNIT_TOKENS for tok in tokens):
        failed.add(GuardReason.UNIT)
    if any(m in t for m in medical_entries()):
        failed.add(GuardReason.MEDICAL)
    if not _script_ok(t, lang):
        failed.add(GuardReason.SCRIPT)
    if any(_norm(b) in t for b in banned if b.strip()):
        failed.add(GuardReason.BANNED)
    reasons = tuple(r for r in GuardReason if r in failed)
    return GuardVerdict(ok=not reasons, reasons=reasons)
