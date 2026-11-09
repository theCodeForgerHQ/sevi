# ruff: noqa: RUF001  (the confusables table is deliberately full of lookalike letters)
"""Zero-numeral output guard for Evon text (contracts §8.5.4; B spec §11.1). Pure.

Every Evon text passes ``check`` before it can reach a page or TTS. A rejected text is never
shown: the caller regenerates once, then falls back to the template.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import Literal

from sevi.contracts.enums import Lang
from sevi.contracts.types import SeviModel

Purpose = Literal["EXPLAIN_SPOKEN", "REPORT_PROSE", "COPY_SPOKEN"]

# EXPLAIN_SPOKEN bounds what Evon may generate live. COPY_SPOKEN is human-reviewed, pre-recorded
# copy: same content rules, but its length is set by the approved wording, not by the LLM cap
# (Tamil needs roughly twice the characters of English for the same sentence).
MAX_LEN: dict[str, int] = {"EXPLAIN_SPOKEN": 300, "REPORT_PROSE": 600, "COPY_SPOKEN": 1000}
SCRIPT_MIN_SHARE = 0.80
TAMIL_PREFIX_MIN = 3  # Tamil-script entries this long also match as token prefixes

UNIT_TOKENS = frozenset({"db", "snr", "percent", "decibel", "டெசிபல்", "சதவீதம்"})
_APOSTROPHES = ("'", "’")


class GuardReason(StrEnum):
    EMPTY = "EMPTY"
    LENGTH = "LENGTH"
    NUMERAL = "NUMERAL"
    NUMBER_WORD = "NUMBER_WORD"
    UNIT = "UNIT"
    MEDICAL = "MEDICAL"
    SCRIPT = "SCRIPT"
    BANNED = "BANNED"
    REASSURANCE = "REASSURANCE"  # review #36: false-reassurance phrases (en + ta)


# Lookalike letters folded to Latin before matching (small confusables skeleton, UTS #39).
_CONFUSABLES = str.maketrans(
    {
        # Cyrillic
        "а": "a",
        "в": "b",
        "е": "e",
        "ё": "e",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "с": "c",
        "т": "t",
        "у": "y",
        "х": "x",
        "ѕ": "s",
        "і": "i",
        "ї": "i",
        "ј": "j",
        "ԁ": "d",
        "ԛ": "q",
        "ԝ": "w",
        "ɡ": "g",
        "һ": "h",
        "ӏ": "l",
        # Greek
        "α": "a",
        "β": "b",
        "ε": "e",
        "η": "n",
        "ι": "i",
        "κ": "k",
        "ν": "v",
        "ο": "o",
        "ρ": "p",
        "τ": "t",
        "υ": "u",
        "χ": "x",
        "ω": "w",
        "ϲ": "c",
        "ζ": "z",
    }
)
_ROMAN = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
# Valid Roman numerals that are also ordinary words or Tanglish syllables.
_ROMAN_WORDS = frozenset({"mix", "mid", "dim", "di", "mi", "li", "cc", "mm", "dc", "cv", "lic"})
# Latin inflections reduced to a base form: plurals, ordinals, -fold (sixes, nineties, fivefold).
_LATIN_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ieths", "y"),
    ("ieth", "y"),
    ("ies", "y"),
    ("fold", ""),
    ("ths", ""),
    ("th", ""),
    ("es", ""),
    ("s", ""),
)
_TAMIL_U = "\u0bc1"  # vowel sign u: பத்து -> stem பத்த (பத்தில், பத்தை)


def clean(text: str) -> str:
    """NFKC, drop format characters (Cf: ZWSP, ZWJ, ZWNJ, soft hyphen, ...), casefold, fold
    lookalike Cyrillic/Greek letters to Latin. Applied before every check."""
    n = unicodedata.normalize("NFKC", text)
    n = "".join(c for c in n if unicodedata.category(c) != "Cf" and c != "\u00ad")
    return unicodedata.normalize("NFC", n.casefold())


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


@cache
def tamil_stems() -> tuple[str, ...]:
    """Inflection stems of Tamil number words: drop the final -u (பத்து -> பத்த) so case
    forms whose -u changes (பத்தில், மூன்றை) still match as prefixes. Stems shorter than
    3 code points are skipped (ஆற, ஏழ would hit ஆற்றல், ஏழை)."""
    out: set[str] = set()
    for seq in number_entries():
        if len(seq) == 1 and seq[0].endswith(_TAMIL_U) and all(_is_tamil(c) for c in seq[0]):
            stem = seq[0][:-1]
            if len(stem) >= TAMIL_PREFIX_MIN:
                out.add(stem)
    return tuple(sorted(out))


@cache
def _latin_singles() -> frozenset[str]:
    return frozenset(seq[0] for seq in number_entries() if len(seq) == 1)


@cache
def reassurance_patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in read_word_list("guard_reassurance.txt"))


def _latin_number_token(tok: str) -> bool:
    if not all(_is_latin(c) for c in tok):
        return False
    if len(tok) >= 2 and _ROMAN.match(tok) and tok not in _ROMAN_WORDS:
        return True
    singles = _latin_singles()
    for suffix, repl in _LATIN_SUFFIXES:
        if tok.endswith(suffix) and len(tok) > len(suffix) + 1:
            base = tok[: -len(suffix)] + repl
            if base in singles or base + "e" in singles:  # fives -> five, ninths -> nine
                return True
    return False


def _token_matches(token: str, entry: str) -> bool:
    if token == entry:
        return True
    return (
        len(entry) >= TAMIL_PREFIX_MIN
        and all(_is_tamil(c) for c in entry)
        and token.startswith(entry)
    )


def _has_number_word(tokens: Sequence[str]) -> bool:
    stems = tamil_stems()
    for tok in tokens:
        if _latin_number_token(tok) or any(tok.startswith(st) for st in stems):
            return True
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
    # Any letter outside the allowed scripts (en: Latin; ta: Tamil + Latin) fails (homoglyphs).
    if any(not (_is_latin(c) or (lang == Lang.TA and _is_tamil(c))) for c in letters):
        return False
    in_range = _is_tamil if lang == Lang.TA else _is_latin
    return sum(1 for c in letters if in_range(c)) / len(letters) >= SCRIPT_MIN_SHARE


# Fixed disclaimer phrases that are always allowed through MEDICAL/REASSURANCE (never through the
# number checks). Copy sentences passed as ``allowed`` are added to these (review #36 round 3).
ALLOWED_PHRASES: tuple[str, ...] = (
    "not a diagnosis",
    "not a medical diagnosis",
    "நோயறிதல் அல்ல",
)


def _number_tokens(tokens: Sequence[str]) -> list[str]:
    """Possessives/contractions removed before number matching: "five's" -> "five"."""
    return [tok.split("'")[0] if "'" in tok else tok for tok in tokens]


def check(
    text: str,
    lang: Lang,
    purpose: Purpose,
    banned: Sequence[str] = (),
    allowed: Sequence[str] = (),
) -> GuardVerdict:
    """All failing reasons, in enum order; ``ok = not reasons`` (B spec §11.1 algorithm).

    ``allowed``: exact copy sentences (e.g. the disclaimer) removed from the text before the
    MEDICAL, REASSURANCE and BANNED checks only; numerals and number words are always checked
    on the full text."""
    n = unicodedata.normalize("NFC", text)
    raw = clean(text)  # NFKC, no Cf, casefolded; lookalikes still visible for SCRIPT
    t = raw.translate(_CONFUSABLES)  # lookalikes folded to Latin for matching
    tokens = tokenize(t)
    failed: set[GuardReason] = set()
    if t.strip() == "":
        failed.add(GuardReason.EMPTY)
    if len(n) > MAX_LEN[purpose]:
        failed.add(GuardReason.LENGTH)
    if _has_numeral(n) or _has_numeral(raw):
        failed.add(GuardReason.NUMERAL)
    if _has_number_word(tokens) or _has_number_word(_number_tokens(tokens)):
        failed.add(GuardReason.NUMBER_WORD)
    if "%" in t or any(tok in UNIT_TOKENS for tok in tokens):
        failed.add(GuardReason.UNIT)
    prose = t
    # Longest first, so a full approved sentence is removed before a disclaimer phrase inside
    # it (e.g. "not a diagnosis") breaks its exact match (issue #77).
    cleaned = {clean(ph).translate(_CONFUSABLES).strip() for ph in (*ALLOWED_PHRASES, *allowed)}
    for p in sorted((c for c in cleaned if c), key=len, reverse=True):
        prose = prose.replace(p, " ")
    if any(m in prose for m in medical_entries()):
        failed.add(GuardReason.MEDICAL)
    reassured = any(p.search(prose) for p in reassurance_patterns())
    if not _script_ok(raw, lang):
        failed.add(GuardReason.SCRIPT)
    if any(_norm(b) in prose for b in banned if b.strip()):
        failed.add(GuardReason.BANNED)
    if reassured:
        failed.add(GuardReason.REASSURANCE)
    reasons = tuple(r for r in GuardReason if r in failed)
    return GuardVerdict(ok=not reasons, reasons=reasons)
