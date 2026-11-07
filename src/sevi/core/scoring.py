"""Answer scoring: lexicon normaliser, Levenshtein alignment and the §4.5.1 status rules.

Implements ``contracts.ports.AnswerScorer`` (contracts §4.5, §4.5.1, §4.13 rules C1-C5).
Pure after construction: ``load_lexicons`` reads and validates the two lexicon files once and
returns an immutable ``LexiconScorer``. Keypad (DTMF) answers and voice transcripts both go
through ``LexiconScorer.score``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sevi.contracts.enums import (
    AnswerSource,
    AnswerStatus,
    Lang,
    Mode,
    ModelVersion,
    PromptKey,
    ScoreAction,
    SystemEvent,
)
from sevi.contracts.trial import AnswerEvidence, PresentationSpec, ScoringResult

__all__ = [
    "LEXICON_SCHEMA_ID",
    "LexiconFile",
    "LexiconScorer",
    "Parse",
    "align",
    "keyterms_sha256",
    "load_lexicons",
]

LEXICON_SCHEMA_ID = "sevi.lexicon/2"
MAX_RECOGNISED = 12
MAX_TOKENS = 32
MAX_PHRASE_TOKENS = 4
KEYTERMS_V3_MAX = 200
KEYTERMS_V2_5_MAX = 100

Intent = Literal["DONT_KNOW", "REPEAT"]

_DEFAULT_DIR = Path(__file__).resolve().parent / "lexicon"

# §7.3 step 1-2 character handling.
_ZERO_WIDTH = dict.fromkeys((0x200B, 0x200C, 0x200D, 0xFEFF), None)
_APOSTROPHES = dict.fromkeys((0x2019, 0x2018, 0x02BC, ord("`")), "'")
_PUNCT_CHARS = '.,!?;:"()[]{}<>-/\\|*#+=_~' + "".join(
    map(chr, (0x2013, 0x2014, 0x2026, 0x0964, 0x0965))
)
_PUNCT = dict.fromkeys(map(ord, _PUNCT_CHARS), " ")
_TRANSLATE: dict[int, str | None] = {**_ZERO_WIDTH, **_APOSTROPHES, **_PUNCT}

_ASCII_CONCAT = re.compile(r"[0-9]{2,}")
_ASCII_LETTERS = re.compile(r"[a-z]+")

# Tamil block and the C4 sandhi tables.
_TAMIL_LO, _TAMIL_HI = 0x0B80, 0x0BFF
_TAMIL_U_SIGN = "ு"  # ு
_TAMIL_CONSONANTS = frozenset(chr(c) for c in range(0x0B95, 0x0BBA))
_VOWEL_SIGN: Mapping[str, str] = MappingProxyType(
    {
        "அ": "",
        "ஆ": "ா",
        "இ": "ி",
        "ஈ": "ீ",
        "உ": "ு",
        "ஊ": "ூ",
        "எ": "ெ",
        "ஏ": "ே",
        "ஐ": "ை",
        "ஒ": "ொ",
        "ஓ": "ோ",
        "ஔ": "ௌ",
    }
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Priority order for a single transcript-form label (§12.7).
FORM_PRIORITY: tuple[str, ...] = (
    "compound",
    "grouping",
    "unparsed",
    "native_numerals",
    "western_concat",
    "western_spaced",
    "english_in_tamil_script",
    "native_words",
    "tanglish",
    "english_words",
    "none",
)


# ---------------------------------------------------------------------------------------------
# Lexicon file schema (§7.2)
# ---------------------------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class LexiconIntents(_Strict):
    DONT_KNOW: tuple[str, ...] = Field(min_length=1)
    REPEAT: tuple[str, ...] = Field(min_length=1)


class LexiconCompound(_Strict):
    hundreds: dict[str, int] = Field(default_factory=dict)
    hundred_multiplier: tuple[str, ...] = ()
    tens: dict[str, int] = Field(default_factory=dict)
    teens: dict[str, int] = Field(default_factory=dict)
    num2: dict[str, int] = Field(default_factory=dict)


class LexiconFile(_Strict):
    """One ``core/lexicon/<lang>.yaml`` file (schema ``sevi.lexicon/2``)."""

    schema_id: Literal["sevi.lexicon/2"]
    lang: Lang
    digits: dict[str, tuple[str, ...]]
    english_in_tamil_script: tuple[str, ...] = ()
    intents: LexiconIntents
    fillers: tuple[str, ...]
    repeat: dict[str, int]
    compound: LexiconCompound


# ---------------------------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------------------------


class _K(Enum):
    DIGIT = "DIGIT"
    DK = "DONT_KNOW"
    REPEAT = "REPEAT"
    FILLER = "FILLER"
    REP = "REP"
    HUNDREDS = "HUNDREDS"
    MULT = "MULT"
    TENS = "TENS"
    TEENS = "TEENS"
    NUM2 = "NUM2"
    UNK = "UNK"


@dataclass(frozen=True, slots=True)
class _Meaning:
    kind: _K
    value: int = 0
    form: str | None = None  # transcript-form label contributed when matched (DIGIT / REP)


@dataclass(frozen=True, slots=True)
class _Sym:
    kind: _K
    value: int = 0


class Parse(NamedTuple):
    digits: tuple[int, ...]
    intent: Intent | None
    number_form: bool
    unparsed: bool
    forms: tuple[str, ...]
    tokens: tuple[str, ...]


# ---------------------------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------------------------


def _clean(s: str) -> str:
    """§7.3 steps 1-2 (without the numeral split)."""
    s = unicodedata.normalize("NFC", s).casefold()
    return s.translate(_TRANSLATE)


def _tokenise_phrase(phrase: str) -> tuple[str, ...]:
    return tuple(_clean(phrase).split())


def _is_tamil(s: str) -> bool:
    return all(_TAMIL_LO <= ord(c) <= _TAMIL_HI for c in s)


def _has_tamil(s: str) -> bool:
    return any(_TAMIL_LO <= ord(c) <= _TAMIL_HI for c in s)


def _split_numerals(s: str) -> tuple[list[str | int], set[str]]:
    """§7.3 steps 3-4: every decimal character becomes its own (int) token."""
    forms: set[str] = set()
    if _ASCII_CONCAT.search(s):
        forms.add("western_concat")
    out: list[str | int] = []
    for word in s.split():
        buf: list[str] = []
        for ch in word:
            d = unicodedata.decimal(ch, None)
            if d is None:
                buf.append(ch)
                continue
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(d)
            if ch.isascii():
                if "western_concat" not in forms:
                    forms.add("western_spaced")
            else:
                forms.add("native_numerals")
        if buf:
            out.append("".join(buf))
    return out, forms


def align(recognised: Sequence[int], triplet: Sequence[int]) -> tuple[bool, bool, bool]:
    """Levenshtein alignment of ``recognised`` (rows) against the 3 presented digits (§7.4)."""
    r = list(recognised)
    t = list(triplet)
    if len(t) != 3:
        raise ValueError("triplet must have exactly 3 digits")
    n = len(r)
    dp = [[0] * 4 for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(4):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, 4):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (r[i - 1] != t[j - 1]),
                dp[i][j - 1] + 1,
                dp[i - 1][j] + 1,
            )
    correct = [False, False, False]
    i, j = n, 3
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (r[i - 1] != t[j - 1]):
            correct[j - 1] = r[i - 1] == t[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            correct[j - 1] = False
            j -= 1
        else:
            i -= 1
    return (correct[0], correct[1], correct[2])


def keyterms_sha256(terms: Sequence[str]) -> str:
    """sha256 of the canonical JSON keyterm list (§7.6)."""
    payload = json.dumps(list(terms), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------------------------


class LexiconScorer:
    """Implements ``contracts.ports.AnswerScorer``. Immutable after ``load_lexicons``."""

    __slots__ = (
        "_digit_forms",
        "_fillers",
        "_keyterms",
        "_lmax",
        "_phrases",
        "_pieces",
        "_pieces_lmax",
    )

    def __init__(self, files: Mapping[Lang, LexiconFile]) -> None:
        phrases, digit_forms, fillers = _build_phrase_map(files)
        pieces = _build_pieces(files, phrases)
        self._phrases: Mapping[tuple[str, ...], _Meaning] = MappingProxyType(phrases)
        self._lmax = max(len(k) for k in phrases)
        self._pieces: Mapping[str, tuple[int, ...]] = MappingProxyType(pieces)
        self._pieces_lmax = max(len(k) for k in pieces)
        self._digit_forms: Mapping[int, tuple[str, ...]] = MappingProxyType(digit_forms)
        self._fillers = fillers
        self._keyterms = _build_keyterms(files)

    # ----- introspection used by tests and calibration -----

    @property
    def fillers(self) -> tuple[str, ...]:
        return self._fillers

    @property
    def sandhi_pairs(self) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType({k: (v[0], v[1]) for k, v in self._pieces.items() if len(v) == 2})

    def forms_all_langs(self, digit: int) -> tuple[str, ...]:
        """Every lexicon phrase (both files) that maps to ``digit``."""
        return self._digit_forms[digit]

    def meaning(self, phrase: str) -> tuple[str, int] | None:
        """(kind, value) of a lexicon phrase, or None if it is not in the lexicon."""
        m = self._phrases.get(_tokenise_phrase(phrase))
        return None if m is None else (m.kind.value, m.value)

    # ----- AnswerScorer -----

    def digits_so_far(self, lang: Lang, texts: Sequence[str]) -> int:
        return min(len(self.parse(texts).digits), MAX_RECOGNISED)

    def intent_so_far(self, lang: Lang, texts: Sequence[str]) -> Intent | None:
        p = self.parse(texts)
        return p.intent if p.digits == () else None

    def keyterms(self, lang: Lang, model_version: ModelVersion) -> tuple[str, ...]:
        base = self._keyterms[lang]
        if model_version is ModelVersion.V2_5:
            return tuple(p for p in base if _ASCII_LETTERS.fullmatch(p))[:KEYTERMS_V2_5_MAX]
        return base[:KEYTERMS_V3_MAX]

    def score(self, spec: PresentationSpec, evidence: AnswerEvidence) -> ScoringResult:
        if evidence.presentation_id != spec.presentation_id:
            raise ValueError(
                f"presentation_id mismatch: evidence {evidence.presentation_id} "
                f"!= spec {spec.presentation_id}"
            )
        triplet = tuple(spec.triplet)

        def finish(
            status: AnswerStatus,
            source: AnswerSource,
            digits: tuple[int, ...] = (),
            action: ScoreAction = ScoreAction.ACCEPT,
            prompt: PromptKey | None = None,
            number_form: bool = False,
            tokens: tuple[str, ...] = (),
        ) -> ScoringResult:
            return ScoringResult(
                presentation_id=evidence.presentation_id,
                mode=evidence.mode,
                status=status,
                action=action,
                reask_prompt=prompt,
                source=source,
                recognised=digits,
                triplet_correct=status is AnswerStatus.OK and digits == triplet,
                position_correct=align(digits, triplet),
                valid=status is not AnswerStatus.CAPTURE_MISS,
                no_response=status is AnswerStatus.NO_RESPONSE,
                number_form=number_form,
                normalised_tokens=tokens[:MAX_TOKENS],
            )

        # 1. DTMF wins.
        if evidence.dtmf:
            keys = tuple(k.key for k in evidence.dtmf[:3])
            dtmf_digits = tuple(int(k) for k in keys)
            status = AnswerStatus.OK if len(dtmf_digits) == 3 else AnswerStatus.PARTIAL
            return finish(status, AnswerSource.DTMF, dtmf_digits, tokens=keys)
        # 2. Keypad without keys.
        if evidence.mode is Mode.KEYPAD:
            return finish(AnswerStatus.NO_RESPONSE, AnswerSource.NONE)
        events = set(evidence.system_events)
        # 3. Capture failures.
        if events & {SystemEvent.BREAKER_TRIP, SystemEvent.RECONNECT, SystemEvent.MIC_LOST}:
            return finish(AnswerStatus.CAPTURE_MISS, AnswerSource.NONE)
        texts = [seg.text for seg in evidence.segments]
        all_empty = all(t.strip() == "" for t in texts)
        # 3b. STT error with speech energy and no text.
        if SystemEvent.STT_ERROR in events and all_empty and evidence.speech_present:
            return finish(AnswerStatus.CAPTURE_MISS, AnswerSource.NONE)
        # 4. Early answer.
        if evidence.early_speech and not evidence.echo_mode and not evidence.reask_used:
            return finish(
                AnswerStatus.EARLY_ANSWER,
                AnswerSource.NONE,
                action=ScoreAction.REASK,
                prompt=PromptKey.REASK_SAY_AGAIN,
            )
        # 5. Nothing transcribed.
        if all_empty:
            status = (
                AnswerStatus.CAPTURE_MISS if evidence.speech_present else AnswerStatus.NO_RESPONSE
            )
            return finish(status, AnswerSource.NONE)
        # 6. Parse.
        p = self.parse(texts)
        n = len(p.digits)
        digits: tuple[int, ...] = ()
        if n == 3:
            status, digits = AnswerStatus.OK, p.digits
        elif n > 3:
            status, digits = AnswerStatus.AMBIGUOUS, p.digits[:MAX_RECOGNISED]
        elif n > 0 and p.unparsed:
            status = AnswerStatus.NO_DIGITS
        elif n > 0:
            status, digits = AnswerStatus.PARTIAL, p.digits
        elif p.intent == "REPEAT":
            status = AnswerStatus.REPEAT_REQUEST
        elif p.intent == "DONT_KNOW":
            status = AnswerStatus.DONT_KNOW
        else:
            status = AnswerStatus.NO_DIGITS
        # 7. Action.
        action, prompt = ScoreAction.ACCEPT, None
        if status is AnswerStatus.REPEAT_REQUEST:
            if not evidence.replay_used:
                action = ScoreAction.REPLAY
            else:
                status = AnswerStatus.NO_DIGITS
        if status in (AnswerStatus.NO_DIGITS, AnswerStatus.AMBIGUOUS) and not evidence.reask_used:
            action, prompt = ScoreAction.REASK, PromptKey.REASK_THREE
        return finish(
            status,
            AnswerSource.SPEECH,
            digits,
            action=action,
            prompt=prompt,
            number_form=p.number_form,
            tokens=p.tokens,
        )

    # ----- normaliser (§7.3) -----

    def parse(self, texts: Sequence[str]) -> Parse:
        raw, forms = _split_numerals(_clean(" ".join(texts)))
        if not raw:
            return Parse((), None, False, False, ("none",), ())
        syms = self._symbolise(raw, forms)
        return self._grammar(syms, forms)

    def _symbolise(self, raw: list[str | int], forms: set[str]) -> list[_Sym]:
        syms: list[_Sym] = []
        i, n = 0, len(raw)
        while i < n:
            tok = raw[i]
            if isinstance(tok, int):
                syms.append(_Sym(_K.DIGIT, tok))
                i += 1
                continue
            matched = False
            for length in range(min(self._lmax, n - i), 0, -1):
                window = raw[i : i + length]
                if not all(isinstance(w, str) for w in window):
                    continue
                key = tuple(w for w in window if isinstance(w, str))
                m = self._phrases.get(key)
                if m is None:
                    continue
                if m.kind is not _K.FILLER:
                    syms.append(_Sym(m.kind, m.value))
                if m.form is not None:
                    forms.add(m.form)
                i += length
                matched = True
                break
            if matched:
                continue
            seg = self._segment(tok)
            if seg is None:
                syms.append(_Sym(_K.UNK))
            else:
                syms.extend(_Sym(_K.DIGIT, d) for d in seg)
                forms.add("native_words")
            i += 1
        return syms

    def _segment(self, tok: str) -> tuple[int, ...] | None:
        """Tamil fused-token DP (§7.3 step 5, C4): fewest pieces, ties -> longest first piece."""
        if len(tok) < 4 or not _is_tamil(tok):
            return None
        n = len(tok)
        # best[i] = (pieces, digits) for tok[i:], built from the end.
        best: list[tuple[int, tuple[int, ...]] | None] = [None] * (n + 1)
        best[n] = (0, ())
        for i in range(n - 1, -1, -1):
            chosen: tuple[int, tuple[int, ...]] | None = None
            for end in range(min(n, i + self._pieces_lmax), i, -1):  # longest piece first
                piece = self._pieces.get(tok[i:end])
                rest = best[end]
                if piece is None or rest is None:
                    continue
                cand = (rest[0] + 1, piece + rest[1])
                if chosen is None or cand[0] < chosen[0]:
                    chosen = cand
            best[i] = chosen
        result = best[0]
        if result is None or len(result[1]) < 2:
            return None
        return result[1]

    def _grammar(self, syms: list[_Sym], forms: set[str]) -> Parse:
        out: list[int] = []
        tokens: list[str] = []
        number_form = False
        unparsed = False
        has_unk = False
        unconsumed = False
        seen_repeat = False
        seen_dk = False

        def emit(*ds: int) -> None:
            out.extend(ds)
            tokens.extend(str(d) for d in ds)

        i, n = 0, len(syms)
        while i < n:
            s = syms[i]
            nxt = syms[i + 1] if i + 1 < n else None
            # (a0) C5 repetition.
            if s.kind is _K.REP:
                if nxt is not None and nxt.kind is _K.DIGIT:
                    emit(*([nxt.value] * s.value))
                    i += 2
                else:
                    has_unk = True
                    tokens.append("UNK")
                    i += 1
                continue
            # (a) hundred compound.
            hundred: int | None = None
            j = i + 1
            if s.kind is _K.HUNDREDS:
                hundred = s.value
            elif s.kind is _K.DIGIT and s.value >= 1 and nxt is not None and nxt.kind is _K.MULT:
                hundred, j = 100 * s.value, i + 2
            if hundred is not None:
                if j < n and syms[j].kind is _K.UNK:
                    unparsed = True
                    tokens.append("UNK")
                    i = j + 1
                    continue
                value = hundred
                if j < n:
                    t = syms[j]
                    if t.kind is _K.TENS:
                        value += t.value
                        j += 1
                        if j < n and syms[j].kind is _K.DIGIT and syms[j].value >= 1:
                            value += syms[j].value
                            j += 1
                    elif t.kind in (_K.TEENS, _K.NUM2) or (t.kind is _K.DIGIT and t.value >= 1):
                        value += t.value
                        j += 1
                emit(value // 100, value // 10 % 10, value % 10)
                number_form = True
                forms.add("compound")
                i = j
                continue
            if s.kind is _K.DIGIT:
                # (b) C2 grouping.
                after = syms[i + 2] if i + 2 < n else None
                if (
                    nxt is not None
                    and nxt.kind is _K.TENS
                    and after is not None
                    and after.kind is _K.DIGIT
                ):
                    emit(s.value, nxt.value // 10, after.value)
                    number_form = True
                    forms.add("grouping")
                    i += 3
                    continue
                if nxt is not None and nxt.kind is _K.TEENS:
                    emit(s.value, 1, nxt.value - 10)
                    number_form = True
                    forms.add("grouping")
                    i += 2
                    continue
                if nxt is not None and nxt.kind is _K.NUM2 and nxt.value % 10 != 0:
                    emit(s.value, nxt.value // 10, nxt.value % 10)
                    number_form = True
                    forms.add("grouping")
                    i += 2
                    continue
                # (c) plain digit.
                emit(s.value)
            elif s.kind in (_K.TENS, _K.TEENS, _K.NUM2, _K.MULT):
                unconsumed = True
            elif s.kind is _K.UNK:
                has_unk = True
                tokens.append("UNK")
            elif s.kind is _K.REPEAT:
                seen_repeat = True
                tokens.append("REPEAT")
            elif s.kind is _K.DK:
                seen_dk = True
                tokens.append("DONT_KNOW")
            i += 1

        if 1 <= len(out) <= 2 and (has_unk or unconsumed):
            unparsed = True
        if unparsed:
            forms.add("unparsed")
        intent: Intent | None = "REPEAT" if seen_repeat else ("DONT_KNOW" if seen_dk else None)
        return Parse(
            digits=tuple(out),
            intent=intent,
            number_form=number_form,
            unparsed=unparsed,
            forms=tuple(sorted(forms)),
            tokens=tuple(tokens[:MAX_TOKENS]),
        )


# ---------------------------------------------------------------------------------------------
# Construction (I/O allowed here only)
# ---------------------------------------------------------------------------------------------


def _digit_form_label(lf: LexiconFile, phrase: str) -> str:
    if lf.lang is Lang.EN:
        return "english_words"
    if phrase in lf.english_in_tamil_script:
        return "english_in_tamil_script"
    return "native_words" if _has_tamil(phrase) else "tanglish"


def _build_phrase_map(
    files: Mapping[Lang, LexiconFile],
) -> tuple[dict[tuple[str, ...], _Meaning], dict[int, tuple[str, ...]], tuple[str, ...]]:
    phrases: dict[tuple[str, ...], _Meaning] = {}
    origin: dict[tuple[str, ...], str] = {}
    digit_forms: dict[int, list[str]] = {d: [] for d in range(10)}
    fillers: list[str] = []

    def add(lf: LexiconFile, phrase: str, meaning: _Meaning) -> None:
        if unicodedata.normalize("NFC", phrase) != phrase:
            raise ValueError(f"{lf.lang}.yaml: phrase {phrase!r} is not NFC")
        if any(unicodedata.decimal(c, None) is not None for c in phrase):
            raise ValueError(f"{lf.lang}.yaml: phrase {phrase!r} contains a numeral")
        key = _tokenise_phrase(phrase)
        if not key or len(key) > MAX_PHRASE_TOKENS:
            raise ValueError(f"{lf.lang}.yaml: phrase {phrase!r} must have 1..4 tokens")
        prev = phrases.get(key)
        if prev is not None and (prev.kind, prev.value) != (meaning.kind, meaning.value):
            raise ValueError(
                f"lexicon collision: {phrase!r} ({lf.lang}.yaml {meaning.kind.value}"
                f"={meaning.value}) vs {origin[key]} ({prev.kind.value}={prev.value})"
            )
        if prev is None:
            phrases[key] = meaning
            origin[key] = f"{lf.lang}.yaml"

    for lang in (Lang.TA, Lang.EN):
        lf = files[lang]
        if lf.lang is not lang:
            raise ValueError(f"{lang}.yaml declares lang {lf.lang}")
        if set(lf.digits) != {str(d) for d in range(10)}:
            raise ValueError(f"{lang}.yaml: digits must have keys '0'..'9'")
        for d_str, plist in lf.digits.items():
            if not plist:
                raise ValueError(f"{lang}.yaml: digit {d_str} has no form")
            for p in plist:
                add(lf, p, _Meaning(_K.DIGIT, int(d_str), _digit_form_label(lf, p)))
                if p not in digit_forms[int(d_str)]:
                    digit_forms[int(d_str)].append(p)
        for p in lf.english_in_tamil_script:
            if lang is not Lang.TA or not _has_tamil(p):
                raise ValueError(f"{lang}.yaml: english_in_tamil_script entry {p!r} invalid")
        for p in lf.intents.DONT_KNOW:
            add(lf, p, _Meaning(_K.DK))
        for p in lf.intents.REPEAT:
            add(lf, p, _Meaning(_K.REPEAT))
        for p in lf.fillers:
            add(lf, p, _Meaning(_K.FILLER))
            if p not in fillers:
                fillers.append(p)
        for p, k in lf.repeat.items():
            if k not in (2, 3):
                raise ValueError(f"{lang}.yaml: repeat {p!r} must be 2 or 3")
            form = "english_in_tamil_script" if _has_tamil(p) else "english_words"
            add(lf, p, _Meaning(_K.REP, k, form))
        c = lf.compound
        for p, v in c.hundreds.items():
            if v not in range(100, 1000, 100):
                raise ValueError(f"{lang}.yaml: hundreds {p!r}={v} not in 100..900")
            add(lf, p, _Meaning(_K.HUNDREDS, v))
        for p in c.hundred_multiplier:
            add(lf, p, _Meaning(_K.MULT, 100))
        for p, v in c.tens.items():
            if v not in range(20, 100, 10):
                raise ValueError(f"{lang}.yaml: tens {p!r}={v} not in 20..90")
            add(lf, p, _Meaning(_K.TENS, v))
        for p, v in c.teens.items():
            if not 10 <= v <= 19:
                raise ValueError(f"{lang}.yaml: teens {p!r}={v} not in 10..19")
            add(lf, p, _Meaning(_K.TEENS, v))
        for p, v in c.num2.items():
            if not 11 <= v <= 99:
                raise ValueError(f"{lang}.yaml: num2 {p!r}={v} not in 11..99")
            add(lf, p, _Meaning(_K.NUM2, v))

    en = files[Lang.EN]
    if en.compound.num2:
        raise ValueError("en.yaml must not define num2 (contracts §4.13 C2)")
    en_all = [p for plist in en.digits.values() for p in plist] + list(en.fillers)
    if any(_DEVANAGARI.search(p) for p in en_all):
        raise ValueError("en.yaml must not contain Devanagari")
    return phrases, {d: tuple(v) for d, v in digit_forms.items()}, tuple(fillers)


def _build_pieces(
    files: Mapping[Lang, LexiconFile], phrases: Mapping[tuple[str, ...], _Meaning]
) -> dict[str, tuple[int, ...]]:
    """Native-script DIGIT pieces plus the precomputed C4 sandhi pairs."""
    native: dict[str, int] = {}
    for lf in files.values():
        for d_str, plist in lf.digits.items():
            for p in plist:
                if _is_tamil(p):
                    native[p] = int(d_str)
    pieces: dict[str, tuple[int, ...]] = {p: (d,) for p, d in native.items()}
    for w1, d1 in native.items():
        if len(w1) < 2 or w1[-1] != _TAMIL_U_SIGN or w1[-2] not in _TAMIL_CONSONANTS:
            continue
        for w2, d2 in native.items():
            sign = _VOWEL_SIGN.get(w2[0])
            if sign is None:
                continue
            fused = w1[:-1] + sign + w2[1:]
            if (fused,) in phrases:
                raise ValueError(f"C4 sandhi form {fused!r} collides with a lexicon phrase")
            prev = pieces.get(fused)
            if prev is not None and prev != (d1, d2):
                raise ValueError(f"C4 sandhi form {fused!r} is ambiguous: {prev} vs {(d1, d2)}")
            pieces[fused] = (d1, d2)
    return pieces


def _build_keyterms(files: Mapping[Lang, LexiconFile]) -> Mapping[Lang, tuple[str, ...]]:
    def digit_words(lf: LexiconFile) -> list[str]:
        return [p for d in range(10) for p in lf.digits[str(d)]]

    def dedupe(seq: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(seq))

    en_words = digit_words(files[Lang.EN])
    return MappingProxyType(
        {
            Lang.TA: dedupe(digit_words(files[Lang.TA]) + en_words),
            Lang.EN: dedupe(en_words),
        }
    )


def _read_file(path: Path) -> LexiconFile:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read lexicon {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"lexicon {path} is not a mapping")
    try:
        return LexiconFile.model_validate(data, strict=False)
    except ValidationError as exc:
        raise ValueError(f"lexicon {path} fails schema {LEXICON_SCHEMA_ID}: {exc}") from exc


def load_lexicons(lexicon_dir: str | None = None) -> LexiconScorer:
    """Read and validate ``ta.yaml`` and ``en.yaml``; raise ``ValueError`` on any violation."""
    base = Path(lexicon_dir) if lexicon_dir is not None else _DEFAULT_DIR
    files = {lang: _read_file(base / f"{lang.value}.yaml") for lang in (Lang.TA, Lang.EN)}
    return LexiconScorer(files)
