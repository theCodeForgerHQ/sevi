"""Guard property tests P-GD-01..03 (B spec §13.5)."""

from __future__ import annotations

import unicodedata

from hypothesis import given, settings
from hypothesis import strategies as st

from sevi.contracts.enums import Lang
from sevi.core import guard
from sevi.core.guard import GuardReason as R

TA_LETTERS = [chr(c) for c in range(0x0B85, 0x0BB9) if unicodedata.category(chr(c)) == "Lo"]
LATIN = list("abcdefghijklmnopqrstuvwxyz")
ENTRIES = [" ".join(seq) for seq in guard.number_entries()]
# Numerals in every script the guard must catch, incl. Tamil/Devanagari/Kannada/fullwidth.
NUMERAL_CHARS = (
    [chr(c) for c in range(0x30, 0x3A)]
    + [chr(c) for c in range(0x0BE6, 0x0BF3)]
    + [chr(c) for c in range(0x0966, 0x0970)]
    + [chr(c) for c in range(0x0CE6, 0x0CF0)]
    + [chr(c) for c in range(0xFF10, 0xFF1A)]
    + ["½", "²", "¼"]
)
LANGS = st.sampled_from([Lang.TA, Lang.EN])


def words(alphabet: list[str]) -> st.SearchStrategy[str]:
    return st.text(st.sampled_from(alphabet), min_size=1, max_size=8)


texts = st.lists(
    st.one_of(
        words(TA_LETTERS), words(LATIN), st.sampled_from(ENTRIES), st.sampled_from(NUMERAL_CHARS)
    ),
    min_size=1,
    max_size=12,
).map(" ".join)


def _entry_in(text: str) -> bool:
    toks = guard.tokenize(unicodedata.normalize("NFC", text).casefold())
    return guard._has_number_word(toks)


@settings(max_examples=300, deadline=None)
@given(texts, LANGS)
def test_P_GD_01_ok_implies_no_numeral_and_no_entry(text: str, lang: Lang) -> None:
    v = guard.check(text, lang, "REPORT_PROSE")
    if v.ok:
        assert not any(unicodedata.category(c) in ("Nd", "No") for c in text)
        assert not _entry_in(text)


clean = st.lists(st.one_of(words(TA_LETTERS), words(LATIN)), min_size=1, max_size=10).map(" ".join)


@settings(max_examples=300, deadline=None)
@given(clean, LANGS, st.sampled_from(NUMERAL_CHARS), st.data())
def test_P_GD_02_injected_numeral_rejected(text: str, lang: Lang, num: str, data) -> None:
    words_ = text.split(" ")
    i = data.draw(st.integers(0, len(words_)))
    injected = " ".join([*words_[:i], num, *words_[i:]])
    assert R.NUMERAL in guard.check(injected, lang, "REPORT_PROSE").reasons


@settings(max_examples=300, deadline=None)
@given(clean, LANGS, st.sampled_from(ENTRIES), st.data())
def test_P_GD_02_injected_number_word_rejected(text: str, lang: Lang, entry: str, data) -> None:
    words_ = text.split(" ")
    i = data.draw(st.integers(0, len(words_)))
    injected = " ".join([*words_[:i], entry, *words_[i:]])
    assert R.NUMBER_WORD in guard.check(injected, lang, "REPORT_PROSE").reasons


@settings(max_examples=200, deadline=None)
@given(texts, LANGS)
def test_P_GD_03_nfd_and_case_invariant(text: str, lang: Lang) -> None:
    base = guard.check(unicodedata.normalize("NFC", text), lang, "REPORT_PROSE")
    assert guard.check(unicodedata.normalize("NFD", text), lang, "REPORT_PROSE") == base
    assert guard.check(text.upper(), lang, "REPORT_PROSE") == base


CF_CHARS = ["​", "‌", "‍", "­", "⁠", "﻿"]


@settings(max_examples=300, deadline=None)
@given(clean, LANGS, st.sampled_from(ENTRIES), st.sampled_from(CF_CHARS), st.data())
def test_P_GD_02_number_word_split_by_format_char_rejected(
    text: str, lang: Lang, entry: str, cf: str, data
) -> None:
    j = data.draw(st.integers(0, len(entry)))
    hidden = entry[:j] + cf + entry[j:]
    assert R.NUMBER_WORD in guard.check(f"{text} {hidden}", lang, "REPORT_PROSE").reasons
