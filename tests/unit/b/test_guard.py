# ruff: noqa: RUF001  (attack strings deliberately use lookalike letters)
"""Guard unit tests (B spec §13.5 U-GD-*, contracts §8.5.4)."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
import yaml

from sevi.contracts.enums import Lang
from sevi.core import guard
from sevi.core.guard import GuardReason as R

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "evon"
LEX = Path(__file__).resolve().parents[3] / "src" / "sevi" / "core" / "lexicon"
GUARD_DIR = Path(__file__).resolve().parents[3] / "src" / "sevi" / "core"
EN_OK = "Your hearing in background noise looks typical today, please keep listening well."
TA_OK = "சத்தமான இடத்தில் உங்கள் கேட்கும் திறன் இன்று சாதாரணமாக தெரிகிறது."
# Documented exclusions from en.txt: lexicon homophones of English function words.
EN_EXCLUDED = {"to", "too", "for", "fore", "ate", "tree"}


def reasons(text: str, lang: Lang = Lang.EN, purpose: str = "EXPLAIN_SPOKEN", banned=()):
    return guard.check(text, lang, purpose, banned).reasons  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "lang", "purpose", "reason"),
    [
        ("  ", Lang.EN, "EXPLAIN_SPOKEN", R.EMPTY),
        ("a" * 301, Lang.EN, "EXPLAIN_SPOKEN", R.LENGTH),
        ("a" * 601, Lang.EN, "REPORT_PROSE", R.LENGTH),
        (TA_OK + " ௨", Lang.TA, "EXPLAIN_SPOKEN", R.NUMERAL),
        (TA_OK + " ஏழு", Lang.TA, "EXPLAIN_SPOKEN", R.NUMBER_WORD),
        (EN_OK + " db", Lang.EN, "EXPLAIN_SPOKEN", R.UNIT),
        (EN_OK + " %", Lang.EN, "EXPLAIN_SPOKEN", R.UNIT),
        (EN_OK + " get a hearing aid", Lang.EN, "EXPLAIN_SPOKEN", R.MEDICAL),
        (EN_OK, Lang.TA, "EXPLAIN_SPOKEN", R.SCRIPT),
        (TA_OK, Lang.EN, "EXPLAIN_SPOKEN", R.SCRIPT),
    ],
)
def test_U_GD_01_one_example_per_reason(text, lang, purpose, reason) -> None:
    assert reason in reasons(text, lang, purpose)


def test_U_GD_01_banned_phrase() -> None:
    assert reasons(EN_OK + " It is Accurate.", banned=["accurate"]) == (R.BANNED,)
    assert reasons(EN_OK) == ()


def test_U_GD_01_length_boundaries() -> None:
    assert R.LENGTH not in reasons("a" * 300)
    assert R.LENGTH not in reasons("a" * 600, purpose="REPORT_PROSE")


def test_U_GD_01_tamil_number_signs() -> None:
    for c in "௰௱௲":
        assert R.NUMERAL in reasons(TA_OK + c, Lang.TA)


def test_U_GD_02_agglutinated_tamil_and_latin_whole_token() -> None:
    assert R.NUMBER_WORD in reasons(TA_OK + " இரண்டாவது", Lang.TA)
    assert R.NUMBER_WORD in reasons(TA_OK + " ஒருவர்", Lang.TA)
    for word in ("someone", "often", "tone", "weight", "phone"):
        assert R.NUMBER_WORD not in reasons(f"{EN_OK} {word}"), word


def test_U_GD_03_script_threshold() -> None:
    ta20 = "அ" * 79 + "a" * 21
    ta80 = "அ" * 80 + "a" * 20
    assert R.SCRIPT in reasons(ta20, Lang.TA)
    assert R.SCRIPT not in reasons(ta80, Lang.TA)


@pytest.mark.parametrize(("name", "lang"), [("good_en.txt", Lang.EN), ("good_ta.txt", Lang.TA)])
def test_U_GD_04_good_samples_pass(name: str, lang: Lang) -> None:
    v = guard.check((FIX / name).read_text(encoding="utf-8"), lang, "EXPLAIN_SPOKEN")
    assert v.ok, v.reasons


def test_U_GD_05_reasons_in_enum_order() -> None:
    text = "a" * 290 + " 5 seven db deaf அஅஅஅஅஅஅஅ accurate"
    got = reasons(text, Lang.TA, banned=["accurate"])
    assert got == (R.LENGTH, R.NUMERAL, R.NUMBER_WORD, R.UNIT, R.MEDICAL, R.SCRIPT, R.BANNED)


@pytest.mark.parametrize(
    "name", ["guard_numbers/ta.txt", "guard_numbers/en.txt", "guard_medical.txt"]
)
def test_U_GD_06_list_files_wellformed(name: str) -> None:
    lines = (GUARD_DIR / name).read_text(encoding="utf-8").splitlines()
    entries = [ln for ln in lines if not ln.startswith("#")]
    assert all(e.strip() for e in entries), "no empty lines"
    assert all(unicodedata.normalize("NFC", e) == e for e in entries)
    assert all(e == e.strip() for e in entries)
    assert len(entries) == len(set(entries)), "no duplicates"


@pytest.mark.parametrize("lang", ["ta", "en"])
def test_U_GD_06_every_lexicon_digit_form_present(lang: str) -> None:
    lex = yaml.safe_load((LEX / f"{lang}.yaml").read_text(encoding="utf-8"))
    listed = set(guard.read_word_list(f"guard_numbers/{lang}.txt"))
    forms = {w for ws in lex["digits"].values() for w in ws} | set(lex["english_in_tamil_script"])
    forms = {unicodedata.normalize("NFC", f).casefold() for f in forms} - EN_EXCLUDED
    assert forms <= listed, forms - listed


def test_U_GD_07_indian_english_and_double_triple() -> None:
    for t in ("double five", "triple", "tu", "nain", "one and a half"):
        assert R.NUMBER_WORD in reasons(f"{EN_OK} {t}"), t
    assert R.NUMBER_WORD in reasons(TA_OK + " டபுள்", Lang.TA)
    assert R.NUMBER_WORD in reasons(EN_OK + " oru")


def test_guard_tokenizer_apostrophe_inside_latin_only() -> None:
    assert guard.tokenize("don't stop") == ["don't", "stop"]
    assert guard.tokenize("'quoted'") == ["quoted"]


# ---------- review #36 bypass classes: one test per attack ----------
@pytest.mark.parametrize(
    ("text", "lang"),
    [
        (EN_OK + " fi​ve", Lang.EN),  # ZWSP
        (EN_OK + " fi­ve", Lang.EN),  # soft hyphen
        (EN_OK + " fi⁠ve", Lang.EN),  # word joiner
        (TA_OK + " ஐ‍ந்து", Lang.TA),  # ZWJ
        (TA_OK + " மூ‌ன்று", Lang.TA),  # ZWNJ
    ],
)
def test_U_GD_08_format_chars_stripped(text: str, lang: Lang) -> None:
    assert R.NUMBER_WORD in reasons(text, lang)


def test_U_GD_08_nfkc_compatibility_forms() -> None:
    assert R.NUMBER_WORD in reasons(EN_OK + " ｆｉｖｅ")  # fullwidth five
    assert R.NUMBER_WORD in reasons(EN_OK + " \U0001d42d\U0001d430\U0001d428")  # math-bold two
    assert R.NUMERAL in reasons(EN_OK + " ⑤")  # circled five


@pytest.mark.parametrize(
    "text",
    [
        "You got twо right today, your result looks typical.",  # Cyrillic o
        "You got ѕix right today, your result looks typical.",  # Cyrillic dze
        "You got nιne right today, your result looks typical.",  # Greek iota
    ],
)
def test_U_GD_09_homoglyphs(text: str) -> None:
    got = reasons(text)
    assert R.NUMBER_WORD in got
    assert R.SCRIPT in got


def test_U_GD_09_any_foreign_letter_fails_script() -> None:
    assert R.SCRIPT in reasons(EN_OK + " а")  # one Cyrillic a in 80 Latin letters
    assert R.SCRIPT in reasons(TA_OK + " а", Lang.TA)
    assert R.SCRIPT not in reasons(TA_OK + " test", Lang.TA)  # Latin allowed in ta


@pytest.mark.parametrize(
    "word",
    [
        "couple",
        "pair",
        "single",
        "dozen",
        "half",
        "quarter",
        "sixes",
        "nineties",
        "twentieth",
        "twelfth",
        "hundredth",
        "fivefold",
        "tenfold",
        "ninths",
        "ones",
        "iv",
        "xii",
        "XII",
        "vii",
        "ix",
    ],
)
def test_U_GD_10_number_meaning_words(word: str) -> None:
    assert R.NUMBER_WORD in reasons(f"{EN_OK} {word}"), word


@pytest.mark.parametrize("word", ["I", "i", "mix", "mid", "dim", "civil", "wait", "with"])
def test_U_GD_10_roman_and_suffix_no_false_positive(word: str) -> None:
    assert R.NUMBER_WORD not in reasons(f"{EN_OK} {word}"), word


@pytest.mark.parametrize(
    "word",
    ["பாதி", "பத்தில்", "பத்தை", "மூன்றை", "இரண்டில்", "நான்கை", "ஐந்தில்", "எட்டில்", "ஒன்பதில்", "ஆறில்", "ஏழில்"],
)
def test_U_GD_10_tamil_inflected_number_words(word: str) -> None:
    assert R.NUMBER_WORD in reasons(f"{TA_OK} {word}", Lang.TA), word


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Your hearing is perfect.", Lang.EN),
        ("Your ears are healthy.", Lang.EN),
        ("you are normal", Lang.EN),
        ("Nothing to worry about.", Lang.EN),
        ("This proves your hearing is good.", Lang.EN),
        ("Your audiogram looks normal.", Lang.EN),
        ("Good hearing overall, see you.", Lang.EN),
        ("There is no need to see an ENT.", Lang.EN),
        ("Signs of presbycusis are absent.", Lang.EN),
        ("உங்கள் காது நன்றாக உள்ளது", Lang.TA),
        ("கவலைப்பட வேண்டாம், எல்லாம் சரி", Lang.TA),
    ],
)
def test_U_GD_11_reassurance(text: str, lang: Lang) -> None:
    assert R.REASSURANCE in reasons(text, lang)


def test_U_GD_11_ent_whole_word_only() -> None:
    assert R.REASSURANCE not in reasons("Your result is consistent with this screening moment.")
