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
