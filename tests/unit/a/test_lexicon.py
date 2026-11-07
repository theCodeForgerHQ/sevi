"""Unit tests for core/lexicon/{ta,en}.yaml and load_lexicons (spec §7.2, §7.6, §7.7)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from sevi.contracts.enums import Lang, ModelVersion
from sevi.core.scoring import keyterms_sha256, load_lexicons

LEX = load_lexicons()
LEX_DIR = Path(__file__).resolve().parents[3] / "src" / "sevi" / "core" / "lexicon"

# contracts §4.13 normative minimum (every token alone -> its meaning).
DIGIT_TOKENS: dict[int, list[str]] = {
    0: ["பூஜ்ஜியம்", "சைபர்", "ஜீரோ", "poojyam", "pujyam", "saibar", "siphar", "zero", "sero"],
    1: ["ஒன்று", "ஒண்ணு", "ஒன்னு", "ஒன்", "onru", "ondru", "onnu", "onu", "one", "won", "wan"],
    2: [
        "இரண்டு",
        "ரெண்டு",
        "இரெண்டு",
        "ரண்டு",
        "இரன்டு",
        "டூ",
        "irandu",
        "irendu",
        "rendu",
        "randu",
        "erandu",
        "rentu",
        "two",
        "too",
        "to",
        "tu",
    ],
    3: [
        "மூன்று",
        "மூணு",
        "மூனு",
        "த்ரீ",
        "moonru",
        "moondru",
        "munru",
        "moonu",
        "munu",
        "three",
        "tree",
        "thri",
    ],
    4: [
        "நான்கு",
        "நாலு",
        "நான்கூ",
        "ஃபோர்",
        "போர்",
        "naangu",
        "nangu",
        "naalu",
        "nalu",
        "naanku",
        "nanku",
        "four",
        "for",
        "fore",
        "foar",
    ],
    5: [
        "ஐந்து",
        "அஞ்சு",
        "ஃபைவ்",
        "பைவ்",
        "ainthu",
        "aindhu",
        "ainju",
        "anju",
        "anchu",
        "aindu",
        "anji",
        "five",
        "fiv",
        "faiv",
    ],
    6: ["ஆறு", "ஆரு", "சிக்ஸ்", "aaru", "aru", "six", "siks", "sicks"],
    7: [
        "ஏழு",
        "ஏழ்",
        "எழு",
        "செவன்",
        "ezhu",
        "ezh",
        "elu",
        "yezhu",
        "ezu",
        "yelu",
        "eazhu",
        "seven",
        "sevan",
        "sevun",
    ],
    8: ["எட்டு", "எய்ட்", "ettu", "yettu", "etu", "eattu", "eight", "ate", "eit", "ait"],
    9: [
        "ஒன்பது",
        "ஒம்போது",
        "நைன்",
        "onbathu",
        "onbadhu",
        "ombothu",
        "ombodhu",
        "nine",
        "nain",
        "nyne",
    ],
}
DK_TOKENS = [
    "தெரியல",
    "தெரியலை",
    "தெரியாது",
    "கேக்கல",
    "கேட்கல",
    "புரியல",
    "ஒண்ணும் கேக்கல",
    "theriyala",
    "theriyalai",
    "theriyadhu",
    "kekkala",
    "puriyala",
    "onnum kekkala",
    "kekalai",
    "kekkalai",
    "puriyalai",
    "don't know",
    "dont know",
    "didn't hear",
    "no idea",
    "not sure",
    "can't hear",
    "cannot hear",
    "not clear",
    "do not know",
    "couldn't hear",
    "could not hear",
    "nothing",
]
REPEAT_TOKENS = [
    "மறுபடியும்",
    "திரும்ப",
    "இன்னொரு தடவை",
    "திரும்பவும்",
    "marupadiyum",
    "thirumba",
    "innoru thadavai",
    "thirumbavum",
    "innoru thadava",
    "repeat",
    "again",
    "once more",
    "once again",
    "say again",
    "come again",
    "pardon",
    "one more time",
    "repeat please",
    "say it again",
]
FILLER_TOKENS = [
    "ம்",
    "அ",
    "அப்புறம்",
    "சரி",
    "நம்பர்",
    "mm",
    "aa",
    "appuram",
    "sari",
    "number",
    "numbers",
    "ah",
    "uh",
    "enna",
    "ம்ம்",
    "ஆ",
    "apram",
    "um",
    "oh",
    "ok",
    "okay",
    "the",
    "is",
    "and",
    "it's",
    "er",
    "so",
    "it",
    "was",
    "please",
]


@pytest.mark.parametrize(("digit", "token"), [(d, t) for d, ts in DIGIT_TOKENS.items() for t in ts])
def test_U_LEX_01_digit_tokens(digit: int, token: str) -> None:
    assert LEX.parse([token]).digits == (digit,)


@pytest.mark.parametrize("token", DK_TOKENS)
def test_U_LEX_01_dont_know_tokens(token: str) -> None:
    p = LEX.parse([token])
    assert (p.digits, p.intent) == ((), "DONT_KNOW")


@pytest.mark.parametrize("token", REPEAT_TOKENS)
def test_U_LEX_01_repeat_tokens(token: str) -> None:
    p = LEX.parse([token])
    assert (p.digits, p.intent) == ((), "REPEAT")


@pytest.mark.parametrize("token", FILLER_TOKENS)
def test_U_LEX_01_filler_tokens(token: str) -> None:
    p = LEX.parse([token])
    assert (p.digits, p.intent, p.tokens) == ((), None, ())


def test_U_LEX_01_dont_know_not_two_and_oh_filler() -> None:
    assert LEX.parse(["don't know"]).digits == ()
    assert LEX.meaning("oh") == ("FILLER", 0)


@pytest.mark.parametrize(
    ("phrase", "meaning"),
    [
        ("to", ("DIGIT", 2)),
        ("too", ("DIGIT", 2)),
        ("for", ("DIGIT", 4)),
        ("ate", ("DIGIT", 8)),
        ("won", ("DIGIT", 1)),
        ("tree", ("DIGIT", 3)),
        ("போர்", ("DIGIT", 4)),
        ("one", ("DIGIT", 1)),
        ("one more time", ("REPEAT", 0)),
        ("double", ("REP", 2)),
        ("triple", ("REP", 3)),
        ("டபுள்", ("REP", 2)),
    ],
)
def test_U_LEX_02_explicit_collision_list(phrase: str, meaning: tuple[str, int]) -> None:
    assert LEX.meaning(phrase) == meaning


def test_U_LEX_02_files_validate_and_cover_every_digit() -> None:
    for lang in ("ta", "en"):
        data = yaml.safe_load((LEX_DIR / f"{lang}.yaml").read_text(encoding="utf-8"))
        assert data["schema_id"] == "sevi.lexicon/2"
        assert sorted(data["digits"]) == [str(d) for d in range(10)]
        assert all(data["digits"][str(d)] for d in range(10))
    en = yaml.safe_load((LEX_DIR / "en.yaml").read_text(encoding="utf-8"))
    assert en["compound"]["num2"] == {}
    assert not re.search(r"[ऀ-ॿ]", (LEX_DIR / "en.yaml").read_text(encoding="utf-8"))


def _copy_lex(tmp_path: Path) -> Path:
    for f in ("ta.yaml", "en.yaml"):
        shutil.copy(LEX_DIR / f, tmp_path / f)
    return tmp_path


def _edit(path: Path, fn: object) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)  # type: ignore[operator]
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


@pytest.mark.parametrize(
    ("file", "mutate", "match"),
    [
        ("en.yaml", lambda d: d["digits"]["3"].append("for"), "collision"),
        ("ta.yaml", lambda d: d["fillers"].append("rendu"), "collision"),
        ("en.yaml", lambda d: d["compound"]["num2"].update({"seventytwo": 72}), "num2"),
        ("en.yaml", lambda d: d["digits"].pop("9"), "digits"),
        ("en.yaml", lambda d: d["digits"]["9"].clear(), "no form"),
        ("en.yaml", lambda d: d.update(schema_id="kj.lexicon/1"), "schema"),
        ("en.yaml", lambda d: d.update(extra=1), "schema"),
        ("en.yaml", lambda d: d["fillers"].append("एक"), "Devanagari"),
        ("en.yaml", lambda d: d["repeat"].update({"quadruple": 4}), "repeat"),
        ("ta.yaml", lambda d: d["compound"]["tens"].update({"x": 25}), "tens"),
        ("ta.yaml", lambda d: d["fillers"].append("நாலேழு"), "sandhi"),
        ("en.yaml", lambda d: d["fillers"].append("a b c d e"), "1..4 tokens"),
        ("en.yaml", lambda d: d["fillers"].append("4u"), "numeral"),
    ],
)
def test_U_LEX_02_loader_rejects_invalid_files(
    tmp_path: Path, file: str, mutate: object, match: str
) -> None:
    d = _copy_lex(tmp_path)
    _edit(d / file, mutate)
    with pytest.raises(ValueError, match=match):
        load_lexicons(str(d))


def test_U_LEX_02_missing_file(tmp_path: Path) -> None:
    shutil.copy(LEX_DIR / "ta.yaml", tmp_path / "ta.yaml")
    with pytest.raises(ValueError, match="cannot read"):
        load_lexicons(str(tmp_path))


def test_U_LEX_03_keyterms() -> None:
    v3 = LEX.keyterms(Lang.TA, ModelVersion.V3)
    assert len(v3) <= 200
    assert v3[:5] == ("பூஜ்ஜியம்", "சைபர்", "ஜீரோ", "poojyam", "pujyam")
    assert len(set(v3)) == len(v3)
    assert "four" in v3
    v25 = LEX.keyterms(Lang.TA, ModelVersion.V2_5)
    assert len(v25) <= 100
    assert all(re.fullmatch(r"[a-z]+", p) for p in v25)
    en = LEX.keyterms(Lang.EN, ModelVersion.V3)
    assert en[:3] == ("zero", "sero", "one")
    assert not any(re.search(r"[஀-௿]", p) for p in en)
    again = load_lexicons()
    for lang in Lang:
        for mv in ModelVersion:
            assert again.keyterms(lang, mv) == LEX.keyterms(lang, mv)
    assert keyterms_sha256(v3) == keyterms_sha256(again.keyterms(Lang.TA, ModelVersion.V3))
    assert re.fullmatch(r"[0-9a-f]{64}", keyterms_sha256(v3))


@pytest.mark.parametrize(
    ("fused", "pair"), [("நாலேழு", (4, 7)), ("ரெண்டெட்டு", (2, 8)), ("நாலஞ்சு", (4, 5))]
)
def test_U_LEX_04_sandhi_pairs(fused: str, pair: tuple[int, int]) -> None:
    assert LEX.sandhi_pairs[fused] == pair
    assert LEX.parse([fused]).digits == pair


def test_U_LEX_04_no_sandhi_collides_with_phrase() -> None:
    for fused in LEX.sandhi_pairs:
        assert LEX.meaning(fused) is None
