"""EXP-4 offline scoring accuracy (issue #61): the lexicon-generated transcript corpus is a
regression suite for ``core.scoring``. No live calls."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from sevi.calibration import scoring_accuracy as sa
from sevi.contracts.enums import Lang
from sevi.core.scoring import load_lexicons

CORPUS = sa.build_corpus()
SCORER = load_lexicons()
KNOWN_GAPS = {"extra_homophone", "wrong_language_hindi"}
GROUPS: dict[tuple[str, str], list[sa.Case]] = defaultdict(list)
for _c in CORPUS:
    GROUPS[(_c.lang.value, _c.form)].append(_c)


def test_corpus_deterministic_and_covers_forms() -> None:
    assert sa.build_corpus() == CORPUS
    ta = {f for (lg, f) in GROUPS if lg == "ta"}
    en = {f for (lg, f) in GROUPS if lg == "en"}
    assert {
        "western_spaced", "western_concat", "punctuation", "native_words", "colloquial",
        "tanglish", "english_words", "english_in_tamil_script", "code_mixed", "filler_ta",
        "extra_words", "partial_two", "split_segments", "compound", "wrong_language_en_words",
    } <= ta  # fmt: skip
    assert {"wrong_language_ta_words", "wrong_language_tanglish", "grouping", "compound"} <= en


@pytest.mark.parametrize(
    "key", sorted(k for k in GROUPS if k[1] not in KNOWN_GAPS), ids=lambda k: f"{k[0]}-{k[1]}"
)
def test_EV_SC_form_extracts_truth(key: tuple[str, str]) -> None:
    for c in GROUPS[key]:
        r = sa.score_case(SCORER, c)
        assert r["recognised"] == c.truth, (c.texts, r)
        assert r["status"] == ("OK" if len(c.truth) == 3 else "PARTIAL")


@pytest.mark.parametrize("form", sorted(KNOWN_GAPS))
def test_known_gaps_fail_safe(form: str) -> None:
    """Documented gaps: never scored as a confident answer, so the app re-asks."""
    for lang in (Lang.TA, Lang.EN):
        for c in GROUPS[(lang.value, form)]:
            r = sa.score_case(SCORER, c)
            assert r["recognised"] != c.truth
            assert r["status"] in {"AMBIGUOUS", "NO_DIGITS"}


def test_no_silent_misscore_anywhere() -> None:
    """No case ends OK with digits other than the ones said."""
    for c in CORPUS:
        r = sa.score_case(SCORER, c)
        assert not (r["status"] == "OK" and r["recognised"] != c.truth), (c, r)


def test_lcs_and_precision_recall() -> None:
    assert sa.lcs_pairs((2, 7, 4), (2, 2, 7, 4)) == [(0, 0), (1, 2), (2, 3)]
    assert sa.lcs_pairs((1, 2), ()) == []
    cases = [
        sa.Case(Lang.EN, "x", (2, 7, 4), ("I want to say 7 4",), (7, 4)),  # 2 extra: P=2/3
        sa.Case(Lang.EN, "x", (2, 7, 4), ("2 7 4",), (2, 7, 4)),
        sa.Case(Lang.EN, "x", (2, 7, 4), ("do saat char",), (2, 7, 4)),  # miss: R loses 3
    ]
    o = sa.evaluate(cases, SCORER)["langs"]["en"]["overall"]
    assert o["n"] == 3
    assert o["precision"] == round(5 / 6, 4)
    assert o["recall"] == round(5 / 8, 4)
    assert o["confusion"]["extra"] == {"2": 1}
    assert o["confusion"]["2"] == {"2": 1, "miss": 1}


def test_cli_writes_json_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "acc.json"
    assert sa.main(["--n", "3", "--out", str(out), "--report"]) == 0
    doc = json.loads(out.read_text())
    assert doc["experiment"] == "EXP-4"
    assert set(doc["langs"]) == {"ta", "en"}
    assert "| ta | **all** |" in capsys.readouterr().out
