# ruff: noqa: PT018, RUF001, E501
"""Contract tests for the result seam (contracts §4.7, §4.11, §5.1, §5.1.1; C-RES-01 in scope).

Reduced scope: the only results file A's result/copy code writes is ``copy_selection.json``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sevi.contracts.copy import CopyFile
from sevi.contracts.enums import Lang, Surface
from sevi.contracts.evals import CopySelectionFile
from sevi.contracts.ports import ClassifierBuilder, ResultClassifier, T8Decider, WordingText
from sevi.contracts.result import ResultDecision
from sevi.core import result as result_mod
from sevi.core.wording import DEFAULT_COPY_PATH
from sevi.fakes.scenarios.result_inputs import cell, inp, norms

REPO = Path(__file__).resolve().parents[2]

DECISION_EXAMPLE = (
    '{"rule":"T13","band":"BORDERLINE","refer_reason":null,"no_band_reason":null,'
    '"wording_class":"W_BORDER","surfaces":["SPOKEN","PAGE","REPORT","OPERATOR"],'
    '"norm_cell":{"lang":"ta","channel":"browser","mode":"keypad","transducer":"earphones"},'
    '"cutoff_version":"v0","nh95_db":-6.1,"n_norm":7,"srt_db":-5.43,"srt_display":"−5.5"}'
)


def test_C_RES_classifier_satisfies_port() -> None:
    builder: ClassifierBuilder = result_mod.build_classifier
    t8: T8Decider = result_mod.t8_decision
    clf = builder()
    assert isinstance(clf, ResultClassifier)
    d = clf.classify(inp())
    w = clf.wording(d, Lang.TA, Surface.PAGE)
    assert isinstance(w, WordingText) and w.copy_key == "wording.W_PASS.page"
    assert t8(None).rule.value == "T8"


def test_C_RES_decision_example_validates() -> None:
    d = ResultDecision.model_validate_json(DECISION_EXAMPLE)
    assert d.srt_display == "−5.5"


def test_C_RES_copy_yaml_validates_as_copyfile() -> None:
    import yaml

    raw = yaml.safe_load(Path(DEFAULT_COPY_PATH).read_text(encoding="utf-8"))
    cf = CopyFile.model_validate(raw)
    assert cf.schema_id == "sevi.copy/1"
    assert Path(DEFAULT_COPY_PATH).is_relative_to(REPO / "src" / "sevi" / "copy")


def test_C_RES_01_copy_selection_file_validates(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "select_copy", REPO / "scripts" / "select_copy.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "copy_selection.json"
    assert mod.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    sel = CopySelectionFile.model_validate(json.loads(out.read_text(encoding="utf-8")))
    assert sel.meta.schema_id == "sevi.results.copy_selection/1"


def test_C_RES_norms_reject_en_cells() -> None:
    with pytest.raises(ValidationError):
        norms(cell(lang=Lang.EN))
