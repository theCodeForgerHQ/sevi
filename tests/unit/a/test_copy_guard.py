"""Approved spoken copy must survive the Evon output guard (issue #77).

Evon mostly restates the approved wording, so the wording itself must pass the zero-number,
unit and length checks for EXPLAIN_SPOKEN. MEDICAL/REASSURANCE are not asserted here: the
``allowed`` handling for those is owned by the guard (B)."""

from __future__ import annotations

import pytest

from sevi.contracts.enums import Lang
from sevi.core.guard import GuardReason, check
from sevi.core.wording import load_copy

COPY = load_copy().copy
CHECKED = {GuardReason.NUMERAL, GuardReason.NUMBER_WORD, GuardReason.UNIT, GuardReason.LENGTH}
SPOKEN = [
    pytest.param(t, Lang(lang), id=f"{e.key}:{v.variant_id}:{lang}")
    for e in COPY.entries
    if e.key.startswith("wording.") and e.key.endswith(".spoken")
    for v in e.variants
    for lang, t in v.text.items()
]
TEMPLATE = [
    pytest.param(t, Lang(lang), id=f"{e.key}:{lang}")
    for e in COPY.entries
    if e.key == "evon.template.explain"
    for v in e.variants
    for lang, t in v.text.items()
]


def test_spoken_wordings_cover_both_languages() -> None:
    assert {p.values[1] for p in SPOKEN} == {Lang.EN, Lang.TA}
    assert len(TEMPLATE) == 2


@pytest.mark.parametrize(("text", "lang"), SPOKEN)
def test_spoken_wording_passes_number_unit_length_guard(text: str, lang: Lang) -> None:
    verdict = check(text, lang, "EXPLAIN_SPOKEN", allowed=[text])
    assert not CHECKED & set(verdict.reasons), verdict.reasons


@pytest.mark.parametrize(("text", "lang"), TEMPLATE)
def test_evon_template_explain_passes_guard(text: str, lang: Lang) -> None:
    verdict = check(text, lang, "EXPLAIN_SPOKEN", allowed=[text])
    assert verdict.ok, verdict.reasons
