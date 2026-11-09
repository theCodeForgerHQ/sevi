"""Approved spoken copy must survive the Evon output guard (issue #77).

Evon mostly restates the approved wording, so the wording itself, passed as ``allowed``, must
pass the content guard (zero-number, unit, medical, reassurance, script) under COPY_SPOKEN."""

from __future__ import annotations

import pytest

from sevi.contracts.enums import Lang
from sevi.core.guard import check
from sevi.core.wording import load_copy

COPY = load_copy().copy
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
def test_spoken_wording_passes_guard(text: str, lang: Lang) -> None:
    # Approved spoken copy is not Evon output; it keeps every content rule but not the live
    # 300-character cap, so reviewed Tamil is never trimmed to fit it.
    verdict = check(text, lang, "COPY_SPOKEN", allowed=[text])
    assert verdict.ok, verdict.reasons


@pytest.mark.parametrize(("text", "lang"), TEMPLATE)
def test_evon_template_explain_passes_guard(text: str, lang: Lang) -> None:
    verdict = check(text, lang, "EXPLAIN_SPOKEN", allowed=[text])
    assert verdict.ok, verdict.reasons
