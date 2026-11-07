"""Consent intent (B spec §11.3; contracts §4.13 guard and consent words)."""

from __future__ import annotations

import pytest

from sevi.contracts.enums import Lang
from sevi.core.intents import consent_intent


@pytest.mark.parametrize(
    ("lang", "texts", "want"),
    [
        (Lang.TA, ["ஆமா"], "YES"),
        (Lang.TA, ["சரி சொல்லுங்க"], "YES"),
        (Lang.TA, ["aama"], "YES"),
        (Lang.TA, ["வேண்டாம்"], "NO"),
        (Lang.TA, ["illa"], "NO"),
        (Lang.TA, ["okay"], "YES"),  # English words also count for ta
        (Lang.EN, ["Yes please"], "YES"),
        (Lang.EN, ["I don't want to"], "NO"),
        (Lang.EN, ["nope"], "NO"),
        (Lang.EN, ["yes", "no"], None),  # both found
        (Lang.EN, ["hello"], None),
        (Lang.EN, ["sari"], None),  # Tamil words do not count for en
        (Lang.EN, ["noble"], None),  # whole-token match
    ],
)
def test_U_INT_consent_intent(lang: Lang, texts: list[str], want: str | None) -> None:
    assert consent_intent(lang, texts) == want
