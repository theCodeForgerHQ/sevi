"""Consent yes/no intent (B spec §11.3; contracts §4.13 "Guard and consent words"). Pure.

Used only by the phone record-consent variant: browser consent is a click, phone default is DTMF.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Final, Literal

from sevi.contracts.enums import Lang
from sevi.core.guard import tokenize

CONSENT_WORDS: Final[dict[Lang, dict[str, frozenset[str]]]] = {
    Lang.TA: {
        "YES": frozenset({"ஆமா", "ஆம்", "சரி", "aama", "sari"}),
        "NO": frozenset({"இல்லை", "வேண்டாம்", "illa", "vendaam"}),
    },
    Lang.EN: {
        "YES": frozenset({"yes", "yeah", "ok", "okay", "sure"}),
        "NO": frozenset({"no", "nope", "don't"}),
    },
}


def consent_intent(lang: Lang, texts: Sequence[str]) -> Literal["YES", "NO"] | None:
    """Token match against the words for ``lang`` plus English; both or neither found -> None."""
    yes = CONSENT_WORDS[lang]["YES"] | CONSENT_WORDS[Lang.EN]["YES"]
    no = CONSENT_WORDS[lang]["NO"] | CONSENT_WORDS[Lang.EN]["NO"]
    tokens: set[str] = set()
    for text in texts:
        tokens.update(tokenize(unicodedata.normalize("NFC", text).casefold()))
    has_yes = bool(tokens & yes)
    has_no = bool(tokens & no)
    if has_yes == has_no:
        return None
    return "YES" if has_yes else "NO"
