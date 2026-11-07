"""Language data that is frozen with the bank (contracts §4.13, U2).

Digit sets and the exact stimulus words sent to Timbre. The normalisation map (accepted spoken
forms, intents, fillers) is scorer data and lives in ``core/lexicon/{ta,en}.yaml``, not here.
"""

from types import MappingProxyType
from typing import Final

from .enums import Lang

# Rule (R2 P1): exclude 0 always, then multisyllabic/confusable digits, keeping >= 8 digits.
# ta (banded): drop 9 (ஒன்பது). en (experimental, never norm-banded): drop 0 and 7.
DIGIT_SETS: Final[MappingProxyType[Lang, tuple[int, ...]]] = MappingProxyType(
    {
        Lang.TA: (1, 2, 3, 4, 5, 6, 7, 8),
        Lang.EN: (1, 2, 3, 4, 5, 6, 8, 9),
    }
)

# Exact text sent to Timbre per digit (never a numeral: R1 §2.6 number reading).
STIMULUS_WORDS: Final[MappingProxyType[Lang, MappingProxyType[int, str]]] = MappingProxyType(
    {
        Lang.TA: MappingProxyType(
            {
                1: "ஒன்று",
                2: "இரண்டு",
                3: "மூன்று",
                4: "நான்கு",
                5: "ஐந்து",
                6: "ஆறு",
                7: "ஏழு",
                8: "எட்டு",
            }
        ),
        Lang.EN: MappingProxyType(
            {
                1: "one",
                2: "two",
                3: "three",
                4: "four",
                5: "five",
                6: "six",
                8: "eight",
                9: "nine",
            }
        ),
    }
)

# VT-TMB-03 fallback words if a render is unusable (ta only).
ALT_STIMULUS_WORDS: Final[MappingProxyType[Lang, MappingProxyType[int, str]]] = MappingProxyType(
    {
        Lang.TA: MappingProxyType({2: "ரெண்டு", 4: "நாலு", 5: "அஞ்சு"}),
        Lang.EN: MappingProxyType({}),
    }
)

# Timbre render language per bank language (BankManifest.timbre.language).
TIMBRE_LANGUAGE: Final[MappingProxyType[Lang, str]] = MappingProxyType(
    {Lang.TA: "ta-IN", Lang.EN: "en-IN"}
)

# Languages that are never norm-banded (Scope v3 (b)); results are NO_BAND unless T1-T6/T10b.
EXPERIMENTAL_LANGS: Final = frozenset({Lang.EN})
