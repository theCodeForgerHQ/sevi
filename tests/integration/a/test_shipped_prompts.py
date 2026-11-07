"""The shipped Timbre prompt sets load in prod mode against the shipped banks (#63).

Skips a language whose prompt set has not been rendered yet. No network, no keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sevi.app.prompts import load_prompts
from sevi.contracts.enums import Lang
from sevi.core.bank import read_manifest

REPO = Path(__file__).resolve().parents[3]
PROMPTS = REPO / "stimuli" / "prompts"


@pytest.mark.parametrize("lang", [Lang.TA, Lang.EN])
def test_shipped_prompts_load_in_prod(lang: Lang) -> None:
    if not (PROMPTS / lang.value / "manifest.json").exists():
        pytest.skip(f"no shipped {lang.value} prompts")
    bank = read_manifest(REPO / "stimuli" / lang.value / "manifest.json")
    load_prompts(lang, PROMPTS, bank, prod=True)
