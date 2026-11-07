"""Golden tests for core/stimulus.py (A spec §5.3, G-STIM-01, U-STIM-04).

Regenerate with ``SEVI_REGEN_GOLDEN=1 uv run pytest tests/golden/a`` in a reviewed PR only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pytest
import scipy

from sevi.contracts.enums import Condition, PresentationKind
from sevi.contracts.rng import derive_seed
from sevi.contracts.trial import PresentationSpec
from sevi.core import dsp
from sevi.core.stimulus import BankStimulusProvider

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "golden"
GOLDEN_JSON = GOLDEN_DIR / "golden.json"
STIM01_OUT = GOLDEN_DIR / "stim01_out.ulaw"
REGEN = os.environ.get("SEVI_REGEN_GOLDEN") == "1"
U_STIM_04_SESSION = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b"


def _platform_triple() -> dict[str, str]:
    return {"machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__}


def stim01_spec() -> PresentationSpec:
    return PresentationSpec(
        presentation_id=1,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=-8.0,
        triplet=(4, 7, 2),
        pool_index=0,
        noise_seed=derive_seed("golden", "stim01"),
        condition=Condition.NORMAL,
    )


def u_stim_04_spec() -> PresentationSpec:
    return PresentationSpec(
        presentation_id=1,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=0.0,
        triplet=(1, 2, 3),
        pool_index=0,
        noise_seed=derive_seed(U_STIM_04_SESSION, "noise:1"),
    )


def _snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    err = float(np.sum((ref - got) ** 2))
    return math.inf if err == 0 else 10 * math.log10(float(np.sum(ref**2)) / err)


@pytest.mark.skipif(not REGEN, reason="set SEVI_REGEN_GOLDEN=1 to regenerate")
def test_regenerate_stim_golden(test_bank: BankStimulusProvider) -> None:  # pragma: no cover
    r = test_bank.render_trial(stim01_spec())
    STIM01_OUT.write_bytes(r.ulaw)
    golden = json.loads(GOLDEN_JSON.read_text())
    golden["stim01"] = hashlib.sha256(r.ulaw).hexdigest()
    golden["stim01_params"] = {
        "triplet": [4, 7, 2],
        "snr_db": -8.0,
        "noise_seed": "derive_seed('golden','stim01')",
        "condition": "NORMAL",
        "bank_hash": test_bank.manifest.bank_hash,
    }
    golden["stim_start_1"] = test_bank.render_trial(u_stim_04_spec()).noise_start_sample
    golden["platform"] = _platform_triple()
    GOLDEN_JSON.write_text(json.dumps(golden, sort_keys=True, indent=2) + "\n")


def test_G_STIM_01_same_spec_identical_bytes_and_golden(test_bank: BankStimulusProvider) -> None:
    golden = json.loads(GOLDEN_JSON.read_text())
    a = test_bank.render_trial(stim01_spec())
    b = test_bank.render_trial(stim01_spec())
    assert a.ulaw == b.ulaw
    assert a == b
    assert golden["stim01_params"]["bank_hash"] == test_bank.manifest.bank_hash
    ref = STIM01_OUT.read_bytes()
    assert len(a.ulaw) == len(ref)
    if golden["platform"] == _platform_triple():
        assert hashlib.sha256(a.ulaw).hexdigest() == golden["stim01"]
    ref_f = dsp.pcm16_to_float(dsp.ulaw_decode(ref))
    got_f = dsp.pcm16_to_float(dsp.ulaw_decode(a.ulaw))
    assert _snr_db(ref_f, got_f) > 60.0


def test_U_STIM_04_noise_start_sample_pinned(test_bank: BankStimulusProvider) -> None:
    golden = json.loads(GOLDEN_JSON.read_text())
    r = test_bank.render_trial(u_stim_04_spec())
    assert r.noise_start_sample == golden["stim_start_1"]
    n = 41600  # three 6400-sample test-bank digits + lead, gaps, lag
    expected = np.random.Generator(np.random.PCG64(u_stim_04_spec().noise_seed)).integers(
        0, 960000 - n + 1
    )
    assert r.noise_start_sample == int(expected)
