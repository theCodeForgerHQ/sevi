"""Golden tests for core/dsp.py (A spec §1.6, G-DSP-01) and the mu-law table fixture for B.

The G-DSP-01 input is test-bank triplet 1-2-3 at -8 dB SNR (``stimuli/test-bank``), rendered by
``BankStimulusProvider.prechain``. Regenerate with ``SEVI_REGEN_GOLDEN=1 uv run pytest
tests/golden/a`` in a reviewed PR only.
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

from sevi.contracts.enums import PresentationKind
from sevi.contracts.rng import derive_seed
from sevi.contracts.trial import PresentationSpec
from sevi.core import dsp
from sevi.core.stimulus import BankStimulusProvider

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "golden"
GOLDEN_JSON = GOLDEN_DIR / "golden.json"
DSP01_OUT = GOLDEN_DIR / "dsp01_out.ulaw"
ULAW_TABLE_JSON = GOLDEN_DIR / "ulaw_table.json"
REGEN = os.environ.get("SEVI_REGEN_GOLDEN") == "1"
SNR_DB = -8.0


def _platform_triple() -> dict[str, str]:
    return {"machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__}


DSP01_TRIPLET = (1, 2, 3)


def dsp01_input(bank: BankStimulusProvider) -> np.ndarray:
    """Test-bank triplet 1-2-3 at -8 dB SNR, NORMAL: the float that enters the phone chain."""
    spec = PresentationSpec(
        presentation_id=1,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=SNR_DB,
        triplet=DSP01_TRIPLET,
        pool_index=0,
        noise_seed=derive_seed("golden", "dsp01"),
    )
    return bank.prechain(spec).y


def _snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    err = float(np.sum((ref - got) ** 2))
    return math.inf if err == 0 else 10 * math.log10(float(np.sum(ref**2)) / err)


def update_golden_json(entries: dict[str, object]) -> None:
    """Read-modify-write of golden.json (shared by the dsp and stimulus goldens)."""
    golden = json.loads(GOLDEN_JSON.read_text()) if GOLDEN_JSON.exists() else {}
    golden.update(entries)
    golden["platform"] = _platform_triple()
    GOLDEN_JSON.write_text(json.dumps(golden, sort_keys=True, indent=2) + "\n")


@pytest.mark.skipif(not REGEN, reason="set SEVI_REGEN_GOLDEN=1 to regenerate")
def test_regenerate_dsp_golden(test_bank: BankStimulusProvider) -> None:  # pragma: no cover
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    data, _ = dsp.phone_chain(dsp01_input(test_bank))
    DSP01_OUT.write_bytes(data)
    update_golden_json(
        {
            "dsp01": hashlib.sha256(data).hexdigest(),
            "params": {
                "input": "test-bank triplet (1,2,3), seed derive_seed('golden','dsp01'), NORMAL",
                "snr_db": SNR_DB,
            },
        }
    )
    ULAW_TABLE_JSON.write_text(json.dumps(dsp.ULAW_DECODE_TABLE.tolist()) + "\n")


def test_G_DSP_01_phone_chain_golden(test_bank: BankStimulusProvider) -> None:
    golden = json.loads(GOLDEN_JSON.read_text())
    data, z = dsp.phone_chain(dsp01_input(test_bank))
    ref = DSP01_OUT.read_bytes()
    assert len(data) == len(ref)
    if golden["platform"] == _platform_triple():
        assert hashlib.sha256(data).hexdigest() == golden["dsp01"]
    ref_f = dsp.pcm16_to_float(dsp.ulaw_decode(ref))
    got_f = dsp.pcm16_to_float(dsp.ulaw_decode(data))
    assert _snr_db(ref_f, got_f) > 60.0
    # 8 kHz float vs the decoded reference: within mu-law quantisation (~38 dB SNR)
    assert _snr_db(z, ref_f) > 30.0


def test_ulaw_table_fixture_matches_decode_table() -> None:
    """Fixture consumed by B's web/static/js/g711.js test (A §1.6 DoD)."""
    table = json.loads(ULAW_TABLE_JSON.read_text())
    assert table == dsp.ULAW_DECODE_TABLE.tolist()
    assert len(table) == 256


@pytest.mark.parametrize("cond_peak", [-3.0])
def test_dsp01_input_peak_and_level(test_bank: BankStimulusProvider, cond_peak: float) -> None:
    x = dsp01_input(test_bank)
    assert float(np.max(np.abs(x))) <= 10 ** (cond_peak / 20) + 1e-9
    assert dsp.dbfs(x) <= -20.0 + 1e-9
