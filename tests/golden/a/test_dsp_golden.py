"""Golden tests for core/dsp.py (A spec §1.6, G-DSP-01) and the mu-law table fixture for B.

The real test bank does not exist yet, so the G-DSP-01 input is a synthetic triplet generated
deterministically from a derived seed (three harmonic "digits" in LTASS-shaped noise at -8 dB SNR).
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

from sevi.contracts.rng import derive_seed
from sevi.core import dsp

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "golden"
GOLDEN_JSON = GOLDEN_DIR / "golden.json"
DSP01_OUT = GOLDEN_DIR / "dsp01_out.ulaw"
ULAW_TABLE_JSON = GOLDEN_DIR / "ulaw_table.json"
REGEN = os.environ.get("SEVI_REGEN_GOLDEN") == "1"
SNR_DB = -8.0


def _platform_triple() -> dict[str, str]:
    return {"machine": platform.machine(), "numpy": np.__version__, "scipy": scipy.__version__}


def dsp01_input() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(derive_seed("golden", "dsp01")))
    digits = []
    for f0, n in ((140.0, 6400), (180.0, 5600), (220.0, 7200)):
        t = np.arange(n) / dsp.FS_MIX
        phases = rng.uniform(0, 2 * np.pi, 19)
        d = sum(np.sin(2 * np.pi * f0 * k * t + phases[k - 1]) / k for k in range(1, 20))
        d = np.asarray(d) * dsp.raised_cosine_envelope(n, 320)
        digits.append(dsp.scale_to_rms_dbfs(d, -20.0))
    speech, _ = dsp.assemble_triplet(digits, [0.0, 0.0, 0.0])
    f, p = dsp.ltass_psd(digits)
    noise = dsp.synth_noise(f, p, 960000, derive_seed("golden", "dsp01-noise"))
    start = int(rng.integers(0, 960000 - speech.size))
    mix, _, _ = dsp.mix_at_snr(speech, noise[start : start + speech.size], SNR_DB, 10 ** (-1.0))
    mix = dsp.scale_to_rms_dbfs(mix, -20.0)
    out, _ = dsp.peak_limit(mix, -3.0)
    return out


def _snr_db(ref: np.ndarray, got: np.ndarray) -> float:
    err = float(np.sum((ref - got) ** 2))
    return math.inf if err == 0 else 10 * math.log10(float(np.sum(ref**2)) / err)


def _regenerate() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    data, _ = dsp.phone_chain(dsp01_input())
    DSP01_OUT.write_bytes(data)
    golden = {
        "dsp01": hashlib.sha256(data).hexdigest(),
        "params": {"input": "dsp01_input() seed derive_seed('golden','dsp01')", "snr_db": SNR_DB},
        "platform": _platform_triple(),
    }
    GOLDEN_JSON.write_text(json.dumps(golden, sort_keys=True, indent=2) + "\n")
    ULAW_TABLE_JSON.write_text(json.dumps(dsp.ULAW_DECODE_TABLE.tolist()) + "\n")


if REGEN:  # pragma: no cover - maintenance path
    _regenerate()


def test_G_DSP_01_phone_chain_golden() -> None:
    golden = json.loads(GOLDEN_JSON.read_text())
    data, z = dsp.phone_chain(dsp01_input())
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
def test_dsp01_input_peak_and_level(cond_peak: float) -> None:
    x = dsp01_input()
    assert float(np.max(np.abs(x))) <= 10 ** (cond_peak / 20) + 1e-9
    assert dsp.dbfs(x) <= -20.0 + 1e-9
