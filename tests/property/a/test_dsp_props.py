"""Property tests for core/dsp.py (A spec §1.6, P-DSP-01..03, P-DSP-06)."""

from __future__ import annotations

import itertools
import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from sevi.contracts.enums import Condition
from sevi.core import dsp

snr = st.floats(-20, 10, allow_nan=False)
sig = hnp.arrays(
    np.float64,
    st.integers(1600, 32000),
    elements=st.floats(-0.9, 0.9, allow_nan=False),
).filter(lambda a: dsp.rms(a) > 1e-4)
pcm = hnp.arrays(np.int16, st.integers(1, 4000))

SETTINGS = settings(deadline=None, max_examples=60)


@SETTINGS
@given(data=st.data(), snr_db=snr)
def test_P_DSP_01_mix_snr_definition(data: st.DataObject, snr_db: float) -> None:
    speech = data.draw(sig)
    noise = data.draw(
        hnp.arrays(np.float64, speech.size, elements=st.floats(-0.9, 0.9, allow_nan=False)).filter(
            lambda a: dsp.rms(a[320:-320]) > 1e-3
        )
    )
    ref = dsp.rms(speech)
    _, _, n_out = dsp.mix_at_snr(speech, noise, snr_db, ref)
    g_n = (ref / dsp.rms(noise)) * 10 ** (-snr_db / 20)
    assert abs(20 * math.log10(ref / dsp.rms(g_n * noise)) - snr_db) < 1e-9
    # post-mix measurement on the noise component, ramp excluded (samples 320..N-320)
    inner = slice(320, speech.size - 320)
    measured = 20 * math.log10(
        ref / dsp.rms(n_out[inner] / (dsp.rms(noise[inner]) / dsp.rms(noise)))
    )
    assert abs(measured - snr_db) < 0.1


def _segment(x: int) -> int:
    m = min(abs(x), 32635) + 0x84
    return (m >> 7).bit_length() - 1


@SETTINGS
@given(x=pcm)
def test_P_DSP_02_ulaw_error_bound_and_idempotence(x: np.ndarray) -> None:
    enc = dsp.ulaw_encode(x)
    # Idempotent up to the negative-zero byte: 0x7F (-0) decodes to 0, which encodes to 0xFF.
    assert dsp.ulaw_encode(dsp.ulaw_decode(enc)) == enc.replace(b"\x7f", b"\xff")
    assert np.array_equal(
        dsp.ulaw_decode(dsp.ulaw_encode(dsp.ulaw_decode(enc))), dsp.ulaw_decode(enc)
    )
    xs = np.clip(x, -32635, 32635).astype(np.int16)
    dec = dsp.ulaw_decode(dsp.ulaw_encode(xs)).astype(int)
    for v, d in zip(xs.tolist(), dec.tolist(), strict=True):
        assert abs(d - v) <= 2 ** (_segment(v) + 3)
        assert abs(d - v) <= 512


@SETTINGS
@given(x=sig, t=st.floats(-40, -6, allow_nan=False))
def test_P_DSP_03_scale_to_rms(x: np.ndarray, t: float) -> None:
    assert abs(dsp.dbfs(dsp.scale_to_rms_dbfs(x, t)) - t) < 0.05


@SETTINGS
@given(x=sig, lim=st.floats(-12, -1, allow_nan=False))
def test_peak_limit_never_exceeds(x: np.ndarray, lim: float) -> None:
    y, _ = dsp.peak_limit(dsp.scale_to_rms_dbfs(x, -20.0), lim)
    assert float(np.max(np.abs(y))) <= 10 ** (lim / 20) + 1e-12


@SETTINGS
@given(n=st.integers(2, 200000))
def test_resample_length(n: int) -> None:
    assert dsp.resample_16k_to_8k(np.zeros(n)).size == math.ceil(n / 2)


_FLOORS: dict[Condition, np.ndarray] = {}


def _floor(c: Condition) -> np.ndarray:
    if c not in _FLOORS:
        _FLOORS[c] = dsp.floor_buffer("ta", c)
    return _FLOORS[c]


@settings(deadline=None, max_examples=10)
@given(start=st.integers(0, 960000 - 49600))
def test_P_DSP_06_floor_segments(start: int) -> None:
    seg = {c: _floor(c)[start : start + 49600] for c in (Condition.N2, Condition.N3)}
    fcs = (500.0, 1000.0, 2000.0, 3150.0)
    hl_diff = dsp.hl_db(Condition.N3, np.array(fcs)) - dsp.hl_db(Condition.N2, np.array(fcs))
    for fc, d in zip(fcs, hl_diff.tolist(), strict=True):
        assert abs(d - 15.0) < 1e-9
        lv2 = dsp.third_octave_level_dbfs(seg[Condition.N2], fc)
        lv3 = dsp.third_octave_level_dbfs(seg[Condition.N3], fc)
        assert abs((lv3 - lv2) - d) < 0.5
    bands = (1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0)
    for c in (Condition.N2, Condition.N3):
        lv = [dsp.third_octave_level_dbfs(seg[c], fc) for fc in bands]
        assert all(b >= a - 0.5 for a, b in itertools.pairwise(lv))
