"""Property tests for core/stimulus.py (A spec §5.3 P-STIM-01/02, §1.6 P-DSP-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sevi.contracts.enums import Condition, Lang, PresentationKind
from sevi.contracts.languages import DIGIT_SETS
from sevi.contracts.trial import PresentationSpec
from sevi.core import dsp
from sevi.core.bank import load_bank
from sevi.core.stimulus import BankStimulusProvider
from sevi.core.triplets import build_pool

POOL = build_pool(DIGIT_SETS[Lang.TA])
PEAK_LIM = 10.0 ** (-3.0 / 20.0)
SETTINGS = settings(deadline=None, max_examples=40)


@st.composite
def specs(draw: st.DrawFn) -> PresentationSpec:
    i = draw(st.integers(0, len(POOL) - 1))
    return PresentationSpec(
        presentation_id=draw(st.integers(1, 10_000)),
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=draw(st.floats(-20, 10, allow_nan=False)),
        triplet=POOL[i],
        pool_index=i,
        noise_seed=draw(st.integers(0, 2**64 - 1)),
        condition=draw(st.sampled_from(list(Condition))),
    )


@SETTINGS
@given(s=specs())
def test_P_STIM_01_timing_invariants(test_bank: BankStimulusProvider, s: PresentationSpec) -> None:
    r = test_bank.render_trial(s)
    t = r.timing
    assert r.n_bytes == len(r.ulaw) == round(t.clip_ms * 8)
    onsets = [d.onset_ms for d in t.digits]
    assert onsets == sorted(onsets)
    assert len(set(onsets)) == 3
    lengths = {a.digit: a.n_samples for a in test_bank.manifest.digits}
    for d in t.digits:
        assert d.offset_ms - d.onset_ms == lengths[d.digit] / 16
    assert t.last_digit_offset_ms == t.digits[2].offset_ms == t.clip_ms - 500.0


@SETTINGS
@given(s=specs())
def test_P_DSP_04_peak_and_no_ulaw_clip(
    test_bank: BankStimulusProvider, s: PresentationSpec
) -> None:
    """P-DSP-04: pre-band-pass peak <= -3 dBFS; decoded output never reaches the mu-law clip."""
    pre = test_bank.prechain(s)
    assert float(np.max(np.abs(pre.y))) <= PEAK_LIM + 1e-9
    r = test_bank.render_trial(s)
    assert int(np.max(np.abs(dsp.ulaw_decode(r.ulaw).astype(np.int32)))) < 32124
    assert b"\x00" not in r.ulaw
    assert b"\x80" not in r.ulaw


def test_P_STIM_02_two_providers_identical(test_bank_dir: Path, test_bank_hash: str) -> None:
    """P-STIM-02: two providers loaded from the same files render identical bytes (50 specs)."""
    a = load_bank(Lang.TA, str(test_bank_dir), {Lang.TA: test_bank_hash}, False)
    b = load_bank(Lang.TA, str(test_bank_dir), {Lang.TA: test_bank_hash}, False)
    assert a is not b
    for k in range(50):
        s = _fixed(k)
        assert a.render_trial(s).ulaw == b.render_trial(s).ulaw


def _fixed(k: int) -> PresentationSpec:
    i = (k * 37) % len(POOL)
    return PresentationSpec(
        presentation_id=k + 1,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=-20.0 + (k % 30) + 0.25 * (k % 3),
        triplet=POOL[i],
        pool_index=i,
        noise_seed=k * 7919,
        condition=list(Condition)[k % len(Condition)],
    )
