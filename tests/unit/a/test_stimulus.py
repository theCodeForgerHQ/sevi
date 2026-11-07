"""Unit tests for core/stimulus.py (A spec §5.2-§5.3, U-STIM-01..03, 05, 07, 08).

U-STIM-04 and G-STIM-01 live in tests/golden/a/test_stim_golden.py. U-STIM-06
(``render_panel_trial``) is cut with the human panel (BUILD-BRIEF §4).
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from sevi.calibration.testbank import build_bank, tone_digit
from sevi.contracts.enums import Condition, Lang, PresentationKind
from sevi.contracts.rng import derive_seed
from sevi.contracts.stimulus import RenderedTrial
from sevi.contracts.trial import PresentationSpec
from sevi.core import dsp
from sevi.core.bank import load_bank
from sevi.core.dsp import F64
from sevi.core.stimulus import BankStimulusProvider

PEAK_LIM = 10.0 ** (-3.0 / 20.0)


def spec(
    triplet: tuple[int, int, int] = (4, 7, 2),
    snr_db: float = -8.0,
    seed: int = 12345,
    condition: Condition = Condition.NORMAL,
    pid: int = 1,
) -> PresentationSpec:
    return PresentationSpec(
        presentation_id=pid,
        kind=PresentationKind.SCORED,
        scored_index=1,
        snr_db=snr_db,
        triplet=triplet,
        pool_index=0,
        noise_seed=seed,
        condition=condition,
    )


@pytest.fixture(scope="session")
def timing_bank(tmp_path_factory: pytest.TempPathFactory) -> BankStimulusProvider:
    """Fixture bank with the worked-example digit lengths 4: 8800, 7: 7040, 2: 8160."""
    d = tmp_path_factory.mktemp("timing")
    build_bank(d, lengths={4: 8800, 7: 7040, 2: 8160})
    return load_bank(Lang.TA, str(d), {}, allow_unfrozen=True)


def _impulse_digit(d: int, n: int) -> F64:
    """Click train, period 160 samples: crest factor ~22 dB."""
    x = np.zeros(n, dtype=np.float64)
    x[80::160] = 1.0
    return x if d == 1 else tone_digit(d, n)


@pytest.fixture(scope="session")
def crest_bank(tmp_path_factory: pytest.TempPathFactory) -> BankStimulusProvider:
    d = tmp_path_factory.mktemp("crest")
    build_bank(d, digit_fn=_impulse_digit)
    return load_bank(Lang.TA, str(d), {}, allow_unfrozen=True)


def test_U_STIM_01_timing_worked_example(timing_bank: BankStimulusProvider) -> None:
    r = timing_bank.render_trial(spec((4, 7, 2)))
    t = r.timing
    assert t.clip_ms == 2900.0
    assert t.noise_on_ms == 0.0
    assert t.noise_off_ms == 2900.0
    assert [d.position for d in t.digits] == [1, 2, 3]
    assert [d.digit for d in t.digits] == [4, 7, 2]
    assert [d.onset_ms for d in t.digits] == [500.0, 1250.0, 1890.0]
    assert [d.offset_ms for d in t.digits] == [1050.0, 1690.0, 2400.0]
    assert t.last_digit_offset_ms == 2400.0
    assert r.n_bytes == 23200 == len(r.ulaw)
    assert isinstance(r, RenderedTrial)
    assert r.bank_hash == timing_bank.manifest.bank_hash
    assert r.lang is Lang.TA


def test_U_STIM_02_ulaw_bytes_roundtrip(test_bank: BankStimulusProvider) -> None:
    r = test_bank.render_trial(spec())
    raw = b"\x00\xff\x7f\x80="
    r2 = r.model_copy(update={"ulaw": raw, "n_bytes": len(raw)})
    r3 = RenderedTrial.model_validate(r2.model_dump())
    assert r3.ulaw == raw
    assert RenderedTrial.model_validate_json(r2.model_dump_json()).ulaw == raw
    assert RenderedTrial.model_validate_json(r.model_dump_json()) == r


def test_U_STIM_03_digit_outside_digit_set(test_bank: BankStimulusProvider) -> None:
    with pytest.raises(ValueError, match="outside"):
        test_bank.render_trial(spec((9, 1, 2)))


def test_level_worked_example(test_bank: BankStimulusProvider) -> None:
    """§5.2 levels: ref_rms = 10**(-23/20) for an offset-0 bank; mix leaves at -20 dBFS."""
    assert test_bank.ref_rms == pytest.approx(10 ** (-23 / 20), abs=1e-4)
    pre = test_bank.prechain(spec(snr_db=-8.0))
    assert dsp.dbfs(pre.y) == pytest.approx(-20.0, abs=1e-9)
    assert not pre.lim1
    assert not pre.lim2


def test_U_STIM_05_level_limited(crest_bank: BankStimulusProvider) -> None:
    hi = spec((1, 2, 3), snr_db=10.0)
    pre = crest_bank.prechain(hi)
    assert pre.lim1
    assert float(np.max(np.abs(pre.y))) == pytest.approx(PEAK_LIM, abs=1e-12)
    assert 20 * np.log10(np.max(np.abs(pre.y))) == pytest.approx(-3.0, abs=5e-4)
    assert crest_bank.render_trial(hi).level_limited is True
    assert crest_bank.render_trial(spec((1, 2, 3), snr_db=-20.0)).level_limited is False


@pytest.mark.parametrize("snr_db", [-20.0, -8.0, 10.0])
def test_U_STIM_07_n3_is_normal_plus_floor(test_bank: BankStimulusProvider, snr_db: float) -> None:
    s_norm = spec(snr_db=snr_db)
    s_n3 = spec(snr_db=snr_db, condition=Condition.N3)
    a, b = test_bank.prechain(s_norm), test_bank.prechain(s_n3)
    ra, rb = test_bank.render_trial(s_norm), test_bank.render_trial(s_n3)
    assert a.start == b.start == ra.noise_start_sample == rb.noise_start_sample
    assert a.y.size == b.y.size
    assert ra.timing == rb.timing
    assert ra.n_bytes == rb.n_bytes
    floor = test_bank.floor(Condition.N3)[b.start : b.start + b.y.size]
    unlimited = a.y + floor
    p = float(np.max(np.abs(unlimited)))
    g = PEAK_LIM / p if p > PEAK_LIM else 1.0
    assert b.lim2 == (g < 1.0)
    np.testing.assert_allclose(b.y / g - a.y, floor, rtol=0, atol=1e-12)
    assert rb.level_limited == (b.lim1 or b.lim2)


def test_conditions_change_bytes_not_timing(test_bank: BankStimulusProvider) -> None:
    rs = {c: test_bank.render_trial(spec(condition=c)) for c in Condition}
    assert len({r.ulaw for r in rs.values()}) == len(Condition)
    assert len({(r.timing, r.noise_start_sample, r.n_bytes) for r in rs.values()}) == 1
    assert all(r.condition is c for c, r in rs.items())


def test_provider_arrays_are_read_only(test_bank: BankStimulusProvider) -> None:
    f = test_bank.floor(Condition.N2)
    with pytest.raises(ValueError, match="read-only"):
        f[0] = 1.0


@pytest.mark.slow
def test_U_STIM_08_render_p95(test_bank: BankStimulusProvider) -> None:
    """Perf on the test bank (400 ms digits); spec budget is for real-size banks."""
    budgets = {Condition.NORMAL: 20.0, Condition.N2: 20.0, Condition.N3: 20.0}
    budgets[Condition.LP1500] = 30.0
    for cond, budget_ms in budgets.items():
        times = []
        for i in range(200):
            s = spec(seed=derive_seed("perf", f"noise:{i}"), condition=cond, pid=i + 1)
            t0 = time.perf_counter()
            test_bank.render_trial(s)
            times.append((time.perf_counter() - t0) * 1000.0)
        p95 = statistics.quantiles(times, n=20)[18]
        print(f"U-STIM-08 {cond.value}: p95 {p95:.2f} ms")
        assert p95 <= budget_ms
