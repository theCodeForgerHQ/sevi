"""Unit tests for core/dsp.py (A spec §1.6, U-DSP-01..11; contracts §4.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from sevi.contracts.constants import MULAW_SILENCE_BYTE
from sevi.contracts.enums import Condition
from sevi.core import dsp

FS = 16000


def tone(freq: float, n: int, fs: int = FS, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs)


# --------------------------------------------------------------------------- U-DSP-01


def test_U_DSP_01_rms_dbfs_reference_levels() -> None:
    assert dsp.rms(np.ones(1000)) == 1.0
    assert dsp.dbfs(np.ones(1000)) == 0.0
    assert dsp.dbfs(tone(1000, 16000)) == pytest.approx(-3.0103, abs=5e-4)
    assert dsp.dbfs(np.zeros(100)) == -200.0


@pytest.mark.parametrize(
    "bad",
    [
        np.array([], dtype=np.float64),
        np.array([0.1, np.nan]),
        np.array([0.1, np.inf]),
        np.ones((2, 3)),
        np.ones(10, dtype=np.int16),
        np.ones(10, dtype=np.float32),
    ],
    ids=["empty", "nan", "inf", "2d", "int", "float32"],
)
def test_U_DSP_01_invalid_input_raises(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"x: "):
        dsp.rms(bad)
    with pytest.raises(ValueError, match=r"x: "):
        dsp.dbfs(bad)


# --------------------------------------------------------------------------- U-DSP-02

KNOWN_ENCODE = {
    0: 0xFF,
    1: 0xFF,
    -1: 0x7F,
    100: 0xF2,
    -100: 0x72,
    1000: 0xCE,
    8000: 0xA0,
    32767: 0x80,
    -32768: 0x00,
}
KNOWN_DECODE = {0xFF: 0, 0x7F: 0, 0xF2: 104, 0xCE: 988, 0xA0: 7932, 0x80: 32124, 0x00: -32124}


def test_U_DSP_02_ulaw_known_itu_vectors() -> None:
    pcm = np.array(list(KNOWN_ENCODE), dtype=np.int16)
    assert dsp.ulaw_encode(pcm) == bytes(KNOWN_ENCODE.values())
    dec = dsp.ulaw_decode(bytes(KNOWN_DECODE))
    assert dec.tolist() == list(KNOWN_DECODE.values())


def test_U_DSP_02_ulaw_tables_shape_and_silence() -> None:
    assert dsp.ULAW_ENCODE_TABLE.shape == (65536,)
    assert dsp.ULAW_ENCODE_TABLE.dtype == np.uint8
    assert dsp.ULAW_DECODE_TABLE.shape == (256,)
    assert dsp.ULAW_DECODE_TABLE.dtype == np.int16
    assert dsp.ulaw_encode(np.zeros(160, np.int16)) == bytes([MULAW_SILENCE_BYTE]) * 160
    assert not dsp.ULAW_ENCODE_TABLE.flags.writeable


def test_U_DSP_02_ulaw_round_trip_all_bytes() -> None:
    all_bytes = bytes(range(256))
    back = dsp.ulaw_encode(dsp.ulaw_decode(all_bytes))
    for b in range(256):
        expected = 0xFF if b == 0x7F else b  # both zeros
        assert back[b] == expected


def test_U_DSP_02_ulaw_round_trip_error_bounds() -> None:
    clipped = np.arange(-32635, 32636, dtype=np.int16)
    err = np.abs(dsp.ulaw_decode(dsp.ulaw_encode(clipped)).astype(int) - clipped.astype(int))
    assert err.max() == 512
    full = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    err_full = np.abs(dsp.ulaw_decode(dsp.ulaw_encode(full)).astype(int) - full.astype(int))
    assert err_full.max() == 644


def test_U_DSP_02_ulaw_decode_empty_raises() -> None:
    with pytest.raises(ValueError, match=r"empty"):
        dsp.ulaw_decode(b"")
    with pytest.raises(ValueError, match=r"empty"):
        dsp.ulaw_encode(np.zeros(0, np.int16))


def test_pcm_conversion_rules() -> None:
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    pcm = dsp.float_to_pcm16(x)
    assert pcm.tolist() == [-32768, -16384, 0, 16384, 32767, 32767]
    assert dsp.pcm16_to_float(np.array([-32768, 16384], np.int16)).tolist() == [-1.0, 0.5]


# --------------------------------------------------------------------------- U-DSP-03

BANDPASS_EXPECTED = {
    100: -40.62,
    200: -15.75,
    300: -3.01,
    1000: 0.00,
    2000: 0.00,
    3400: -3.01,
    4000: -9.79,
    5000: -24.26,
}


def test_U_DSP_03_bandpass_response_at_band_edges() -> None:
    sos = dsp.bandpass_sos()
    assert sos.shape == (4, 6)
    freqs = list(BANDPASS_EXPECTED)
    _, h = signal.sosfreqz(sos, worN=freqs, fs=FS)
    got = 20 * np.log10(np.abs(h))
    for f, g in zip(freqs, got, strict=True):
        assert g == pytest.approx(BANDPASS_EXPECTED[f], abs=0.02), f


def test_U_DSP_03_bandpass_is_causal_zero_state() -> None:
    x = np.zeros(64)
    x[10] = 1.0
    y = dsp.bandpass(x)
    assert np.all(y[:10] == 0.0)
    assert y.size == 64


# --------------------------------------------------------------------------- U-DSP-04


@pytest.mark.parametrize("freq", [1000.0, 3000.0])
def test_U_DSP_04_resample_keeps_rms(freq: float) -> None:
    x = tone(freq, 16000, amp=0.5)
    y = dsp.resample_16k_to_8k(x)
    lvl_in = dsp.dbfs(x[800:-800])
    lvl_out = dsp.dbfs(y[400:-400])
    assert lvl_out == pytest.approx(lvl_in, abs=0.02)


def test_U_DSP_04_resample_length() -> None:
    assert dsp.resample_16k_to_8k(np.zeros(49601)).size == 24801
    assert dsp.resample_48k_to_16k(np.zeros(48001)).size == math.ceil(48001 / 3)


# --------------------------------------------------------------------------- U-DSP-05


def test_U_DSP_05_scale_silent_raises() -> None:
    with pytest.raises(ValueError, match="silent input"):
        dsp.scale_to_rms_dbfs(np.zeros(100), -20.0)


def test_U_DSP_05_peak_limit() -> None:
    quiet = tone(1000, 1600, amp=0.5)  # peak -6 dBFS
    out, limited = dsp.peak_limit(quiet, -3.0)
    assert out is quiet
    assert limited is False
    loud = tone(1000, 1600, amp=0.99)
    out, limited = dsp.peak_limit(loud, -3.0)
    assert limited is True
    assert 20 * math.log10(np.max(np.abs(out))) == pytest.approx(-3.0, abs=1e-9)


def test_scale_to_rms_level_accuracy() -> None:
    x = tone(500, 8000, amp=0.1)
    assert dsp.dbfs(dsp.scale_to_rms_dbfs(x, -20.0)) == pytest.approx(-20.0, abs=1e-9)


def test_raised_cosine_envelope() -> None:
    env = dsp.raised_cosine_envelope(1000, 320)
    assert env[0] == 0.0
    assert env[-1] == 0.0
    assert np.all(env[320:680] == 1.0)
    assert np.allclose(env[:320], env[-320:][::-1])
    with pytest.raises(ValueError, match=r"2\*ramp"):
        dsp.raised_cosine_envelope(600, 320)


# --------------------------------------------------------------------------- U-DSP-06


def test_U_DSP_06_synth_noise_unit_rms_and_determinism() -> None:
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    psd = np.ones_like(freqs)
    a = dsp.synth_noise(freqs, psd, 960000, 1)
    assert dsp.rms(a) == pytest.approx(1.0, abs=1e-12)
    b = dsp.synth_noise(freqs, psd, 960000, 1)
    assert np.array_equal(a, b)
    c = dsp.synth_noise(freqs, psd, 960000, 2)
    assert not np.array_equal(a, c)


def test_U_DSP_06_synth_noise_odd_n_raises() -> None:
    freqs = np.fft.rfftfreq(1024, 1 / FS)
    with pytest.raises(ValueError, match=r"even"):
        dsp.synth_noise(freqs, np.ones_like(freqs), 1001, 1)


def test_ltass_psd_shape() -> None:
    f, p = dsp.ltass_psd([tone(1000, 8000), tone(2000, 8000)])
    assert f.shape == (513,)
    assert p.shape == (513,)
    assert f[np.argmax(p)] in (1000.0, 2000.0)


# --------------------------------------------------------------------------- U-DSP-07


def test_U_DSP_07_assemble_triplet_spans() -> None:
    digits = [np.full(n, 0.1) for n in (5120, 6400, 4800)]
    track, spans = dsp.assemble_triplet(digits, [0.0, 0.0, 0.0])
    assert spans == ((8000, 13120), (16320, 22720), (25920, 30720))
    assert track.size == 38720
    assert np.all(track[:8000] == 0.0)
    assert np.all(track[30720:] == 0.0)


def test_U_DSP_07_assemble_triplet_gains_and_odd_length() -> None:
    digits = [np.full(n, 0.1) for n in (100, 100, 100)]
    track, spans = dsp.assemble_triplet(digits, [0.0, 6.0, -6.0])
    assert track[spans[1][0]] == pytest.approx(0.1 * 10 ** (6 / 20))
    with pytest.raises(ValueError, match=r"even"):
        dsp.assemble_triplet([np.full(101, 0.1), digits[1], digits[2]], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"three digits"):
        dsp.assemble_triplet(digits[:2], [0.0, 0.0])


# --------------------------------------------------------------------------- U-DSP-08


def test_U_DSP_08_normalize_ulaw() -> None:
    x = dsp.scale_to_rms_dbfs(tone(1000, 8000, fs=8000), -35.0)
    data = dsp.ulaw_encode(dsp.float_to_pcm16(x))
    out = dsp.normalize_ulaw(data)
    assert len(out) == len(data)
    assert dsp.dbfs(dsp.pcm16_to_float(dsp.ulaw_decode(out))) == pytest.approx(-20.0, abs=0.3)
    with pytest.raises(ValueError, match="silent input"):
        dsp.normalize_ulaw(b"\xff" * 160)


# --------------------------------------------------------------------------- U-DSP-09

LP1500_EXPECTED = {1000: 0.00, 1400: 0.00, 1500: -6.02, 1600: -68.93, 2000: -73.12}


def test_U_DSP_09_lp1500_response() -> None:
    h = dsp.condition_taps(Condition.LP1500)
    assert h is not None
    assert h.size == 1025
    freqs = list(LP1500_EXPECTED)
    _, resp = signal.freqz(h, worN=freqs, fs=FS)
    got = 20 * np.log10(np.abs(resp))
    for f, g in zip(freqs, got, strict=True):
        assert g == pytest.approx(LP1500_EXPECTED[f], abs=0.1), f


def test_U_DSP_09_condition_application() -> None:
    for c in (Condition.NORMAL, Condition.N2, Condition.N3):
        assert dsp.condition_taps(c) is None
    rng = np.random.Generator(np.random.PCG64(9))
    x = rng.uniform(-0.1, 0.1, 4000)
    seg = rng.uniform(-0.01, 0.01, 4000)
    assert dsp.apply_condition(x, Condition.NORMAL, None) is x
    assert np.array_equal(dsp.apply_condition(x, Condition.N3, seg), x + seg)
    for c in (Condition.N2, Condition.N3):
        with pytest.raises(ValueError, match=r"requires floor_seg"):
            dsp.apply_condition(x, c, None)
    lp = dsp.apply_condition(x, Condition.LP1500, None)
    assert lp.size == x.size


def test_U_DSP_09_lp1500_is_delay_compensated() -> None:
    x = tone(500, 8000, amp=0.5)
    y = dsp.apply_condition(x, Condition.LP1500, None)
    assert np.max(np.abs(y[1000:7000] - x[1000:7000])) < 1e-3


# --------------------------------------------------------------------------- U-DSP-10


@pytest.fixture(scope="module")
def floors() -> dict[Condition, np.ndarray]:
    return {c: dsp.floor_buffer("ta", c) for c in (Condition.N2, Condition.N3)}


@pytest.mark.parametrize(
    ("cond", "targets"),
    [
        (Condition.N2, {500: -53.5, 1000: -53.0, 2000: -41.0}),
        (Condition.N3, {500: -38.5, 1000: -38.0, 2000: -26.0}),
    ],
)
def test_U_DSP_10_floor_band_levels(
    floors: dict[Condition, np.ndarray], cond: Condition, targets: dict[int, float]
) -> None:
    y = floors[cond]
    assert y.size == 960000
    for fc, t in targets.items():
        assert dsp.third_octave_level_dbfs(y, float(fc)) == pytest.approx(t, abs=1.0), fc
    assert dsp.third_octave_level_dbfs(y, 1000.0) == pytest.approx(targets[1000], abs=1e-9)


def test_U_DSP_10_floor_psd_band_limited() -> None:
    f = np.fft.rfftfreq(1024, 1 / FS)
    for c in (Condition.N2, Condition.N3):
        p = dsp.floor_psd(f, c)
        assert np.all(p[f < 250] == 0.0)
        assert np.all(p[f > 3400] == 0.0)
        assert np.all(p[(f >= 250) & (f <= 3400)] > 0.0)
    with pytest.raises(ValueError, match=r"N2/N3 only"):
        dsp.floor_psd(f, Condition.NORMAL)


def test_U_DSP_10_t0_and_hl_tables() -> None:
    f = np.array([125.0, 1000.0, 3400.0, 8000.0, 20.0, 12000.0])
    assert dsp.threshold_t0_db_spl(f).tolist()[:2] == [45.0, 7.0]
    assert dsp.threshold_t0_db_spl(f)[4] == 45.0
    assert dsp.threshold_t0_db_spl(f)[5] == 13.0
    hl = dsp.hl_db(Condition.N2, np.array([100.0, 1000.0, 3400.0, 6000.0]))
    assert hl.tolist() == [20.0, 25.0, 42.175, 42.175]
    expected_3400 = 40 + 5 * math.log(3400 / 3000) / math.log(4000 / 3000)
    assert expected_3400 == pytest.approx(42.175, abs=1e-3)
    assert dsp.hl_db(Condition.NORMAL, f).tolist() == [0.0] * 6
    with pytest.raises(ValueError, match=r"no hearing-loss model"):
        dsp.hl_db(Condition.LP1500, f)


# --------------------------------------------------------------------------- U-DSP-11


def test_U_DSP_11_prompt_chain_48k() -> None:
    x48 = tone(1000, 48000, fs=48000, amp=0.3)
    out = dsp.prompt_chain_48k(x48)
    assert len(out) == 8000
    lvl = dsp.dbfs(dsp.pcm16_to_float(dsp.ulaw_decode(out)))
    assert lvl == pytest.approx(-20.0, abs=0.3)
    assert out == dsp.prompt_chain_16k(dsp.resample_48k_to_16k(x48))


def test_phone_chain_lengths() -> None:
    data, z = dsp.phone_chain(tone(1000, 16000, amp=0.1))
    assert len(data) == 8000
    assert z.size == 8000


# --------------------------------------------------------------------------- uplink low-pass


def test_uplink_lowpass_127_taps_response() -> None:
    """contracts §9.4: 127 taps give -0.05 dB at 3.0 kHz and <= -56 dB at 4.4 kHz (48 kHz ctx)."""
    h = dsp.uplink_lowpass_taps(48000)
    assert h.size == 127
    assert np.allclose(h, h[::-1])
    _, resp = signal.freqz(h, worN=[3000, 4400], fs=48000)
    got = 20 * np.log10(np.abs(resp))
    assert got[0] == pytest.approx(-0.05, abs=0.02)
    assert got[1] <= -56.0
    with pytest.raises(ValueError, match=r"twice the cutoff"):
        dsp.uplink_lowpass_taps(7000)


def test_mix_at_snr_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match=r"same length"):
        dsp.mix_at_snr(np.ones(1000), np.ones(999), 0.0, 0.1)
