"""Signal processing for the DIN test (A spec §1, contracts §4.3).

Pure module: no I/O, no clock, no global mutable state. Float audio is ``float64`` at full scale
±1.0. Every public function validates its numeric input and raises ``ValueError`` naming the
failed check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import signal  # type: ignore[import-untyped]

from sevi.contracts.constants import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BANDPASS_ORDER,
    MIX_RMS_DBFS,
    PEAK_MAX_DBFS,
    RAMP_MS,
    RESAMPLE_16K_TO_8K,
    SAMPLE_RATE_MIX_HZ,
    SAMPLE_RATE_WIRE_HZ,
    TIMBRE_RENDER_SAMPLE_RATE_HZ,
)
from sevi.contracts.enums import Condition
from sevi.contracts.rng import derive_seed

F64 = npt.NDArray[np.float64]
I16 = npt.NDArray[np.int16]
U8 = npt.NDArray[np.uint8]

FS_MIX: Final = SAMPLE_RATE_MIX_HZ
SPL_AT_MINUS20_DBFS: Final = 65.0  # ADR-012 mapping assumption: -20 dBFS mix == 65 dB SPL
DBFS_TO_SPL_OFFSET: Final = 85.0  # SPL = dBFS + 85 under that assumption
SILENT_DBFS: Final = -200.0  # dbfs() of an all-zero array

RAMP_SAMPLES: Final = int(RAMP_MS * FS_MIX / 1000)  # 320
LEAD_SAMPLES: Final = 8000  # 500 ms at 16 kHz
GAP_SAMPLES: Final = 3200  # 200 ms
LAG_SAMPLES: Final = 8000  # 500 ms
FLOOR_BUFFER_SAMPLES: Final = 960000  # 60 s at 16 kHz

# G.711 mu-law (ITU-T), bias 0x84, clip 32635
ULAW_BIAS: Final = 0x84
ULAW_CLIP: Final = 32635

# LP1500 simulated condition (A §1.3)
LP1500_NUMTAPS: Final = 1025
LP1500_CUTOFF_HZ: Final = 1500.0

# Browser uplink anti-alias low-pass (contracts §9.4, CCR-v3-35): reference taps for B's worklet
UPLINK_LOWPASS_NUMTAPS: Final = 127
UPLINK_LOWPASS_CUTOFF_HZ: Final = 3600.0

# Simulated hearing loss, Bisgaard 2010 (A §1.3)
HL_FREQS_HZ: Final = (250.0, 375.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 3400.0)
HL_N2_DB: Final = (20.0, 20.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0, 42.175)
HL_N3_DB: Final = (35.0, 35.0, 35.0, 35.0, 40.0, 45.0, 50.0, 55.0, 57.175)

# Threshold T0, dB SPL (TDH-39 RETSPL, ANSI S3.6) (A §1.4)
T0_FREQS_HZ: Final = (
    125.0,
    250.0,
    500.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
    3000.0,
    4000.0,
    6000.0,
    8000.0,
)
T0_DB_SPL: Final = (45.0, 25.5, 11.5, 8.0, 7.0, 6.5, 9.0, 10.0, 9.5, 15.5, 13.0)
FLOOR_BAND_LOW_HZ: Final = 250.0
FLOOR_BAND_HIGH_HZ: Final = 3400.0
THIRD_OCTAVE_BW_FACTOR: Final = 0.2316

WELCH_NPERSEG: Final = 1024
WELCH_NOVERLAP: Final = 512


# --------------------------------------------------------------------------- validation


def _check_float(x: F64, name: str = "x") -> F64:
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name}: not a numpy array")
    if x.dtype != np.float64:
        raise ValueError(f"{name}: dtype must be float64, got {x.dtype}")
    if x.ndim != 1:
        raise ValueError(f"{name}: must be 1-D, got ndim={x.ndim}")
    if x.size == 0:
        raise ValueError(f"{name}: empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: non-finite values")
    return x


def _check_int16(x: I16, name: str = "pcm") -> I16:
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name}: not a numpy array")
    if x.dtype != np.int16:
        raise ValueError(f"{name}: dtype must be int16, got {x.dtype}")
    if x.ndim != 1:
        raise ValueError(f"{name}: must be 1-D, got ndim={x.ndim}")
    if x.size == 0:
        raise ValueError(f"{name}: empty")
    return x


def _check_finite_scalar(v: float, name: str) -> float:
    if not math.isfinite(v):
        raise ValueError(f"{name}: non-finite")
    return float(v)


# --------------------------------------------------------------------------- levels


def rms(x: F64) -> float:
    """``sqrt(mean(x**2))`` over all samples."""
    _check_float(x)
    return float(np.sqrt(np.mean(x * x)))


def dbfs(x: F64) -> float:
    """RMS level in dBFS (0 dBFS = RMS 1.0); ``-200.0`` for an all-zero array."""
    r = rms(x)
    if r == 0.0:
        return SILENT_DBFS
    return 20.0 * math.log10(r)


def scale_to_rms_dbfs(x: F64, target_dbfs: float) -> F64:
    _check_finite_scalar(target_dbfs, "target_dbfs")
    r = rms(x)
    if r == 0.0:
        raise ValueError("silent input")
    out: F64 = x * (10.0 ** (target_dbfs / 20.0) / r)
    return out


def peak_limit(x: F64, peak_max_dbfs: float) -> tuple[F64, bool]:
    """Scale ``x`` down to a ``peak_max_dbfs`` peak if it exceeds it (RMS first, headroom wins)."""
    _check_float(x)
    _check_finite_scalar(peak_max_dbfs, "peak_max_dbfs")
    p = float(np.max(np.abs(x)))
    lim = 10.0 ** (peak_max_dbfs / 20.0)
    if p > lim:
        out: F64 = x * (lim / p)
        return out, True
    return x, False


def raised_cosine_envelope(n: int, ramp: int) -> F64:
    if ramp < 0 or n <= 0:
        raise ValueError("n must be > 0 and ramp >= 0")
    if n < 2 * ramp:
        raise ValueError("n must be >= 2*ramp")
    env: F64 = np.ones(n, dtype=np.float64)
    if ramp == 0:
        return env
    k = np.arange(ramp, dtype=np.float64)
    r = 0.5 * (1.0 - np.cos(np.pi * k / ramp))
    env[:ramp] = r
    env[n - ramp :] = r[::-1]
    return env


# --------------------------------------------------------------------------- resampling, filters


def _resample(x: F64, up: int, down: int) -> F64:
    _check_float(x)
    y = signal.resample_poly(x, up, down, window=("kaiser", 5.0), padtype="constant")
    return np.asarray(y, dtype=np.float64)


def resample_48k_to_16k(x: F64) -> F64:
    return _resample(x, 1, TIMBRE_RENDER_SAMPLE_RATE_HZ // SAMPLE_RATE_MIX_HZ)


def resample_16k_to_8k(x: F64) -> F64:
    up, down = RESAMPLE_16K_TO_8K
    return _resample(x, up, down)


@lru_cache(maxsize=1)
def _bandpass_sos_cached() -> F64:
    sos = signal.butter(
        BANDPASS_ORDER,
        [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
        btype="bandpass",
        fs=FS_MIX,
        output="sos",
    )
    arr = np.asarray(sos, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def bandpass_sos() -> F64:
    """Butterworth order-4 band-pass 300-3400 Hz at 16 kHz, shape (4, 6). Read-only, cached."""
    return _bandpass_sos_cached()


def bandpass(x: F64) -> F64:
    """Causal ``sosfilt`` with zero initial state."""
    _check_float(x)
    return np.asarray(signal.sosfilt(np.array(bandpass_sos()), x), dtype=np.float64)


def uplink_lowpass_taps(fs_in: int = TIMBRE_RENDER_SAMPLE_RATE_HZ) -> F64:
    """127-tap Hamming-windowed sinc low-pass, cutoff 3.6 kHz, for the browser uplink decimator.

    Reference implementation of the capture worklet filter (contracts §9.4); B's JS must match it.
    """
    if fs_in <= 2 * UPLINK_LOWPASS_CUTOFF_HZ:
        raise ValueError("fs_in must exceed twice the cutoff")
    h = signal.firwin(UPLINK_LOWPASS_NUMTAPS, UPLINK_LOWPASS_CUTOFF_HZ, window="hamming", fs=fs_in)
    return np.asarray(h, dtype=np.float64)


# --------------------------------------------------------------------------- PCM and G.711 mu-law


def float_to_pcm16(x: F64) -> I16:
    _check_float(x)
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)


def pcm16_to_float(x: I16) -> F64:
    _check_int16(x)
    return x.astype(np.float64) / 32768.0


def _ulaw_encode_sample(s: int) -> int:
    sign = 0x80 if s < 0 else 0
    m = min(abs(s), ULAW_CLIP) + ULAW_BIAS
    e = (m >> 7).bit_length() - 1  # floor(log2(m >> 7)); m >= 132 so m >> 7 >= 1
    mant = (m >> (e + 3)) & 0x0F
    return ~(sign | (e << 4) | mant) & 0xFF


def _ulaw_decode_byte(b: int) -> int:
    u = ~b & 0xFF
    e = (u >> 4) & 7
    mant = u & 0x0F
    v = (((mant << 3) + ULAW_BIAS) << e) - ULAW_BIAS
    return -v if u & 0x80 else v


def _build_encode_table() -> U8:
    t = np.array([_ulaw_encode_sample(s) for s in range(-32768, 32768)], dtype=np.uint8)
    t.setflags(write=False)
    return t


def _build_decode_table() -> I16:
    t = np.array([_ulaw_decode_byte(b) for b in range(256)], dtype=np.int16)
    t.setflags(write=False)
    return t


ULAW_ENCODE_TABLE: Final[U8] = _build_encode_table()  # index = int16 + 32768
ULAW_DECODE_TABLE: Final[I16] = _build_decode_table()  # index = byte


def ulaw_encode(pcm: I16) -> bytes:
    _check_int16(pcm)
    idx = pcm.astype(np.int32) + 32768
    return ULAW_ENCODE_TABLE[idx].tobytes()


def ulaw_decode(data: bytes) -> I16:
    if len(data) == 0:
        raise ValueError("data: empty")
    idx = np.frombuffer(data, dtype=np.uint8)
    return ULAW_DECODE_TABLE[idx].copy()


# --------------------------------------------------------------------------- spectra, noise


def _welch(x: F64, fs: int) -> tuple[F64, F64]:
    f, p = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=WELCH_NPERSEG,
        noverlap=WELCH_NOVERLAP,
        detrend=False,
        scaling="density",
        average="mean",
    )
    return np.asarray(f, dtype=np.float64), np.asarray(p, dtype=np.float64)


def ltass_psd(signals: Sequence[F64]) -> tuple[F64, F64]:
    """Welch LTASS of the concatenated signals: ``(freqs_hz[513], psd[513])``."""
    if len(signals) == 0:
        raise ValueError("signals: empty")
    for i, s in enumerate(signals):
        _check_float(s, f"signals[{i}]")
    c = np.concatenate(list(signals))
    if c.size < WELCH_NPERSEG:
        raise ValueError(f"signals: need >= {WELCH_NPERSEG} samples in total")
    return _welch(c, FS_MIX)


def synth_noise(freqs: F64, psd: F64, n: int, seed: int) -> F64:
    """Random-phase noise with the given PSD shape, unit RMS, circular buffer of ``n`` samples."""
    _check_float(freqs, "freqs")
    _check_float(psd, "psd")
    if freqs.shape != psd.shape:
        raise ValueError("freqs and psd must have the same shape")
    if (psd < 0).any():
        raise ValueError("psd: negative values")
    if n <= 0 or n % 2 != 0:
        raise ValueError("n must be positive and even")
    fk = np.fft.rfftfreq(n, 1.0 / FS_MIX)
    a = np.sqrt(np.interp(fk, freqs, psd))
    a[0] = 0.0
    a[-1] = 0.0
    phi = 2.0 * np.pi * np.random.Generator(np.random.PCG64(seed)).random(n // 2 + 1)
    y = np.asarray(np.fft.irfft(a * np.exp(1j * phi), n=n), dtype=np.float64)
    r = float(np.sqrt(np.mean(y * y)))
    if r == 0.0:
        raise ValueError("psd: zero everywhere")
    out: F64 = y / r
    return out


def third_octave_level_dbfs(x: F64, fc: float, fs: int = FS_MIX) -> float:
    """Band level of ``x`` (dBFS) in the one-third-octave band centred on ``fc``."""
    _check_float(x)
    _check_finite_scalar(fc, "fc")
    if fc <= 0:
        raise ValueError("fc must be > 0")
    if x.size < WELCH_NPERSEG:
        raise ValueError(f"x: need >= {WELCH_NPERSEG} samples")
    f, p = _welch(x, fs)
    df = float(f[1] - f[0])
    sel = (f >= fc * 2.0 ** (-1.0 / 6.0)) & (f < fc * 2.0 ** (1.0 / 6.0))
    total = float(np.sum(p[sel]) * df)
    if total <= 0.0:
        return SILENT_DBFS
    return 10.0 * math.log10(total)


# --------------------------------------------------------------------------- triplet assembly, mix


def assemble_triplet(
    digits: Sequence[F64], gains_db: Sequence[float]
) -> tuple[F64, tuple[tuple[int, int], ...]]:
    """lead 500 ms, d1, gap 200 ms, d2, gap, d3, lag 500 ms at 16 kHz; returns (track, spans)."""
    if len(digits) != 3 or len(gains_db) != 3:
        raise ValueError("exactly three digits and three gains")
    parts: list[F64] = [np.zeros(LEAD_SAMPLES, dtype=np.float64)]
    spans: list[tuple[int, int]] = []
    pos = LEAD_SAMPLES
    for j, (d, g_db) in enumerate(zip(digits, gains_db, strict=True)):
        _check_float(d, f"digits[{j}]")
        _check_finite_scalar(g_db, f"gains_db[{j}]")
        if d.size % 2 != 0:
            raise ValueError(f"digits[{j}]: length must be even")
        if j > 0:
            parts.append(np.zeros(GAP_SAMPLES, dtype=np.float64))
            pos += GAP_SAMPLES
        parts.append(d * (10.0 ** (g_db / 20.0)))
        spans.append((pos, pos + d.size))
        pos += d.size
    parts.append(np.zeros(LAG_SAMPLES, dtype=np.float64))
    return np.concatenate(parts), tuple(spans)


def mix_at_snr(speech: F64, noise_seg: F64, snr_db: float, ref_rms: float) -> tuple[F64, F64, F64]:
    """Add ramped noise at ``snr_db`` re ``ref_rms``; returns ``(mix, speech, noise)``."""
    _check_float(speech, "speech")
    _check_float(noise_seg, "noise_seg")
    _check_finite_scalar(snr_db, "snr_db")
    _check_finite_scalar(ref_rms, "ref_rms")
    if noise_seg.size != speech.size:
        raise ValueError("noise_seg and speech must have the same length")
    if ref_rms <= 0:
        raise ValueError("ref_rms must be > 0")
    r = rms(noise_seg)
    if r == 0.0:
        raise ValueError("silent input")
    g_n = (ref_rms / r) * 10.0 ** (-snr_db / 20.0)
    noise: F64 = g_n * noise_seg * raised_cosine_envelope(speech.size, RAMP_SAMPLES)
    mix: F64 = speech + noise
    return mix, speech, noise


# --------------------------------------------------------------------------- simulated loss


def _interp_log2(f: F64, xs: Sequence[float], ys: Sequence[float]) -> F64:
    _check_float(f, "f")
    if (f <= 0).any():
        raise ValueError("f: must be > 0")
    out = np.interp(np.log2(f), np.log2(np.asarray(xs)), np.asarray(ys, dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


def threshold_t0_db_spl(f: F64) -> F64:
    """Normal-hearing threshold (dB SPL), linear in log2(f), held outside 125-8000 Hz."""
    return _interp_log2(f, T0_FREQS_HZ, T0_DB_SPL)


def hl_db(condition: Condition, f: F64) -> F64:
    """Bisgaard hearing loss (dB HL) for N2/N3; zero for NORMAL. LP1500 has no HL model."""
    if condition is Condition.N2:
        return _interp_log2(f, HL_FREQS_HZ, HL_N2_DB)
    if condition is Condition.N3:
        return _interp_log2(f, HL_FREQS_HZ, HL_N3_DB)
    if condition is Condition.NORMAL:
        _check_float(f, "f")
        return np.zeros_like(f)
    raise ValueError(f"no hearing-loss model for {condition.value}")


@lru_cache(maxsize=1)
def _lp1500_taps() -> F64:
    h = signal.firwin(LP1500_NUMTAPS, LP1500_CUTOFF_HZ, window="hamming", fs=FS_MIX)
    arr = np.asarray(h, dtype=np.float64)
    arr.setflags(write=False)
    return arr


def condition_taps(condition: Condition) -> F64 | None:
    """FIR taps for LP1500; ``None`` for NORMAL, N2, N3."""
    if condition is Condition.LP1500:
        return _lp1500_taps()
    return None


def floor_psd(freqs: F64, condition: Condition) -> F64:
    """Threshold-simulating floor PSD (dBFS scale, power per Hz); 0 outside 250-3400 Hz."""
    if condition not in (Condition.N2, Condition.N3):
        raise ValueError("floor_psd: N2/N3 only")
    _check_float(freqs, "freqs")
    out = np.zeros_like(freqs)
    band = (freqs >= FLOOR_BAND_LOW_HZ) & (freqs <= FLOOR_BAND_HIGH_HZ)
    fb = freqs[band]
    if fb.size:
        lvl = threshold_t0_db_spl(fb) + hl_db(condition, fb) - DBFS_TO_SPL_OFFSET
        out[band] = 10.0 ** ((lvl - 10.0 * np.log10(THIRD_OCTAVE_BW_FACTOR * fb)) / 10.0)
    return out


def floor_buffer(lang: str, condition: Condition, n: int = FLOOR_BUFFER_SAMPLES) -> F64:
    """60 s floor buffer for N2/N3, scaled so its 1 kHz 1/3-octave level is T0+HL-85 dBFS."""
    f_grid = np.fft.rfftfreq(WELCH_NPERSEG, 1.0 / FS_MIX).astype(np.float64)
    y = synth_noise(
        f_grid, floor_psd(f_grid, condition), n, derive_seed(f"floor:{lang}", condition.value)
    )
    f1k = np.array([1000.0])
    target = float(threshold_t0_db_spl(f1k)[0] + hl_db(condition, f1k)[0] - DBFS_TO_SPL_OFFSET)
    cur = third_octave_level_dbfs(y, 1000.0)
    out: F64 = y * 10.0 ** ((target - cur) / 20.0)
    return out


def apply_condition(x: F64, condition: Condition, floor_seg: F64 | None) -> F64:
    """Apply a simulated-loss condition after the fixed-level step. NORMAL returns ``x`` itself."""
    if condition is Condition.NORMAL:
        return x
    _check_float(x)
    if condition is Condition.LP1500:
        h = _lp1500_taps()
        half = (LP1500_NUMTAPS - 1) // 2
        y = signal.fftconvolve(x, h)[half : half + x.size]
        return np.asarray(y, dtype=np.float64)
    if floor_seg is None:
        raise ValueError(f"{condition.value} requires floor_seg")
    _check_float(floor_seg, "floor_seg")
    if floor_seg.size != x.size:
        raise ValueError("floor_seg and x must have the same length")
    out: F64 = x + floor_seg
    return out


# --------------------------------------------------------------------------- chains


def phone_chain(x16: F64) -> tuple[bytes, F64]:
    """band-pass -> 8 kHz -> mu-law; also returns the 8 kHz float signal."""
    y = bandpass(x16)
    z = resample_16k_to_8k(y)
    return ulaw_encode(float_to_pcm16(z)), z


def normalize_ulaw(
    data: bytes, rms_dbfs: float = MIX_RMS_DBFS, peak_max_dbfs: float = PEAK_MAX_DBFS
) -> bytes:
    x = pcm16_to_float(ulaw_decode(data))
    x = scale_to_rms_dbfs(x, rms_dbfs)
    x, _ = peak_limit(x, peak_max_dbfs)
    return ulaw_encode(float_to_pcm16(x))


def prompt_chain_16k(x16: F64) -> bytes:
    """Prompt chain (contracts §8.4 order): band-pass, -20 dBFS, peak <= -3 dBFS, 8 kHz, mu-law."""
    y = bandpass(x16)
    y = scale_to_rms_dbfs(y, MIX_RMS_DBFS)
    y, _ = peak_limit(y, PEAK_MAX_DBFS)
    z = resample_16k_to_8k(y)
    return ulaw_encode(float_to_pcm16(z))


def prompt_chain_48k(x48: F64) -> bytes:
    return prompt_chain_16k(resample_48k_to_16k(x48))


__all__ = [
    "DBFS_TO_SPL_OFFSET",
    "FS_MIX",
    "SAMPLE_RATE_WIRE_HZ",
    "SPL_AT_MINUS20_DBFS",
    "ULAW_DECODE_TABLE",
    "ULAW_ENCODE_TABLE",
    "apply_condition",
    "assemble_triplet",
    "bandpass",
    "bandpass_sos",
    "condition_taps",
    "dbfs",
    "float_to_pcm16",
    "floor_buffer",
    "floor_psd",
    "hl_db",
    "ltass_psd",
    "mix_at_snr",
    "normalize_ulaw",
    "pcm16_to_float",
    "peak_limit",
    "phone_chain",
    "prompt_chain_16k",
    "prompt_chain_48k",
    "raised_cosine_envelope",
    "resample_16k_to_8k",
    "resample_48k_to_16k",
    "rms",
    "scale_to_rms_dbfs",
    "synth_noise",
    "third_octave_level_dbfs",
    "threshold_t0_db_spl",
    "ulaw_decode",
    "ulaw_encode",
    "uplink_lowpass_taps",
]
