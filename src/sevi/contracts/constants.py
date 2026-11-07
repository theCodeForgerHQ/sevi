"""DIN and protocol constants (contracts §4.3).

These are not configuration: changing one is a contract change (bump ``CONTRACTS_VERSION``).
"""

from typing import Final

# --- audio chain (R2 P4-P7, ADR-003/004)
SAMPLE_RATE_WIRE_HZ: Final = 8000
SAMPLE_RATE_MIX_HZ: Final = 16000
TIMBRE_RENDER_SAMPLE_RATE_HZ: Final = 48000  # render at 48 kHz wav, resample_poly(1, 3) to 16 kHz
NOISE_LEAD_MS: Final = 500.0
INTER_DIGIT_GAP_MS: Final = 200.0
NOISE_LAG_MS: Final = 500.0
RAMP_MS: Final = 20.0  # raised-cosine on/off of the noise burst
MIX_RMS_DBFS: Final = -20.0  # RMS over the whole clip (speech+noise)
PEAK_MAX_DBFS: Final = -3.0  # if exceeded, scale down to -3 dBFS peak; set level_limited
BANDPASS_LOW_HZ: Final = 300.0
BANDPASS_HIGH_HZ: Final = 3400.0
# scipy.signal.butter(4, [300, 3400], "bandpass", fs=16000, output="sos"); sosfilt (causal)
BANDPASS_ORDER: Final = 4
RESAMPLE_16K_TO_8K: Final = (1, 2)  # scipy.signal.resample_poly(x, 1, 2), default kaiser(5.0)
NOISE_BUFFER_S: Final = 60.0
MULAW_SILENCE_BYTE: Final = 0xFF
# G.711 mu-law: ITU-T G.711 segment encoder (bias 0x84, clip 32635), numpy lookup table;
# one implementation in core/dsp.py

# --- triplet design (ADR-003; G19)
POOL_SIZE: Final = {9: 126, 8: 120}
CAL_SUBSET_SIZE: Final = {9: 54, 8: 56}
OFFSET_PAIRS: Final = {
    9: (
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
        (5, 1),
        (6, 3),
        (7, 5),
        (8, 7),
        (1, 3),
        (2, 6),
        (4, 3),
        (5, 6),
        (7, 3),
        (8, 6),
    ),  # fmt: skip
    8: (
        (1, 2),
        (2, 4),
        (3, 6),
        (5, 2),
        (6, 4),
        (7, 6),
        (1, 3),
        (2, 6),
        (3, 1),
        (5, 7),
        (6, 2),
        (7, 5),
        (1, 5),
        (3, 7),
        (5, 1),
    ),  # fmt: skip
}
# pool[i] for i = p * n + a  (p = pair index, a = 0..n-1):
#   (digit_set[a], digit_set[(a + d1) % n], digit_set[(a + d2) % n])
# Calibration subset = pool[0 : CAL_SUBSET_SIZE[n]]  (first 6 pairs for n=9, first 7 for n=8)
# => each pair contributes every digit exactly once per position, so:
#    n=9: 14 per position (42 per digit) in the pool, 6 per position (18 per digit) in the subset;
#    n=8: 15 per position (45 per digit) in the pool, 7 per position (21 per digit) in the subset.

# --- adaptive procedure (ADR-006, R2 P8-P11, L1 §3.1); worked example §4.3.1 (U-ST-010)
START_SNR_DB: Final = 0.0
STEP_INITIAL_DB: Final = 4.0  # every move made while no scored trial has been wrong
STEP_DB: Final = 2.0  # from (and including) the move caused by the first wrong scored trial
SNR_MIN_DB: Final = -20.0
SNR_MAX_DB: Final = 10.0
N_SCORED: Final = 24
SRT_FIRST_TRIAL: Final = 5  # SRT = mean of SNRs of trials 5..24 plus virtual 25th (21 values)
VOLUME_SNR_DB: Final = 10.0
VOLUME_N: Final = 2
VOLUME_ROUNDS: Final = 2
VOLUME_PASS_MIN: Final = 1  # >= 1 of 2 correct passes a round
PRACTICE_SNRS_DB: Final = (6.0, 2.0, -2.0)
PRACTICE_ROUNDS: Final = 2
PRACTICE_PASS_MIN: Final = 2
LOUD_SNR_DB: Final = 10.0
LOUD_N: Final = 2
NO_RESPONSE_RUN: Final = 3
MAX_LOUD_CHECKS: Final = 2
LOUD_WRONG_REFER: Final = 2  # cumulative wrong loud triplets across the session -> LOUD_FAILED
CEILING_CONSECUTIVE: Final = 6
CEILING_MIN_VALID: Final = 4
CEILING_MAX_CORRECT: Final = 1
REVERSALS_MIN: Final = 4
FLAG_LIMIT: Final = 3  # flags > 3 -> INCONCLUSIVE

# --- result (ADR-011)
NORM_MIN_N: Final = 6
NORM_V1_N: Final = 10
NH95_Z: Final = 1.645  # NH95 = mean + 1.645 * sd (ddof=1)
# srt_db >= 0 dB SNR -> REFER (T10b) in every context; >= 6 dB above every published
# normal-hearing DIN mean, telephone penalty included
ABS_REFER_DB: Final = 0.0
REFER_MARGIN_DB: Final = 3.0
V0_BORDER_HALFWIDTH_DB: Final = 1.0

# --- Prisma framing (R1 §1.2)
PRISMA_FRAME_BYTES: Final = 1024  # 512 samples s16le
PRISMA_FRAME_MS_8K: Final = 64.0
