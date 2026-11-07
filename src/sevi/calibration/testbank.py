"""Deterministic synthetic test bank (A spec §3.4, §3.5). CI only; never shippable.

Digit ``d`` = 400 ms of ``sum_{h=1..3} sin(2*pi*h*(200+50d)*t)/h`` with 5 ms raised-cosine fades,
equalised to -23 dBFS. Noise: LTASS of the digits, ``synth_noise`` seeded with
``derive_seed("bank:test", "noise")``, 60 s, -23 dBFS. ``experimental=True`` and voice
``TESTTONE``; its hash is only ever allowed by tests.

Regenerate (byte-identical on the pinned platform)::

    uv run python -m sevi.calibration.testbank stimuli/test-bank

``build_bank`` also builds fixture banks for tests (custom digit lengths or waveforms).
"""

from __future__ import annotations

import hashlib
import json
import sys
import wave
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from sevi.contracts.enums import BankVariant, Lang, ModelVersion, OffsetSource
from sevi.contracts.languages import DIGIT_SETS, STIMULUS_WORDS
from sevi.contracts.rng import derive_seed
from sevi.contracts.stimulus import (
    BankManifest,
    CalibrationProvenance,
    DigitAsset,
    TimbreRenderConfig,
)
from sevi.core import dsp, triplets
from sevi.core.bank import compute_bank_hash
from sevi.core.dsp import F64

DIGIT_RMS_DBFS: Final = -23.0  # [A-D04]
NOISE_RMS_DBFS: Final = -23.0
FADE_SAMPLES: Final = 80  # 5 ms
TEST_DIGIT_SAMPLES: Final = 6400  # 400 ms
TEST_CREATED_AT: Final = datetime(2026, 11, 4, tzinfo=UTC)
TEST_NOISE_SCOPE: Final = "bank:test"

DigitFn = Callable[[int, int], F64]  # (digit, n_samples) -> raw waveform


def tone_digit(d: int, n: int = TEST_DIGIT_SAMPLES) -> F64:
    """Three-harmonic tone at ``200 + 50 d`` Hz (unscaled, unfaded)."""
    t = np.arange(n, dtype=np.float64) / dsp.FS_MIX
    f0 = 200.0 + 50.0 * d
    out: F64 = sum(
        (np.sin(2 * np.pi * h * f0 * t) / h for h in range(1, 4)),
        start=np.zeros(n, dtype=np.float64),
    )
    return out


def finish_digit(x: F64) -> F64:
    """5 ms fades, then -23 dBFS (A spec §3.3 steps 5-6)."""
    return dsp.scale_to_rms_dbfs(
        x * dsp.raised_cosine_envelope(x.size, FADE_SAMPLES), DIGIT_RMS_DBFS
    )


def write_wav16(path: Path, x: F64) -> bytes:
    """PCM_16 16 kHz mono via the stdlib (header has no timestamps: byte-deterministic)."""
    pcm = dsp.float_to_pcm16(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(dsp.FS_MIX)
        w.writeframes(pcm.astype("<i2").tobytes())
    return path.read_bytes()


def build_bank(
    bank_dir: Path,
    *,
    lengths: Mapping[int, int] | None = None,
    digit_fn: DigitFn = tone_digit,
    offsets_db: Mapping[int, float] | None = None,
    frozen: bool = True,
    lang: Lang = Lang.TA,
) -> BankManifest:
    """Write ``{bank_dir}/{lang}/{manifest.json,digits/<d>.wav,noise.wav}``; return the manifest."""
    root = bank_dir / lang.value
    digit_set = DIGIT_SETS[lang]
    lengths = lengths or {}
    offsets_db = offsets_db or {}
    stored: list[F64] = []
    assets: list[DigitAsset] = []
    for d in digit_set:
        n = lengths.get(d, TEST_DIGIT_SAMPLES)
        x = finish_digit(digit_fn(d, n))
        data = write_wav16(root / "digits" / f"{d}.wav", x)
        y = dsp.pcm16_to_float(dsp.float_to_pcm16(x))
        stored.append(y)
        assets.append(
            DigitAsset(
                digit=d,
                word=STIMULUS_WORDS[lang][d],
                file=f"digits/{d}.wav",
                sha256=hashlib.sha256(data).hexdigest(),
                n_samples=n,
                rms_dbfs=round(dsp.dbfs(y), 6),
                offset_db=offsets_db.get(d, 0.0),
                onset_trim_ms=0.0,
            )
        )
    # §3.5: LTASS of the offset-corrected digits, fixed phases, -23 dBFS
    corrected = [
        s * 10.0 ** (offsets_db.get(d, 0.0) / 20.0) for d, s in zip(digit_set, stored, strict=True)
    ]
    f, p = dsp.ltass_psd(corrected)
    seed = derive_seed(TEST_NOISE_SCOPE, "noise")
    noise = dsp.scale_to_rms_dbfs(
        dsp.synth_noise(f, p, dsp.FLOOR_BUFFER_SAMPLES, seed), NOISE_RMS_DBFS
    )
    noise_bytes = write_wav16(root / "noise.wav", noise)
    m = BankManifest(
        lang=lang,
        bank_version=f"{lang.value}-0.0.1",
        variant=BankVariant.SHIPPED,
        model_version=ModelVersion.V3,
        timbre=TimbreRenderConfig(
            model="timbre-v3.0", voice="TESTTONE", language="ta-IN" if lang is Lang.TA else "en-IN"
        ),
        digit_set=digit_set,
        digits=tuple(assets),
        noise_sha256=hashlib.sha256(noise_bytes).hexdigest(),
        noise_seed=seed,
        design=triplets.design(digit_set),
        calibration=CalibrationProvenance(
            source=OffsetSource.RMS,
            iterations=0,
            prisma_model=None,
            prisma_format=None,
            keyterms_sha256=None,
            prisma_sample_rate_hz=None,
            listener_fit_sha256=None,
            panel_outcome_sha256=None,
        ),
        experimental=True,
        created_at=TEST_CREATED_AT,
        frozen=frozen,
        frozen_at=TEST_CREATED_AT if frozen else None,
        bank_hash="0" * 64,
    )
    m = m.model_copy(update={"bank_hash": compute_bank_hash(m)})
    write_manifest(root / "manifest.json", m)
    return m


def write_manifest(path: Path, m: BankManifest) -> None:
    text = json.dumps(m.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0] if args else "stimuli/test-bank")
    m = build_bank(out)
    print(json.dumps({"step": "test-bank", "bank_hash": m.bank_hash, "dir": str(out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
