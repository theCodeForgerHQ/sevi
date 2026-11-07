"""LTASS noise regeneration (A spec §3.5, P5).

``regenerate`` loads the RMS-equalised digits of a bank language dir, applies the per-digit
offsets, computes their LTASS, synthesises 60 s of noise with the bank's fixed seed
(``derive_seed(f"bank:{lang}", "noise")``) and writes ``noise.wav`` (PCM_16, -23 dBFS).
Same phases for every offset set, so variants differ only in spectral shape.

CLI::

    uv run python -m sevi.calibration.noise --lang ta [--dir stimuli/ta]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from sevi.calibration.testbank import NOISE_RMS_DBFS, write_wav16
from sevi.contracts.enums import Lang
from sevi.contracts.languages import DIGIT_SETS
from sevi.contracts.rng import derive_seed
from sevi.core import dsp
from sevi.core.dsp import F64

NOISE_FILE: Final = "noise.wav"
# P-DSP-05 for real banks: ALL 11 one-third-octave bands, no relevance filter (unlike the
# tone test bank's test-only restriction in tests/property/a/test_dsp_props.py).
LTASS_BANDS_HZ: Final = (
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
)
LTASS_TOLERANCE_DB: Final = 2.0


def noise_seed(lang: Lang) -> int:
    return derive_seed(f"bank:{lang.value}", "noise")


def _read_digit(path: Path) -> F64:
    pcm, fs = sf.read(str(path), dtype="int16", always_2d=False)
    if fs != dsp.FS_MIX or np.ndim(pcm) != 1:
        raise ValueError(f"{path}: must be 16 kHz mono")
    return dsp.pcm16_to_float(np.asarray(pcm, dtype=np.int16))


def regenerate(
    bank_dir: Path, lang: Lang, offsets_db: Mapping[int, float] | None = None
) -> tuple[Path, str, int]:
    """Write ``{bank_dir}/noise.wav``; return ``(path, sha256, seed)``."""
    offsets_db = offsets_db or {}
    signals = [
        _read_digit(bank_dir / "digits" / f"{d}.wav") * 10.0 ** (offsets_db.get(d, 0.0) / 20.0)
        for d in DIGIT_SETS[lang]
    ]
    f, p = dsp.ltass_psd(signals)
    seed = noise_seed(lang)
    y = dsp.scale_to_rms_dbfs(dsp.synth_noise(f, p, dsp.FLOOR_BUFFER_SAMPLES, seed), NOISE_RMS_DBFS)
    path = bank_dir / NOISE_FILE
    data = write_wav16(path, y)
    return path, hashlib.sha256(data).hexdigest(), seed


def ltass_band_diffs(
    bank_dir: Path, lang: Lang, offsets_db: Mapping[int, float] | None = None
) -> dict[float, float]:
    """P-DSP-05: noise minus digit-LTASS level (dB) in every band of ``LTASS_BANDS_HZ``.

    Unfiltered signals (no band-pass), all 11 bands 315-3150 Hz, offsets applied as in
    ``regenerate``.
    """
    offsets_db = offsets_db or {}
    speech = np.concatenate(
        [
            _read_digit(bank_dir / "digits" / f"{d}.wav") * 10.0 ** (offsets_db.get(d, 0.0) / 20.0)
            for d in DIGIT_SETS[lang]
        ]
    )
    noise = _read_digit(bank_dir / NOISE_FILE)
    return {
        fc: dsp.third_octave_level_dbfs(noise, fc) - dsp.third_octave_level_dbfs(speech, fc)
        for fc in LTASS_BANDS_HZ
    }


def ltass_failures(diffs: Mapping[float, float]) -> list[str]:
    return [
        f"{fc:.0f} Hz: {d:+.2f} dB"
        for fc, d in diffs.items()
        if not np.isfinite(d) or abs(d) > LTASS_TOLERANCE_DB
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sevi.calibration.noise")
    ap.add_argument("--lang", required=True, choices=[lg.value for lg in Lang])
    ap.add_argument("--dir", default=None)
    a = ap.parse_args(argv)
    lang = Lang(a.lang)
    path, sha, seed = regenerate(Path(a.dir or f"stimuli/{lang.value}"), lang)
    print(json.dumps({"step": "noise", "path": str(path), "sha256": sha, "seed": seed}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
