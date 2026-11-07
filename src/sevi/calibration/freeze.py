"""Bank freeze (A spec §4.3, reduced scope: RMS-equalised, no Prisma grid, no panel).

Reads the ``render.json`` record and ``digits/`` written by ``sevi.calibration.render``,
re-verifies every digit hash, regenerates ``noise.wav`` with zero offsets (§3.5), and writes a
SHIPPED ``manifest.json`` with ``calibration.source = RMS``, ``iterations = 0``, all Prisma and
panel provenance ``None``, ``frozen = True`` and the ``bank_hash`` (contracts §4.4). ``en`` is
always ``experimental = True``. Prints the ``SEVI_RESULT__FROZEN_BANK_HASHES`` fragment.

CLI (no network)::

    uv run python -m sevi.calibration.freeze --lang ta [--dir stimuli/ta] [--version ta-1.0.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sevi.calibration.noise import regenerate
from sevi.calibration.render import read_render_record
from sevi.calibration.testbank import write_manifest
from sevi.contracts.enums import BankVariant, Lang, ModelVersion, OffsetSource
from sevi.contracts.languages import DIGIT_SETS
from sevi.contracts.stimulus import BankManifest, CalibrationProvenance
from sevi.core import triplets
from sevi.core.bank import MANIFEST_NAME, compute_bank_hash

MODEL_VERSIONS = {"timbre-v3.0": ModelVersion.V3, "timbre-v2.5": ModelVersion.V2_5}


class FreezeError(Exception):
    pass


def freeze(
    bank_dir: Path,
    lang: Lang,
    *,
    bank_version: str | None = None,
    now: datetime | None = None,
) -> BankManifest:
    """Freeze ``bank_dir`` (a language dir, e.g. ``stimuli/ta``) and return the manifest."""
    rec_lang, timbre, assets = read_render_record(bank_dir)
    if rec_lang is not lang:
        raise FreezeError(f"render record is {rec_lang.value}, not {lang.value}")
    if tuple(a.digit for a in assets) != DIGIT_SETS[lang]:
        raise FreezeError("render record digits != the §4.13 digit set")
    for a in assets:
        if hashlib.sha256((bank_dir / a.file).read_bytes()).hexdigest() != a.sha256:
            raise FreezeError(f"{a.file} changed since render")
    assets = tuple(a.model_copy(update={"offset_db": 0.0}) for a in assets)  # RMS: zero offsets
    _, noise_sha, seed = regenerate(bank_dir, lang, {})
    ts = (now or datetime.now(UTC)).replace(microsecond=0)
    m = BankManifest(
        lang=lang,
        bank_version=bank_version or f"{lang.value}-1.0.0",
        variant=BankVariant.SHIPPED,
        model_version=MODEL_VERSIONS[timbre.model],
        timbre=timbre,
        digit_set=DIGIT_SETS[lang],
        digits=assets,
        noise_sha256=noise_sha,
        noise_seed=seed,
        design=triplets.design(DIGIT_SETS[lang]),
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
        experimental=lang is Lang.EN,
        created_at=ts,
        frozen=True,
        frozen_at=ts,
        bank_hash="0" * 64,
    )
    m = m.model_copy(update={"bank_hash": compute_bank_hash(m)})
    write_manifest(bank_dir / MANIFEST_NAME, m)
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sevi.calibration.freeze")
    ap.add_argument("--lang", required=True, choices=[lg.value for lg in Lang])
    ap.add_argument("--dir", default=None, help="bank language dir (default stimuli/<lang>)")
    ap.add_argument("--version", default=None, help="bank_version (default <lang>-1.0.0)")
    a = ap.parse_args(argv)
    lang = Lang(a.lang)
    try:
        m = freeze(Path(a.dir or f"stimuli/{lang.value}"), lang, bank_version=a.version)
    except (FreezeError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"step": "freeze", "bank_version": m.bank_version, "bank_hash": m.bank_hash}))
    print(json.dumps({"SEVI_RESULT__FROZEN_BANK_HASHES": {lang.value: m.bank_hash}}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
