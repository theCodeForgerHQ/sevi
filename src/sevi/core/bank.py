"""Load and verify a SHIPPED stimulus bank (A spec §4.1-§4.2, contracts §4.4, §5.1.1).

Construction I/O only (contracts §5.0): reads ``{bank_dir}/{lang}/manifest.json``, the digit WAVs
and ``noise.wav``, verifies every sha256 and the manifest ``bank_hash``, and refuses a bank whose
hash is not allowed (``BANK_HASH_REJECTED``). Panel variants are cut (BUILD-BRIEF reduced scope).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from pydantic import ValidationError

from sevi.contracts.enums import BankVariant, Lang
from sevi.contracts.errors import BankError, ErrorCode
from sevi.contracts.languages import DIGIT_SETS
from sevi.contracts.stimulus import BankManifest, ChainSpec
from sevi.core import dsp, triplets
from sevi.core.dsp import F64
from sevi.core.stimulus import NOISE_SAMPLES, BankStimulusProvider

MANIFEST_NAME: Final = "manifest.json"
TEST_VOICE: Final = "TESTTONE"  # synthetic CI test bank (A spec §3.4); never loadable in prod
PRISMA_UPSAMPLER: Final = {8000: "none", 16000: "resample_poly(2,1)"}
HASH_EXCLUDE: Final = frozenset({"bank_hash", "frozen", "frozen_at"})
# prisma_upsampler is fixed by prisma_sample_rate_hz (8k -> none, 16k -> resample_poly(2,1)).
CHAIN_FREE_FIELDS: Final = frozenset({"prisma_sample_rate_hz", "prisma_upsampler"})


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, no whitespace, UTF-8 (key order in the file never changes a hash)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def compute_bank_hash(m: BankManifest) -> str:
    """sha256 of the canonical JSON dump without ``bank_hash``, ``frozen``, ``frozen_at``."""
    return hashlib.sha256(
        canonical_json(m.model_dump(mode="json", exclude=set(HASH_EXCLUDE)))
    ).hexdigest()


def _integrity(msg: str) -> BankError:
    return BankError(ErrorCode.BANK_INTEGRITY, msg)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        raise _integrity(f"cannot read {path.name}: {e}") from e


def _read_wav(path: Path, n_expected: int) -> F64:
    try:
        info = sf.info(str(path))
        if info.format != "WAV" or info.subtype != "PCM_16":
            raise _integrity(f"{path.name}: not WAV PCM_16")
        if info.samplerate != dsp.FS_MIX or info.channels != 1:
            raise _integrity(f"{path.name}: must be 16000 Hz mono")
        if info.frames != n_expected:
            raise _integrity(f"{path.name}: {info.frames} samples, expected {n_expected}")
        pcm = np.asarray(sf.read(str(path), dtype="int16", always_2d=False)[0], dtype=np.int16)
    except BankError:
        raise
    except Exception as e:  # libsndfile errors on corrupt files
        raise _integrity(f"{path.name}: unreadable WAV ({e})") from e
    return dsp.pcm16_to_float(pcm)


def read_manifest(path: Path) -> BankManifest:
    """Parse and validate a manifest; any failure is ``BANK_INTEGRITY``."""
    try:
        return BankManifest.model_validate_json(path.read_bytes())
    except OSError as e:
        raise _integrity(f"cannot read manifest: {e}") from e
    except ValidationError as e:
        raise _integrity(f"invalid manifest: {e.error_count()} error(s)") from e


def load_bank(
    lang: Lang,
    bank_dir: str,
    allowed_hashes: dict[Lang, str],
    allow_unfrozen: bool,
    *,
    prod: bool = False,
) -> BankStimulusProvider:
    """A spec §4.2 steps 1-7 for the SHIPPED variant. Raises ``BankError`` only.

    ``prod=True`` (caller passes ``SEVI_ENV == "prod"``; core reads no settings) refuses the
    synthetic test bank (voice ``TESTTONE``) and any experimental ta bank, whatever
    ``allowed_hashes`` says. en banks are experimental by design (contracts §4.13) and stay
    loadable; only the TESTTONE rule applies to them.
    """
    root = Path(bank_dir) / lang.value
    m = read_manifest(root / MANIFEST_NAME)  # (1)
    if m.lang is not lang:
        raise _integrity(f"manifest lang {m.lang.value} != {lang.value}")
    if m.variant is not BankVariant.SHIPPED:
        raise _integrity(f"manifest variant {m.variant.value} is not SHIPPED")

    for a in m.digits:  # (2)
        if _sha256_file(root / a.file) != a.sha256:
            raise _integrity(f"sha256 mismatch: {a.file}")
    if _sha256_file(root / m.noise_file) != m.noise_sha256:
        raise _integrity(f"sha256 mismatch: {m.noise_file}")

    if compute_bank_hash(m) != m.bank_hash:  # (3)
        raise _integrity("bank_hash mismatch")

    digits = {a.digit: _read_wav(root / a.file, a.n_samples) for a in m.digits}  # (4)
    for a in m.digits:
        if a.n_samples % 2 != 0:
            raise _integrity(f"{a.file}: odd sample count")
    noise = _read_wav(root / m.noise_file, NOISE_SAMPLES)

    if not allow_unfrozen and (  # (5)
        not m.frozen or m.bank_hash != allowed_hashes.get(lang)
    ):
        raise BankError(
            ErrorCode.BANK_HASH_REJECTED,
            "bank not frozen" if not m.frozen else "bank_hash not allowed",
            details={"lang": lang.value, "bank_hash": m.bank_hash},
        )

    if prod and (m.timbre.voice == TEST_VOICE or (m.experimental and lang is Lang.TA)):
        raise BankError(
            ErrorCode.BANK_HASH_REJECTED,
            "experimental/test bank refused in prod",
            details={"lang": lang.value, "bank_hash": m.bank_hash},
        )

    if m.design != triplets.design(m.digit_set):  # (6)
        raise _integrity("design does not match triplets.design(digit_set)")
    if m.digit_set != DIGIT_SETS[lang]:
        raise _integrity(f"digit_set {m.digit_set} != {DIGIT_SETS[lang]}")
    free = set(CHAIN_FREE_FIELDS)
    if m.chain.model_dump(exclude=free) != ChainSpec().model_dump(exclude=free):
        raise _integrity("chain differs from ChainSpec() defaults")
    if m.chain.prisma_upsampler != PRISMA_UPSAMPLER.get(m.chain.prisma_sample_rate_hz):
        raise _integrity("prisma_upsampler inconsistent with prisma_sample_rate_hz")

    try:  # (7)
        return BankStimulusProvider(m, digits, noise)
    except ValueError as e:  # pragma: no cover - guarded by the checks above
        raise _integrity(str(e)) from e


__all__ = ["canonical_json", "compute_bank_hash", "load_bank", "read_manifest"]
