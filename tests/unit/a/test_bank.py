"""Unit tests for core/bank.py (A spec §4.2, U-BANK-01..03, U-BANK-05; U-CAL-02).

U-BANK-04 and U-BANK-06 are cut with the human panel and the freeze tool (BUILD-BRIEF §4).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sevi.calibration.testbank import build_bank
from sevi.contracts.enums import Lang
from sevi.contracts.errors import BankError, ErrorCode
from sevi.contracts.ports import StimulusProvider
from sevi.contracts.stimulus import BankManifest
from sevi.core.bank import canonical_json, compute_bank_hash, load_bank


@pytest.fixture
def bank(tmp_path: Path, test_bank_dir: Path) -> Path:
    """A writable copy of the test bank; returns the bank_dir (``<dir>/ta/...``)."""
    shutil.copytree(test_bank_dir, tmp_path / "bank")
    return tmp_path / "bank"


def _manifest_path(bank: Path) -> Path:
    return bank / "ta" / "manifest.json"


def _read(bank: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_manifest_path(bank).read_text(encoding="utf-8"))
    return data


def _write(bank: Path, data: dict[str, Any], *, rehash: bool) -> str:
    if rehash:
        m = BankManifest.model_validate(data)
        data["bank_hash"] = compute_bank_hash(m)
    _manifest_path(bank).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(data["bank_hash"])


def _load(bank: Path, h: str | None, allow_unfrozen: bool = False) -> StimulusProvider:
    allowed = {Lang.TA: h} if h is not None else {}
    return load_bank(Lang.TA, str(bank), allowed, allow_unfrozen)


def _code(exc: pytest.ExceptionInfo[BankError]) -> ErrorCode:
    return exc.value.code


def test_load_test_bank_ok(test_bank: StimulusProvider, test_bank_hash: str) -> None:
    assert isinstance(test_bank, StimulusProvider)
    m = test_bank.manifest
    assert m.bank_hash == test_bank_hash
    assert m.experimental is True
    assert m.timbre.voice == "TESTTONE"
    assert m.frozen is True


def test_U_BANK_01_unfrozen_rejected(bank: Path) -> None:
    data = _read(bank)
    data["frozen"] = False
    data["frozen_at"] = None
    h = _write(bank, data, rehash=False)  # frozen fields are not part of the hash
    with pytest.raises(BankError) as e:
        _load(bank, h)
    assert _code(e) is ErrorCode.BANK_HASH_REJECTED
    assert _load(bank, None, allow_unfrozen=True).manifest.frozen is False


def test_U_BANK_01_hash_not_allowed_rejected(bank: Path, test_bank_hash: str) -> None:
    for allowed in (None, "f" * 64):
        with pytest.raises(BankError) as e:
            _load(bank, allowed)
        assert _code(e) is ErrorCode.BANK_HASH_REJECTED
    assert _load(bank, test_bank_hash).manifest.bank_hash == test_bank_hash


def test_U_BANK_01_flipped_digit_byte_is_integrity(bank: Path, test_bank_hash: str) -> None:
    p = bank / "ta" / "digits" / "3.wav"
    raw = bytearray(p.read_bytes())
    raw[200] ^= 0x01
    p.write_bytes(bytes(raw))
    for allow_unfrozen in (False, True):
        with pytest.raises(BankError) as e:
            _load(bank, test_bank_hash, allow_unfrozen)
        assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_U_BANK_01_flipped_noise_byte_is_integrity(bank: Path, test_bank_hash: str) -> None:
    p = bank / "ta" / "noise.wav"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0x01
    p.write_bytes(bytes(raw))
    with pytest.raises(BankError) as e:
        _load(bank, test_bank_hash)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_U_BANK_01_edited_manifest_without_rehash_is_integrity(
    bank: Path, test_bank_hash: str
) -> None:
    data = _read(bank)
    data["noise_seed"] = data["noise_seed"] + 1
    _write(bank, data, rehash=False)
    with pytest.raises(BankError) as e:
        _load(bank, test_bank_hash, allow_unfrozen=True)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_U_BANK_02_hash_independent_of_key_order(bank: Path, test_bank_hash: str) -> None:
    data = _read(bank)
    reordered = {k: data[k] for k in sorted(data, reverse=True)}
    reordered["timbre"] = {k: data["timbre"][k] for k in sorted(data["timbre"], reverse=True)}
    _manifest_path(bank).write_text(json.dumps(reordered), encoding="utf-8")
    assert _load(bank, test_bank_hash).manifest.bank_hash == test_bank_hash


def test_U_BANK_02_canonical_json_is_sorted_and_compact() -> None:
    assert canonical_json({"b": 1, "a": [1.5, "த"]}) == '{"a":[1.5,"த"],"b":1}'.encode()


def test_U_BANK_03_digits_out_of_order_is_integrity(bank: Path, test_bank_hash: str) -> None:
    data = _read(bank)
    data["digits"] = list(reversed(data["digits"]))
    _write(bank, data, rehash=False)
    with pytest.raises(BankError) as e:
        _load(bank, test_bank_hash)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def _replace_digit3(bank: Path, *, channels: int, rate: int) -> str:
    p = bank / "ta" / "digits" / "3.wav"
    n = 6400
    pcm = (np.sin(np.arange(n * channels) * 0.1) * 3000).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    data = _read(bank)
    data["digits"][2]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _write(bank, data, rehash=True)


@pytest.mark.parametrize(("channels", "rate"), [(2, 16000), (1, 48000)])
def test_U_BANK_03_stereo_or_48k_wav_is_integrity(bank: Path, channels: int, rate: int) -> None:
    h = _replace_digit3(bank, channels=channels, rate=rate)
    with pytest.raises(BankError) as e:
        _load(bank, h)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_n_samples_mismatch_is_integrity(bank: Path) -> None:
    data = _read(bank)
    data["digits"][0]["n_samples"] = 6402
    h = _write(bank, data, rehash=True)
    with pytest.raises(BankError) as e:
        _load(bank, h)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lang", "en"),
        ("variant", "BALANCED"),
        ("design", None),
    ],
)
def test_wrong_lang_variant_or_design_is_integrity(bank: Path, field: str, value: Any) -> None:
    data = _read(bank)
    if field == "lang":
        data["bank_version"] = "en-0.0.1"
    if field == "design":
        value = dict(data["design"], offset_pairs=list(reversed(data["design"]["offset_pairs"])))
    data[field] = value
    h = _write(bank, data, rehash=True)
    with pytest.raises(BankError) as e:
        _load(bank, h)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_prisma_16k_chain_accepted(bank: Path) -> None:
    data = _read(bank)
    data["chain"]["prisma_sample_rate_hz"] = 16000
    data["chain"]["prisma_upsampler"] = "resample_poly(2,1)"
    h = _write(bank, data, rehash=True)
    assert _load(bank, h).manifest.chain.prisma_sample_rate_hz == 16000


def test_missing_files_are_integrity(bank: Path, test_bank_hash: str, tmp_path: Path) -> None:
    (bank / "ta" / "noise.wav").unlink()
    with pytest.raises(BankError) as e:
        _load(bank, test_bank_hash)
    assert _code(e) is ErrorCode.BANK_INTEGRITY
    with pytest.raises(BankError) as e:
        _load(tmp_path / "nowhere", test_bank_hash)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_invalid_json_is_integrity(bank: Path, test_bank_hash: str) -> None:
    _manifest_path(bank).write_text("{not json", encoding="utf-8")
    with pytest.raises(BankError) as e:
        _load(bank, test_bank_hash)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_en_bank_not_present_is_integrity(test_bank_dir: Path) -> None:
    with pytest.raises(BankError) as e:
        load_bank(Lang.EN, str(test_bank_dir), {}, True)
    assert _code(e) is ErrorCode.BANK_INTEGRITY


def test_U_CAL_02_test_bank_build_is_byte_identical(
    tmp_path: Path, test_bank_dir: Path, test_bank_hash: str
) -> None:
    """U-CAL-02: regenerating the test bank reproduces the committed files and the pinned hash."""
    m = build_bank(tmp_path)
    assert m.bank_hash == test_bank_hash
    for rel in ["manifest.json", "noise.wav", *(f"digits/{d}.wav" for d in range(1, 9))]:
        assert (tmp_path / "ta" / rel).read_bytes() == (test_bank_dir / "ta" / rel).read_bytes()


def test_test_bank_files_are_small(test_bank_dir: Path) -> None:
    total = sum(p.stat().st_size for p in test_bank_dir.rglob("*") if p.is_file())
    assert total < 2_200_000  # 60 s noise (1.92 MB, required by §4.2) + 8 x 12.8 kB digits


@pytest.mark.slow
def test_U_BANK_05_load_wall_time(test_bank_dir: Path, test_bank_hash: str) -> None:
    t0 = time.perf_counter()
    load_bank(Lang.TA, str(test_bank_dir), {Lang.TA: test_bank_hash}, False)
    assert time.perf_counter() - t0 <= 2.0
