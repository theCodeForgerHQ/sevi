"""app/prompts.py: mark naming (contracts §6.4) and prompt asset loading (B spec §7.12)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sevi.app.prompts import (
    PLACEHOLDER_VOICE,
    compute_prompts_hash,
    copy_key_for,
    load_prompts,
    mark_for_feedback,
    mark_for_playlist,
    mark_for_prompt,
    mark_for_stimulus,
    placeholder_prompts,
    required_keys,
    wording_key,
)
from sevi.contracts.enums import Lang, PromptKey
from sevi.contracts.errors import BankError, ErrorCode
from sevi.contracts.stimulus import BankManifest, PromptAsset, PromptManifest

TA = (1, 2, 3, 4, 5, 6, 7, 8)


def test_U_ORC_marks_contract_6_4() -> None:
    assert mark_for_prompt(1) == "p0001"
    assert mark_for_stimulus(7, 1) == "s007r1"
    assert mark_for_stimulus(123, 0) == "s123r0"
    assert mark_for_feedback(4) == "f004"
    assert mark_for_playlist(0) == "z00"


def test_required_keys_drop_digits_outside_set() -> None:
    req = required_keys(TA)
    assert PromptKey.DIGIT_9 not in req
    assert PromptKey.DIGIT_1 in req
    assert PromptKey.DIGIT_8 in req
    assert PromptKey.WORDING_W_PASS in req
    assert len(req) == len(PromptKey) - 1


def test_wording_key_and_copy_keys() -> None:
    assert wording_key("W_PASS") is PromptKey.WORDING_W_PASS
    assert wording_key("W_NOREPLY") is None  # no clip (contracts §6.5)
    assert copy_key_for(PromptKey.WORDING_W_INC) == "wording.W_INC.spoken"
    assert copy_key_for(PromptKey.GOODBYE_EARLY) == "prompt.goodbye_early"


def test_placeholder_set_is_silent_and_complete() -> None:
    ps = placeholder_prompts(Lang.TA, TA, duration_ms=250)
    assert ps.placeholder
    assert ps.voice == PLACEHOLDER_VOICE
    assert len(ps) == len(required_keys(TA))
    clip = ps.get(PromptKey.INSTRUCTIONS)
    assert clip.duration_ms == 250
    assert len(clip.ulaw) == 2000
    assert set(clip.ulaw) == {0xFF}
    assert not ps.has(PromptKey.DIGIT_9)
    with pytest.raises(KeyError):
        ps.get(PromptKey.DIGIT_9)


def _write_set(root: Path, lang: Lang, keys: frozenset[PromptKey], voice: str = "kavya") -> Path:
    d = root / lang.value
    d.mkdir(parents=True)
    assets = []
    for k in sorted(keys, key=lambda k: k.value):
        data = bytes([0xFF]) * 800
        (d / f"{k.value}.ulaw").write_bytes(data)
        assets.append(
            PromptAsset(
                key=k,
                lang=lang,
                copy_key=copy_key_for(k),
                text_sha256="0" * 64,
                file=f"{k.value}.ulaw",
                sha256=hashlib.sha256(data).hexdigest(),
                duration_ms=100.0,
            )
        )
    m = PromptManifest(
        lang=lang,
        voice=voice,
        timbre_model="timbre-v3.0",
        copy_sha256="0" * 64,
        assets=tuple(assets),
        prompts_hash="0" * 64,
    )
    m = m.model_copy(update={"prompts_hash": compute_prompts_hash(m)})
    (d / "manifest.json").write_text(m.model_dump_json(), encoding="utf-8")
    return d


def test_load_prompts_round_trip(tmp_path: Path, test_bank_dir: Path) -> None:
    bank = BankManifest.model_validate_json((test_bank_dir / "ta" / "manifest.json").read_text())
    _write_set(tmp_path, Lang.TA, required_keys(bank.digit_set))
    ps = load_prompts(Lang.TA, tmp_path, bank)
    assert ps.get(PromptKey.GOODBYE).duration_ms == 100.0
    assert len(ps.prompts_hash) == 64
    assert not ps.placeholder


@pytest.mark.parametrize("fault", ["sha", "hash", "missing", "lang"])
def test_load_prompts_integrity(tmp_path: Path, test_bank_dir: Path, fault: str) -> None:
    bank = BankManifest.model_validate_json((test_bank_dir / "ta" / "manifest.json").read_text())
    keys = required_keys(bank.digit_set)
    if fault == "missing":
        keys = keys - {PromptKey.GOODBYE}
    d = _write_set(tmp_path, Lang.TA, keys)
    if fault == "sha":
        (d / "INSTRUCTIONS.ulaw").write_bytes(b"\x00" * 10)
    if fault == "hash":
        raw = json.loads((d / "manifest.json").read_text())
        raw["voice"] = "other"
        (d / "manifest.json").write_text(json.dumps(raw))
    lang = Lang.EN if fault == "lang" else Lang.TA
    if fault == "lang":
        (tmp_path / "en").mkdir()
        (tmp_path / "en" / "manifest.json").write_text((d / "manifest.json").read_text())
    with pytest.raises(BankError) as e:
        load_prompts(lang, tmp_path, bank)
    assert e.value.code is ErrorCode.BANK_INTEGRITY


def test_placeholder_refused_in_prod(tmp_path: Path, test_bank_dir: Path) -> None:
    bank = BankManifest.model_validate_json((test_bank_dir / "ta" / "manifest.json").read_text())
    _write_set(tmp_path, Lang.TA, required_keys(bank.digit_set), voice=PLACEHOLDER_VOICE)
    assert load_prompts(Lang.TA, tmp_path, bank).placeholder  # dev/test/demo: fine
    with pytest.raises(BankError) as e:
        load_prompts(Lang.TA, tmp_path, bank, prod=True)
    assert e.value.code is ErrorCode.BANK_HASH_REJECTED
