"""End-to-end: fake Timbre -> render -> freeze -> core.bank.load_bank (A spec §3-§4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sevi.calibration import freeze, render
from sevi.contracts.enums import Lang
from sevi.contracts.errors import BankError, ErrorCode
from sevi.core.bank import load_bank
from sevi.fakes.scenarios.timbre_render import FakeTimbre


@pytest.mark.parametrize("lang", [Lang.TA, Lang.EN])
def test_E2E_CAL_render_freeze_load(
    tmp_path: Path, lang: Lang, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeTimbre()
    bank_root = tmp_path / "stimuli"
    lang_dir = bank_root / lang.value
    args = ["--lang", lang.value, "--cache-dir", str(tmp_path / "cache"), "--out", str(lang_dir)]
    assert render.main(args, client=fake) == 1  # no --yes: nothing billed
    assert fake.calls == []
    assert render.main([*args, "--yes"], client=fake) == 0
    assert len(fake.calls) == 8
    assert freeze.main(["--lang", lang.value, "--dir", str(lang_dir)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    bank_hash = json.loads(lines[-1])["SEVI_RESULT__FROZEN_BANK_HASHES"][lang.value]

    provider = load_bank(lang, str(bank_root), {lang: bank_hash}, allow_unfrozen=False)
    assert provider.manifest.bank_hash == bank_hash
    assert provider.manifest.calibration.source.value == "RMS"

    with pytest.raises(BankError) as e:
        load_bank(lang, str(bank_root), {}, allow_unfrozen=False)
    assert e.value.code is ErrorCode.BANK_HASH_REJECTED

    # a re-render from the cache is free and reproduces the same digits and hash
    assert render.main([*args, "--offline"], client=fake) == 0
    assert len(fake.calls) == 8
