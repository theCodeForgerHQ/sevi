"""U-CAL-01, U-CAL-03, U-CAL-04, U-BANK-06 (RMS subset) and CLI credit-safety tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from sevi.calibration import freeze as fz
from sevi.calibration import noise, render
from sevi.calibration.render import (
    AUDIO_CONFIG,
    RenderCache,
    RenderError,
    cache_key,
    make_plan,
    trim_bounds,
)
from sevi.contracts.enums import Lang, OffsetSource
from sevi.core import dsp
from sevi.fakes.scenarios.timbre_render import FakeTimbre, fake_wav48


def _tone16(n: int, f: float = 500.0) -> np.ndarray:
    return 0.3 * np.sin(2 * np.pi * f * np.arange(n) / 16000)


def test_U_CAL_01_trim_on_silence_tone_silence() -> None:
    x = np.concatenate([np.zeros(1600), _tone16(4800), np.zeros(1600)])
    s0, s1, onset_ms = trim_bounds(x, plosive_onset=False)
    assert abs(s0 - (1600 - 80)) <= 80
    assert abs(s1 - (6400 + 480)) <= 80
    assert (s1 - s0) % 2 == 0
    assert onset_ms == 5.0


def test_U_CAL_01_plosive_margin_is_15ms() -> None:
    x = np.concatenate([np.zeros(1600), _tone16(4800), np.zeros(1600)])
    s0, _, onset_ms = trim_bounds(x, plosive_onset=True)
    assert onset_ms == 15.0
    assert abs(s0 - (1600 - 240)) <= 80


def test_U_CAL_01_trim_clips_at_edges() -> None:
    s0, s1, onset_ms = trim_bounds(_tone16(3201), plosive_onset=True)
    assert s0 == 0
    assert onset_ms == 0.0
    assert s1 <= 3201
    assert (s1 - s0) % 2 == 0


def test_U_CAL_04_cache_key_identity_and_sensitivity() -> None:
    base = {"text": "ஒன்று", "voice": "Chetana", "model": "timbre-v3.0", "language": "ta-IN"}
    k = cache_key(**base)
    assert k == cache_key(**base)
    for field, val in [
        ("text", "இரண்டு"),
        ("voice", "Yashraj"),
        ("model", "timbre-v2.5"),
        ("language", "en-IN"),
    ]:
        assert cache_key(**{**base, field: val}) != k
    assert cache_key(**base, speed=1.1) != k
    assert cache_key(**base, audio_config={**AUDIO_CONFIG, "sample_rate": 16000}) != k


def test_U_CAL_04_cache_hit_performs_no_call(tmp_path: Path) -> None:
    cache = RenderCache(tmp_path)
    fake = FakeTimbre()
    plan = make_plan(Lang.TA, cache)
    assert asyncio.run(render.fetch_missing(plan, fake, cache)) == 8
    assert len(fake.calls) == 8
    again = make_plan(Lang.TA, cache)
    assert not again.misses
    assert again.chars_to_bill == 0
    assert asyncio.run(render.fetch_missing(again, fake, cache)) == 0
    assert asyncio.run(render.fetch_missing(plan, fake, cache)) == 0  # stale plan: re-checked
    assert len(fake.calls) == 8


def test_cache_is_write_once(tmp_path: Path) -> None:
    cache = RenderCache(tmp_path)
    cache.put("k", b"first")
    cache.put("k", b"second")
    assert cache.get("k") == b"first"
    with pytest.raises(RenderError):
        cache.put("empty", b"")


def test_plan_character_counts_match_spec(tmp_path: Path) -> None:
    """A spec §3.2 / contracts §4.13: ta 8 words = 39 chars, en 8 words = 31 chars."""
    cache = RenderCache(tmp_path)
    assert make_plan(Lang.TA, cache).chars_to_bill == 39
    assert make_plan(Lang.EN, cache).chars_to_bill == 31
    assert make_plan(Lang.TA, cache).credits_estimate(0.25) == pytest.approx(9.75)


def test_plan_alt_words(tmp_path: Path) -> None:
    plan = make_plan(Lang.TA, RenderCache(tmp_path), alt_digits=(2,))
    assert plan.items[1].word == "ரெண்டு"
    with pytest.raises(RenderError):
        make_plan(Lang.EN, RenderCache(tmp_path), alt_digits=(2,))


def test_request_uses_synthesize_wav_contract(tmp_path: Path) -> None:
    """contracts §8.4: one request per digit, text = the word alone, 48 kHz."""
    cache, fake = RenderCache(tmp_path), FakeTimbre()
    asyncio.run(render.fetch_missing(make_plan(Lang.EN, cache), fake, cache))
    assert [c[0] for c in fake.calls] == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "eight",
        "nine",
    ]
    assert all(c[1:] == (Lang.EN, "Chetana", "timbre-v3.0", 48000) for c in fake.calls)


def test_process_digit_equalises_to_minus_23() -> None:
    p = render.process_digit(fake_wav48("x"), plosive_onset=False)
    assert dsp.dbfs(p.x) == pytest.approx(-23.0, abs=1e-9)
    assert p.x.size % 2 == 0
    assert render.sanity_failures(p) == []


def test_process_digit_rejects_wrong_rate_and_stereo() -> None:
    for kw in ({"fs": 44100}, {"channels": 2}):
        with pytest.raises(RenderError) as e:
            render.process_digit(fake_wav48("x", **kw), plosive_onset=False)  # type: ignore[arg-type]
        assert e.value.exit_code == 2


def test_sanity_flags_quiet_render() -> None:
    p = render.process_digit(fake_wav48("x", amp=0.001), plosive_onset=False)
    assert any("active RMS" in f for f in render.sanity_failures(p))


def _fill(tmp_path: Path, lang: Lang) -> tuple[RenderCache, Path]:
    cache = RenderCache(tmp_path / "cache")
    asyncio.run(render.fetch_missing(make_plan(lang, cache), FakeTimbre(), cache))
    return cache, tmp_path / "bank" / lang.value


def test_U_CAL_03_regenerate_deterministic_and_offset_local(tmp_path: Path) -> None:
    cache, out = _fill(tmp_path, Lang.TA)
    render.build_digits(make_plan(Lang.TA, cache), cache, out)
    _, sha1, seed = noise.regenerate(out, Lang.TA, {})
    b1 = (out / "noise.wav").read_bytes()
    _, sha2, _ = noise.regenerate(out, Lang.TA, {})
    assert sha1 == sha2
    assert (out / "noise.wav").read_bytes() == b1
    assert seed == noise.noise_seed(Lang.TA)
    _, sha3, _ = noise.regenerate(out, Lang.TA, {1: 3.0})
    assert sha3 != sha1


# ---------- CLI credit safety ----------
def _cli(tmp_path: Path, *extra: str, client: FakeTimbre | None = None) -> int:
    return render.main(
        ["--lang", "ta", "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "o"), *extra],
        client=client,
    )


def test_cli_requires_yes_and_prints_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeTimbre()
    assert _cli(tmp_path, client=fake) == 1
    assert fake.calls == []
    out = capsys.readouterr()
    plan = json.loads(out.out.splitlines()[0])
    assert plan["chars"] == 39
    assert plan["credits_estimate"] == pytest.approx(9.75)
    assert "--yes" in out.err


def test_cli_offline_miss_exits_2(tmp_path: Path) -> None:
    fake = FakeTimbre()
    assert _cli(tmp_path, "--offline", "--yes", client=fake) == 2
    assert fake.calls == []


def test_cli_yes_renders_once_then_cache_only(tmp_path: Path) -> None:
    fake = FakeTimbre()
    assert _cli(tmp_path, "--yes", client=fake) == 0
    assert len(fake.calls) == 8
    assert _cli(tmp_path, "--yes", client=fake) == 0
    assert _cli(tmp_path, "--offline", client=fake) == 0
    assert len(fake.calls) == 8
    assert (tmp_path / "o" / "render.json").is_file()


def test_cli_missing_adapter_exits_2(tmp_path: Path) -> None:
    assert _cli(tmp_path, "--yes", "--client", "sevi.no_such_module:factory") == 2


def test_cli_ineligible_voice_exits_4(tmp_path: Path) -> None:
    assert _cli(tmp_path, "--yes", client=FakeTimbre(amp=0.001)) == 4


# ---------- freeze ----------
def test_U_BANK_06_rms_freeze_ta(tmp_path: Path) -> None:
    cache, out = _fill(tmp_path, Lang.TA)
    render.build_digits(make_plan(Lang.TA, cache), cache, out)
    m = fz.freeze(out, Lang.TA)
    assert m.frozen
    assert m.frozen_at is not None
    assert m.calibration.source is OffsetSource.RMS
    assert m.calibration.iterations == 0
    assert m.calibration.panel_outcome_sha256 is None
    assert all(a.offset_db == 0.0 for a in m.digits)
    assert m.experimental is False
    assert m.bank_version == "ta-1.0.0"
    assert (out / "manifest.json").is_file()


def test_U_BANK_06_rms_freeze_en(tmp_path: Path) -> None:
    cache, out = _fill(tmp_path, Lang.EN)
    render.build_digits(make_plan(Lang.EN, cache), cache, out)
    m = fz.freeze(out, Lang.EN)
    assert m.digit_set == (1, 2, 3, 4, 5, 6, 8, 9)
    assert m.experimental is True
    assert m.calibration.panel_outcome_sha256 is None
    assert m.timbre.language == "en-IN"


def test_freeze_refuses_edited_digit(tmp_path: Path) -> None:
    cache, out = _fill(tmp_path, Lang.TA)
    render.build_digits(make_plan(Lang.TA, cache), cache, out)
    p = out / "digits" / "3.wav"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    with pytest.raises(fz.FreezeError):
        fz.freeze(out, Lang.TA)
    assert fz.main(["--lang", "ta", "--dir", str(out)]) == 2
