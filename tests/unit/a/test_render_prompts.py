"""scripts/render_prompts.py (contracts §8.4, B spec §11.4; U-TMB-05, U-TMB-06). Fakes only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sevi.app.prompts import load_prompts
from sevi.calibration import render
from sevi.contracts.enums import Lang, PromptKey
from sevi.core.bank import read_manifest
from sevi.fakes.scenarios.timbre_render import FakeTimbre

REPO = Path(__file__).resolve().parents[3]
TEST_BANK = REPO / "stimuli" / "test-bank" / "ta"


def _mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_prompts", REPO / "scripts" / "render_prompts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_prompts"] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _mod()


def _args(tmp: Path, *extra: str, lang: str = "ta", bank: Path = TEST_BANK) -> list[str]:
    return [
        "--lang",
        lang,
        "--bank",
        str(bank),
        "--out",
        str(tmp / "prompts"),
        "--cache-dir",
        str(tmp / "cache"),
        "--selection",
        str(tmp / "no-selection.json"),
        *extra,
    ]


def _plan(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    first = capsys.readouterr().out.splitlines()[0]
    out: dict[str, object] = json.loads(first)
    return out


# ---------- split (U-TMB-05) ----------
def test_U_TMB_05_split_620_char_tamil_into_3_parts() -> None:
    s120 = "அ" * 119 + "."
    text = " ".join([s120] * 4 + ["ஆ" * 135 + "."])
    assert len(text) == 620
    parts = rp.split_text(text, 300)
    assert len(parts) == 3
    assert all(len(p) <= 300 for p in parts)
    assert all(p.endswith(".") for p in parts)
    assert " ".join(parts) == text


def test_U_TMB_05_long_sentence_cut_at_last_space() -> None:
    text = "word " * 100  # 500 chars, no sentence end
    parts = rp.split_text(text, 300)
    assert [len(p) <= 300 for p in parts] == [True, True]
    assert " ".join(parts) == text.strip()


def test_U_TMB_05_short_text_is_one_part() -> None:
    assert rp.split_text("  Correct. ", 300) == ["Correct."]


def test_U_TMB_05_parts_joined_with_150ms_gap(tmp_path: Path) -> None:
    fake = FakeTimbre()
    long_text = "One two three. " * 30
    texts = rp.selected_texts(Lang.EN, rp.DEFAULT_COPY_PATH, None)
    texts[PromptKey.INSTRUCTIONS] = ("prompt.instructions", "default", long_text.strip())
    cache = render.RenderCache(tmp_path / "cache")
    cfg = render.timbre_config(Lang.EN, "timbre-v3.0", "Chetana")
    plan = rp.make_plan(Lang.EN, cfg, texts, cache, 300)
    kp = next(k for k in plan.keys if k.key is PromptKey.INSTRUCTIONS)
    assert len(kp.parts) == 2
    import asyncio

    asyncio.run(render.fetch_texts(plan.jobs(), Lang.EN, cfg, fake, cache))
    one = sum(rp.decode16(cache.get(p.key)).size for p in kp.parts)
    assert rp.GAP_SAMPLES_16K == 2400
    assert one + rp.GAP_SAMPLES_16K == 2 * 12000 + 2400  # two 0.75 s fake renders + gap


# ---------- plan / dry run ----------
def test_plan_covers_every_non_digit_prompt_key_both_langs(tmp_path: Path) -> None:
    for lang in Lang:
        texts = rp.selected_texts(lang, rp.DEFAULT_COPY_PATH, None)
        keys = {k for k in PromptKey if not k.value.startswith("DIGIT_")}
        assert set(texts) == keys
        for ck, vid, text in texts.values():
            assert text.strip(), ck
            assert "{" not in text, f"{ck}: slot in a pre-rendered clip"
            assert vid


def test_U_TMB_06_dry_run_prints_bodies_and_calls_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEVI_GNANI__KEY_A", "SENTINEL-SECRET")
    fake = FakeTimbre()
    assert rp.main(_args(tmp_path, "--dry-run"), client=fake) == 0
    plan = _plan(capsys)
    assert fake.calls == []
    reqs = plan["requests"]
    assert isinstance(reqs, list)
    assert reqs
    for body in reqs:
        assert set(body) == {"text", "voice", "model", "language", "speed", "audio_config"}
        assert body["audio_config"] == dict(render.AUDIO_CONFIG)
        assert body["language"] == "ta-IN"
        assert 0 < len(body["text"]) <= 300
    chars = sum(len(b["text"]) for b in reqs)
    assert plan["chars"] == chars
    assert plan["credits_estimate"] == pytest.approx(chars * 0.25)
    dumped = json.dumps(plan)
    assert "SENTINEL-SECRET" not in dumped
    assert "x-api-key" not in dumped.lower()
    assert not (tmp_path / "cache").exists()


def test_no_yes_means_no_call_and_exit_1(tmp_path: Path) -> None:
    fake = FakeTimbre()
    assert rp.main(_args(tmp_path), client=fake) == 1
    assert fake.calls == []


def test_U_TMB_06_rerun_hits_cache_with_zero_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeTimbre()
    assert rp.main(_args(tmp_path, "--yes"), client=fake) == 0
    n = len(fake.calls)
    assert n > 0
    assert len({c[0] for c in fake.calls}) == n  # each distinct text sent once
    capsys.readouterr()
    assert rp.main(_args(tmp_path, "--yes"), client=fake) == 0
    assert len(fake.calls) == n
    plan = _plan(capsys)
    assert plan["chars"] == 0
    assert plan["requests"] == []
    assert rp.main(_args(tmp_path, "--offline"), client=fake) == 0


def test_build_loads_through_app_prompt_loader(tmp_path: Path) -> None:
    fake = FakeTimbre()
    assert rp.main(_args(tmp_path, "--yes"), client=fake) == 0
    bank = read_manifest(TEST_BANK / "manifest.json")
    ps = load_prompts(Lang.TA, tmp_path / "prompts", bank)
    assert len(ps) == len([k for k in PromptKey if not k.value.startswith("DIGIT_")]) + 8
    assert not ps.placeholder
    clip = ps.get(PromptKey.DIGIT_3)
    assert clip.duration_ms == pytest.approx(len(clip.ulaw) / 8.0)
    m = json.loads((tmp_path / "prompts" / "ta" / "manifest.json").read_text("utf-8"))
    assert m["timbre_model"] == "timbre-v3.0"
    assert m["voice"] == "Chetana"


def test_only_and_limit_credits_batch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTimbre()
    assert rp.main(_args(tmp_path, "--only", "GOODBYE,PRACTICE_CORRECT", "--yes"), client=fake) == 0
    assert sorted(c[0] for c in fake.calls) == sorted(["சரி.", "சோதனை செய்ததற்கு நன்றி. வணக்கம்."])
    capsys.readouterr()
    assert rp.main(_args(tmp_path, "--limit-credits", "20", "--yes"), client=fake) == 0
    plan = _plan(capsys)
    assert plan["credits_estimate"] <= 20
    assert plan["deferred_requests"] > 0
    new = fake.calls[2:]
    assert sum(len(c[0]) for c in new) * 0.25 <= 20


def test_limit_jobs_never_exceeds_cap() -> None:
    jobs = [("a", "x" * 40), ("b", "y" * 40), ("c", "z" * 4)]
    take, rest = rp.limit_jobs(jobs, 20.0, 0.25)
    assert take == jobs[:2]
    assert rest == jobs[2:]
    take, rest = rp.limit_jobs(jobs, 5.0, 0.25)
    assert take == []
    assert rest == jobs


def test_build_without_bank_exits_2(tmp_path: Path) -> None:
    fake = FakeTimbre()
    code = rp.main(_args(tmp_path, "--yes", bank=tmp_path / "nobank"), client=fake)
    assert code == 2


def test_bank_voice_wins_over_flags() -> None:
    m = read_manifest(TEST_BANK / "manifest.json")
    real = m.model_copy(update={"timbre": m.timbre.model_copy(update={"voice": "Vedika"})})
    assert rp.timbre_config(Lang.TA, real, "timbre-v3.0", "Chetana").voice == "Vedika"
    assert rp.timbre_config(Lang.TA, m, "timbre-v3.0", "Chetana").voice == "Chetana"


def test_prompt_clip_levels_follow_prompt_chain(tmp_path: Path) -> None:
    from sevi.core import dsp

    fake = FakeTimbre()
    assert rp.main(_args(tmp_path, "--yes"), client=fake) == 0
    data = (tmp_path / "prompts" / "ta" / "GOODBYE.ulaw").read_bytes()
    x = dsp.pcm16_to_float(dsp.ulaw_decode(data))
    assert -21.0 < dsp.dbfs(x) < -19.0
    assert 20 * np.log10(np.max(np.abs(x))) <= -2.9
