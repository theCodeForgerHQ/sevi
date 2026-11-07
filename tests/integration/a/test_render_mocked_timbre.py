"""Phase-1 proof for the live render (#60, #63), with NO network.

The real default client (``render.strict_offline_client`` = B's ``adapters.timbre.offline_client``
on Key A with the real SQLite ledger, retries off) talks to a respx-mocked Timbre that asserts
the exact documented request (docs.gnani.ai ``api/TTS/tts-inference``) and answers with a real
RIFF WAV. The output goes through the whole pipeline (trim, RMS, freeze, ``load_bank``; prompt
chain to mu-law, manifest, ``app.prompts.load_prompts``). Failure modes stop after one request.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import respx
import soundfile as sf  # type: ignore[import-untyped]

from sevi.app.prompts import load_prompts
from sevi.app.settings import Settings
from sevi.calibration import freeze, render
from sevi.contracts.enums import Lang, PromptKey
from sevi.core import dsp
from sevi.core.bank import load_bank
from sevi.fakes.scenarios.timbre_render import fake_wav48

REPO = Path(__file__).resolve().parents[3]
URL = "https://api.vachana.ai/api/v1/tts/inference"
KEY = "dummy-key-a-not-real"
DOC_AUDIO_CONFIG = {  # docs.gnani.ai TTS REST: allowed rates 8000..48000; wav = RIFF, audio/wav
    "sample_rate": 48000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}
VOICES_V3 = {  # docs.gnani.ai available-voices: every v3.0 voice speaks all 10 languages
    "Avinash", "Chetana", "Devansh", "Kalpana", "Lakshmi", "Prakash", "Riyaa", "Shivani",
    "Shubham", "Vasudha", "Tarika", "Yashraj", "Harshad",
}  # fmt: skip


def _rp() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "render_prompts_it", REPO / "scripts" / "render_prompts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_prompts_it"] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _rp()


def _env(mp: pytest.MonkeyPatch, data: Path, budget: float) -> None:
    mp.setenv("SEVI_GNANI__KEY_A", KEY)
    mp.setenv("SEVI_DATA_DIR", str(data))
    mp.setenv("SEVI_LEDGER__BUDGET_CREDITS_PER_KEY", str(budget))
    mp.setenv("SEVI_LEDGER__CREDITS_PER_TTS_CHAR", "0.25")


def riff(text: str, fs: int) -> bytes:
    """A real-format Timbre body: RIFF/WAVE PCM_16 mono at ``fs``."""
    data = fake_wav48(text, fs=fs)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    return data


class StrictTimbre:
    """respx side effect asserting the documented request; ``mode`` picks the answer."""

    def __init__(self, lang: Lang, mode: str = "ok") -> None:
        self.lang = lang
        self.mode = mode
        self.bodies: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == URL
        assert request.headers["x-api-key-id"] == KEY
        assert request.headers["content-type"] == "application/json"
        body = json.loads(request.content)
        assert set(body) == {"text", "voice", "model", "language", "speed", "audio_config"}
        assert body["model"] == "timbre-v3.0"
        assert body["voice"] in VOICES_V3
        assert body["language"] == f"{self.lang.value}-IN"
        assert body["speed"] == 1.0
        assert body["audio_config"] == DOC_AUDIO_CONFIG
        assert 0 < len(body["text"]) <= 300
        self.bodies.append(body)
        if self.mode == "500":
            return httpx.Response(500, json={"success": False, "message": "x", "status_code": 500})
        if self.mode == "400":
            return httpx.Response(400, json={"success": False, "message": "bad voice"})
        fs = 8000 if self.mode == "riff8k" else 48000
        return httpx.Response(
            200, content=riff(body["text"], fs), headers={"content-type": "audio/wav"}
        )


def _ledger_events(data: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(Settings().sqlite_path) as con:
        return [tuple(r) for r in con.execute("SELECT key_id, purpose FROM ledger_events")]


@pytest.mark.parametrize("lang", [Lang.TA, Lang.EN])
def test_mocked_timbre_digits_and_prompts_load_in_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lang: Lang
) -> None:
    _env(monkeypatch, tmp_path / "data", budget=2000.0)
    stim = tmp_path / "stimuli"
    cache = tmp_path / "cache"
    timbre = StrictTimbre(lang)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL).mock(side_effect=timbre)
        base = ["--lang", lang.value, "--cache-dir", str(cache), "--out", str(stim / lang.value)]
        # probe: one digit only, then the rest; the probe is reused, never re-sent
        assert render.main([*base, "--digits", "1", "--yes"]) == 0
        assert route.call_count == 1
        assert render.main([*base, "--yes"]) == 0
        assert route.call_count == 8
        assert len({b["text"] for b in timbre.bodies}) == 8
        assert freeze.main(["--lang", lang.value, "--dir", str(stim / lang.value)]) == 0
        bank_hash = json.loads((stim / lang.value / "manifest.json").read_text(encoding="utf-8"))[
            "bank_hash"
        ]
        provider = load_bank(lang, str(stim), {lang: bank_hash}, allow_unfrozen=False)
        assert provider.manifest.timbre.voice == "Chetana"

        args = [
            "--lang", lang.value, "--bank", str(stim / lang.value), "--out", str(stim / "prompts"),
            "--cache-dir", str(cache), "--selection", str(tmp_path / "none.json"),
        ]  # fmt: skip
        assert rp.main([*args, "--dry-run"]) == 0
        assert route.call_count == 8  # dry run sends nothing
        assert rp.main([*args, "--yes"]) == 0
        n = route.call_count
        assert rp.main([*args, "--yes"]) == 0
        assert route.call_count == n  # write-once cache: nothing rendered twice

    ps = load_prompts(lang, stim / "prompts", provider.manifest, prod=True)
    assert ps.voice == "Chetana"
    assert not ps.placeholder
    for k in PromptKey:
        if k.value.startswith("DIGIT_") and int(k.value[6:]) not in provider.manifest.digit_set:
            continue
        clip = ps.get(k)
        assert clip.duration_ms > 0
        x = dsp.pcm16_to_float(dsp.ulaw_decode(clip.ulaw))
        assert -21.0 < dsp.dbfs(x) < -19.0
    events = _ledger_events(tmp_path / "data")
    assert events.count(("A", "calibration")) == 8
    assert events.count(("A", "prompt_render")) == n - 8


def test_riff_8k_answer_is_rejected_after_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The format seen on the Key B probe (8 kHz RIFF) is decodable, but the bank needs 48 kHz:
    the run stops on the first such answer, keeps it in rejected/, caches nothing."""
    body = riff("ஒன்று", 8000)
    _, fs = sf.read(io.BytesIO(body))
    assert fs == 8000
    _env(monkeypatch, tmp_path / "data", budget=100.0)
    with respx.mock as mock:
        route = mock.post(URL).mock(side_effect=StrictTimbre(Lang.TA, "riff8k"))
        code = render.main(["--lang", "ta", "--cache-dir", str(tmp_path / "c"), "--yes"])
    assert code == 2
    assert route.call_count == 1
    assert "48000" in capsys.readouterr().err
    assert len(list((tmp_path / "c" / "rejected").iterdir())) == 1
    assert not (tmp_path / "c" / "timbre").exists()


@pytest.mark.parametrize("mode", ["500", "400"])
def test_non_200_stops_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _env(monkeypatch, tmp_path / "data", budget=100.0)
    with respx.mock as mock:
        route = mock.post(URL).mock(side_effect=StrictTimbre(Lang.TA, mode))
        code = render.main(["--lang", "ta", "--cache-dir", str(tmp_path / "c"), "--yes"])
        assert code == 5
        assert route.call_count == 1
        args = [
            "--lang", "ta", "--bank", str(REPO / "stimuli/test-bank/ta"),
            "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "p"), "--yes",
        ]  # fmt: skip
        assert rp.main(args) == 5
        assert route.call_count == 2
    assert not (tmp_path / "c" / "timbre").exists()


def test_budget_refusal_sends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, tmp_path / "data", budget=0.5)  # 2 chars: below one digit word
    with respx.mock as mock:
        route = mock.post(URL).mock(side_effect=StrictTimbre(Lang.TA))
        code = render.main(["--lang", "ta", "--cache-dir", str(tmp_path / "c"), "--yes"])
    assert code == 5
    assert route.call_count == 0
