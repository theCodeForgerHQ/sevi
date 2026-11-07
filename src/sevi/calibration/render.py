"""Offline Timbre digit render, trim and RMS-equalisation (A spec §3.1-§3.3, P3).

Reduced scope (BUILD-BRIEF §4): RMS-equalised bank only, no Prisma grid.

Credit safety (BUILD-BRIEF §5):

* Every real request goes through a Timbre adapter (B's ledger-guarded ``adapters/timbre.py``);
  this module only ever calls ``client.synthesize_wav`` (contracts §5.2 ``TTSClient``), so the
  same code runs against a fake in tests.
* Raw renders live in a content-addressed, write-once cache
  ``<cache_dir>/timbre/<key>.wav`` with ``key = sha256(canonical_json({text, voice, model,
  language, speed, audio_config}))`` (A spec §3.1, contracts §8.8). A cached digit is never
  re-sent.
* The CLI prints the character count and the estimated credits of the cache misses and makes no
  call without ``--yes``.

CLI::

    uv run python -m sevi.calibration.render --lang ta            # plan only, no network
    uv run python -m sevi.calibration.render --lang ta --yes      # render misses (costs credits)
    uv run python -m sevi.calibration.render --lang ta --offline  # cache only; a miss -> exit 2
    uv run python -m sevi.calibration.render --lang ta --digits 1 --yes  # probe: one digit only

The plan line lists every request body that would be sent (no key, no headers) and the
character / credit estimate. The default client is ``strict_offline_client``: B's ledger-guarded
Key A adapter with retries disabled, so the first non-200 stops the run and nothing is re-sent.
A response that is not a 48 kHz mono RIFF WAV is moved to ``<cache_dir>/rejected/`` (never into
the cache) and the run stops with exit 2.

Exit codes: 0 ok (or ``--digits`` subset fetched); 1 confirmation required (misses and no
``--yes``); 2 cache miss in ``--offline``, bad WAV or no client; 3 stored RMS off target; 4 a digit
fails the sanity rule (the voice is ineligible, §3.2); 5 the API or the ledger refused (non-200,
budget reached); nothing further is sent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import io
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from sevi.calibration.testbank import DIGIT_RMS_DBFS, FADE_SAMPLES, write_wav16
from sevi.contracts.enums import Lang
from sevi.contracts.errors import SeviError
from sevi.contracts.languages import ALT_STIMULUS_WORDS, DIGIT_SETS, STIMULUS_WORDS
from sevi.contracts.stimulus import DigitAsset, TimbreRenderConfig
from sevi.core import dsp
from sevi.core.bank import canonical_json
from sevi.core.dsp import F64

RENDER_SAMPLE_RATE_HZ: Final = 48000
AUDIO_CONFIG: Final[Mapping[str, Any]] = {
    "sample_rate": RENDER_SAMPLE_RATE_HZ,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}  # contracts §8.4 synthesize_wav body
DEFAULT_MODEL: Final = "timbre-v3.0"
DEFAULT_VOICE: Final = "Chetana"  # A spec §3.2: v3 default, same voice for en (contracts §4.13)
DEFAULT_CREDITS_PER_CHAR: Final = 0.25  # planning q (A spec §3.2); the dashboard is the truth
DEFAULT_CLIENT: Final = "sevi.calibration.render:strict_offline_client"
RENDER_FILE: Final = "render.json"

# Trim (§3.3 step 4), 16 kHz
FRAME: Final = 160
HOP: Final = 80
TRIM_DB_BELOW_MAX: Final = 35.0
MARGIN_PLOSIVE: Final = 240  # 15 ms [A-D30]
MARGIN_DEFAULT: Final = 80  # 5 ms
RELEASE_TAIL: Final = 320  # 20 ms
PLOSIVE_ONSET: Final[Mapping[Lang, frozenset[int]]] = {
    Lang.TA: frozenset(),
    Lang.EN: frozenset({2, 3, 4, 5, 6}),
}
RMS_TOLERANCE_DB: Final = 0.05
MIN_DURATION_MS: Final = 200.0
MAX_DURATION_MS: Final = 1000.0
MIN_ACTIVE_RMS_DBFS: Final = -45.0
MAX_PEAK_DBFS: Final = -1.0


class RenderError(Exception):
    def __init__(self, exit_code: int, msg: str) -> None:
        super().__init__(msg)
        self.exit_code = exit_code


class WavSynthesizer(Protocol):
    """The ``synthesize_wav`` half of contracts §5.2 ``TTSClient`` (any fake or stub fits)."""

    async def synthesize_wav(
        self, text: str, lang: Lang, voice: str, model: str, sample_rate_hz: int = 48000
    ) -> bytes: ...


# ---------- cache ----------
def cache_key(
    text: str,
    voice: str,
    model: str,
    language: str,
    speed: float = 1.0,
    audio_config: Mapping[str, Any] = AUDIO_CONFIG,
) -> str:
    """U-CAL-04: sha256 of the canonical JSON of the six request fields."""
    body = {
        "text": text,
        "voice": voice,
        "model": model,
        "language": language,
        "speed": speed,
        "audio_config": dict(audio_config),
    }
    return hashlib.sha256(canonical_json(body)).hexdigest()


class RenderCache:
    """Write-once ``<root>/timbre/<key>.wav``. Entries are never edited or overwritten."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "timbre"

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.wav"

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def get(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def reject(self, key: str, data: bytes) -> Path:
        """Keep an unusable (but billed) response for inspection, outside the cache."""
        p = self.dir.parent / "rejected" / f"{key}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self.path(key)
        if p.exists():
            return  # write-once
        if not data:
            raise RenderError(2, "empty render body; not cached")
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)


# ---------- plan ----------
@dataclass(frozen=True)
class PlanItem:
    digit: int
    word: str
    key: str
    cached: bool


@dataclass(frozen=True)
class RenderPlan:
    config: TimbreRenderConfig
    lang: Lang
    items: tuple[PlanItem, ...]

    @property
    def misses(self) -> tuple[PlanItem, ...]:
        return tuple(i for i in self.items if not i.cached)

    @property
    def chars_to_bill(self) -> int:
        return sum(len(i.word) for i in self.misses)

    def credits_estimate(self, credits_per_char: float = DEFAULT_CREDITS_PER_CHAR) -> float:
        return self.chars_to_bill * credits_per_char


def timbre_config(lang: Lang, model: str, voice: str) -> TimbreRenderConfig:
    return TimbreRenderConfig.model_validate(
        {"model": model, "voice": voice, "language": f"{lang.value}-IN", "speed": 1.0}
    )


def make_plan(
    lang: Lang,
    cache: RenderCache,
    *,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
    alt_digits: Sequence[int] = (),
) -> RenderPlan:
    """§3.3 step 1: one request per digit, text = the §4.13 word (alt word only if asked)."""
    cfg = timbre_config(lang, model, voice)
    items = []
    for d in DIGIT_SETS[lang]:
        if d in alt_digits:
            if d not in ALT_STIMULUS_WORDS.get(lang, {}):
                raise RenderError(2, f"digit {d} has no alt word for {lang.value}")
            word = ALT_STIMULUS_WORDS[lang][d]
        else:
            word = STIMULUS_WORDS[lang][d]
        key = cache_key(word, voice, model, cfg.language, cfg.speed)
        items.append(PlanItem(d, word, key, cache.has(key)))
    return RenderPlan(cfg, lang, tuple(items))


def request_body(text: str, cfg: TimbreRenderConfig) -> dict[str, Any]:
    """The exact JSON body B's adapter sends for ``synthesize_wav`` (contracts §8.4); the
    ``X-API-Key-ID`` header is never part of it. Its canonical hash is the cache key."""
    return {
        "text": text,
        "voice": cfg.voice,
        "model": cfg.model,
        "language": cfg.language,
        "speed": cfg.speed,
        "audio_config": dict(AUDIO_CONFIG),
    }


def check_response(data: bytes) -> None:
    """A usable render is a decodable 48 kHz mono RIFF WAV with at least one frame."""
    if data[:4] != b"RIFF":
        raise RenderError(2, "response is not a RIFF WAV")
    x = decode_wav48(data)
    if x.size < FRAME * 3:
        raise RenderError(2, f"response too short ({x.size} samples)")


async def fetch_texts(
    jobs: Sequence[tuple[str, str]],
    lang: Lang,
    cfg: TimbreRenderConfig,
    client: WavSynthesizer,
    cache: RenderCache,
) -> int:
    """``jobs`` = ``(cache_key, text)``. One request per miss, sequentially; the first error
    (non-200, budget, bad body) propagates and stops the run. Returns the number of calls."""
    calls = 0
    for key, text in jobs:
        if cache.has(key):  # re-checked: never re-render a cached text
            continue
        data = await client.synthesize_wav(text, lang, cfg.voice, cfg.model, RENDER_SAMPLE_RATE_HZ)
        calls += 1
        try:
            check_response(data)
        except RenderError as e:
            where = cache.reject(key, data)
            raise RenderError(2, f"{e}; kept at {where}") from e
        cache.put(key, data)
    return calls


async def fetch_missing(
    plan: RenderPlan,
    client: WavSynthesizer,
    cache: RenderCache,
    only_digits: Sequence[int] | None = None,
) -> int:
    """Render only cache misses (optionally only ``only_digits``). Returns the number of calls."""
    jobs = [(i.key, i.word) for i in plan.items if only_digits is None or i.digit in only_digits]
    return await fetch_texts(jobs, plan.lang, plan.config, client, cache)


def strict_offline_client(purpose: str = "calibration") -> WavSynthesizer:
    """B's ``adapters.timbre.offline_client`` (Key A, ledger-guarded: ``BudgetExceeded`` before
    any I/O) with the adapter's 5xx/timeout retries switched off: one request per text, and
    the first failure stops the run (a retry could be billed twice)."""
    from sevi.adapters.timbre import offline_client  # lazy: settings/store only when billed

    client = offline_client(purpose=purpose)  # type: ignore[arg-type]
    build = client._build

    async def build_no_retry() -> Any:
        tc = await build()
        tc.retries = 0
        return tc

    client._build = build_no_retry
    return client


async def aclose_client(client: object) -> None:
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()


async def run_fetch(
    plan: RenderPlan,
    client: WavSynthesizer,
    cache: RenderCache,
    only_digits: Sequence[int] | None,
) -> int:
    try:
        return await fetch_missing(plan, client, cache, only_digits)
    finally:
        await aclose_client(client)


# ---------- per-digit processing (P3) ----------
def trim_bounds(x16: F64, plosive_onset: bool) -> tuple[int, int, float]:
    """§3.3 step 4 (U-CAL-01). Returns ``(s0, s1, onset_trim_ms)``; ``s1 - s0`` is even."""
    n = x16.size
    if n < FRAME:
        raise RenderError(4, "render shorter than one frame")
    n_frames = 1 + (n - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n_frames)[:, None]
    e = 10.0 * np.log10(np.mean(x16[idx] ** 2, axis=1) + 1e-20)
    active = np.flatnonzero(e >= e.max() - TRIM_DB_BELOW_MAX)
    k0, k1 = int(active[0]), int(active[-1])
    margin = MARGIN_PLOSIVE if plosive_onset else MARGIN_DEFAULT
    s0 = max(0, k0 * HOP - margin)
    s1 = min(n, k1 * HOP + FRAME + RELEASE_TAIL)
    if (s1 - s0) % 2:
        s1 -= 1
    return s0, s1, (k0 * HOP - s0) / 16.0


@dataclass(frozen=True)
class ProcessedDigit:
    x: F64  # trimmed, faded, equalised, 16 kHz
    onset_trim_ms: float
    active_rms_dbfs: float  # before equalisation
    peak_dbfs: float  # after equalisation
    crest_db: float


def decode_wav48(data: bytes) -> F64:
    """§3.3 step 2: soundfile decode; assert 48 kHz mono."""
    try:
        y, fs = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    except Exception as e:
        raise RenderError(2, f"undecodable WAV: {e}") from e
    if fs != RENDER_SAMPLE_RATE_HZ or y.shape[1] != 1:
        raise RenderError(2, f"expected 48000 Hz mono, got {fs} Hz x{y.shape[1]}")
    return np.ascontiguousarray(y[:, 0], dtype=np.float64)


def process_digit(data: bytes, plosive_onset: bool) -> ProcessedDigit:
    """§3.3 steps 2-6."""
    x = dsp.resample_48k_to_16k(decode_wav48(data))
    x = x - float(np.mean(x))
    s0, s1, onset_ms = trim_bounds(x, plosive_onset)
    t = x[s0:s1]
    active = dsp.dbfs(t) if np.any(t) else float("-inf")
    t = t * dsp.raised_cosine_envelope(t.size, FADE_SAMPLES)
    if not np.any(t):
        raise RenderError(4, "silent render")
    y = dsp.scale_to_rms_dbfs(t, DIGIT_RMS_DBFS)
    peak = float(np.max(np.abs(y)))
    peak_db = 20.0 * np.log10(peak)
    return ProcessedDigit(y, onset_ms, active, peak_db, peak_db - dsp.dbfs(y))


def sanity_failures(p: ProcessedDigit) -> list[str]:
    """§3.2 eligibility / §3.3 step 8."""
    out = []
    dur = p.x.size / 16.0
    if not MIN_DURATION_MS <= dur <= MAX_DURATION_MS:
        out.append(f"duration {dur:.1f} ms outside 200-1000")
    if p.active_rms_dbfs < MIN_ACTIVE_RMS_DBFS:
        out.append(f"active RMS {p.active_rms_dbfs:.1f} dBFS < -45")
    if p.peak_dbfs > MAX_PEAK_DBFS:
        out.append(f"peak {p.peak_dbfs:.2f} dBFS > -1.0")
    return out


@dataclass(frozen=True)
class RenderResult:
    config: TimbreRenderConfig
    lang: Lang
    assets: tuple[DigitAsset, ...]
    report: tuple[dict[str, Any], ...]

    @property
    def failures(self) -> list[str]:
        return [f"digit {r['digit']}: {f}" for r in self.report for f in r["failures"]]


def build_digits(plan: RenderPlan, cache: RenderCache, out_dir: Path) -> RenderResult:
    """§3.3 steps 2-8 from the cache only; writes ``digits/<d>.wav`` and ``render.json``."""
    assets: list[DigitAsset] = []
    report: list[dict[str, Any]] = []
    for item in plan.items:
        if not cache.has(item.key):
            raise RenderError(2, f"cache miss for digit {item.digit} ({item.key[:12]})")
        p = process_digit(cache.get(item.key), item.digit in PLOSIVE_ONSET[plan.lang])
        data = write_wav16(out_dir / "digits" / f"{item.digit}.wav", p.x)
        stored = dsp.pcm16_to_float(dsp.float_to_pcm16(p.x))
        rms_db = dsp.dbfs(stored)
        if abs(rms_db - DIGIT_RMS_DBFS) > RMS_TOLERANCE_DB:
            raise RenderError(3, f"digit {item.digit}: stored RMS {rms_db:.3f} dBFS off target")
        assets.append(
            DigitAsset(
                digit=item.digit,
                word=item.word,
                file=f"digits/{item.digit}.wav",
                sha256=hashlib.sha256(data).hexdigest(),
                n_samples=p.x.size,
                rms_dbfs=round(rms_db, 6),
                offset_db=0.0,
                onset_trim_ms=p.onset_trim_ms,
            )
        )
        report.append(
            {
                "digit": item.digit,
                "word": item.word,
                "cache_key": item.key,
                "duration_ms": p.x.size / 16.0,
                "active_rms_dbfs": round(p.active_rms_dbfs, 3),
                "peak_dbfs": round(p.peak_dbfs, 3),
                "crest_db": round(p.crest_db, 3),
                "failures": sanity_failures(p),
            }
        )
    res = RenderResult(plan.config, plan.lang, tuple(assets), tuple(report))
    write_render_record(out_dir, res)
    return res


def write_render_record(out_dir: Path, res: RenderResult) -> None:
    doc = {
        "lang": res.lang.value,
        "timbre": res.config.model_dump(mode="json"),
        "digits": [a.model_dump(mode="json") for a in res.assets],
        "report": list(res.report),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False)
    (out_dir / RENDER_FILE).write_text(text + "\n", encoding="utf-8")


def read_render_record(out_dir: Path) -> tuple[Lang, TimbreRenderConfig, tuple[DigitAsset, ...]]:
    doc = json.loads((out_dir / RENDER_FILE).read_text(encoding="utf-8"))
    return (
        Lang(doc["lang"]),
        TimbreRenderConfig.model_validate(doc["timbre"]),
        tuple(DigitAsset.model_validate(a) for a in doc["digits"]),
    )


# ---------- CLI ----------
def load_client(spec: str) -> WavSynthesizer:
    """``module:factory``; the factory returns B's ledger-guarded Timbre adapter (Key A)."""
    mod_name, _, attr = spec.partition(":")
    try:
        factory = getattr(importlib.import_module(mod_name), attr)
    except (ImportError, AttributeError) as e:
        raise RenderError(2, f"Timbre client {spec!r} unavailable: {e}") from e
    client: WavSynthesizer = factory()
    return client


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sevi.calibration.render", description=__doc__)
    ap.add_argument("--lang", required=True, choices=[lg.value for lg in Lang])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=["timbre-v3.0", "timbre-v2.5"])
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--alt", default="", help="comma-separated digits to render with alt words")
    ap.add_argument("--out", default=None, help="bank language dir (default stimuli/<lang>)")
    ap.add_argument(
        "--cache-dir", default=os.environ.get("SEVI_CAL__CACHE_DIR", "var/cache"), type=Path
    )
    ap.add_argument("--credits-per-char", type=float, default=DEFAULT_CREDITS_PER_CHAR)
    ap.add_argument("--client", default=DEFAULT_CLIENT, help="module:factory of the adapter")
    ap.add_argument("--offline", action="store_true", help="cache only; a miss exits 2")
    ap.add_argument("--yes", action="store_true", help="confirm the real (billed) render")
    ap.add_argument(
        "--digits",
        default="",
        help="comma-separated subset to fetch (probe/batching); the bank is built only when "
        "every digit is cached",
    )
    return ap


def main(argv: list[str] | None = None, client: WavSynthesizer | None = None) -> int:
    a = _parser().parse_args(argv)
    lang = Lang(a.lang)
    cache = RenderCache(a.cache_dir)
    out = Path(a.out) if a.out else Path("stimuli") / lang.value
    try:
        alt = tuple(int(s) for s in a.alt.split(",") if s.strip())
        only = tuple(int(s) for s in a.digits.split(",") if s.strip()) or None
        plan = make_plan(lang, cache, model=a.model, voice=a.voice, alt_digits=alt)
        if only is not None:
            unknown = set(only) - {i.digit for i in plan.items}
            if unknown:
                raise RenderError(2, f"digits {sorted(unknown)} not in the {lang.value} set")
        todo = [i for i in plan.misses if only is None or i.digit in only]
        chars = sum(len(i.word) for i in todo)
        est = {
            "step": "plan",
            "lang": lang.value,
            "model": a.model,
            "voice": a.voice,
            "digits": len(plan.items),
            "cached": len(plan.items) - len(plan.misses),
            "to_render": [i.digit for i in todo],
            "chars": chars,
            "credits_per_char": a.credits_per_char,
            "credits_estimate": chars * a.credits_per_char,
            "requests": [request_body(i.word, plan.config) for i in todo],
        }
        print(json.dumps(est, ensure_ascii=False))
        if todo:
            if a.offline:
                raise RenderError(2, "cache miss in --offline mode")
            if not a.yes:
                print(
                    f"{chars} chars (~{est['credits_estimate']:.2f} credits) would be "
                    "billed. Re-run with --yes to render.",
                    file=sys.stderr,
                )
                return 1
            calls = asyncio.run(run_fetch(plan, client or load_client(a.client), cache, only))
            print(json.dumps({"step": "rendered", "calls": calls}))
            plan = make_plan(lang, cache, model=a.model, voice=a.voice, alt_digits=alt)
        if only is not None and plan.misses:
            left = [i.digit for i in plan.misses]
            print(json.dumps({"step": "partial", "still_missing": left}))
            return 0
        res = build_digits(plan, cache, out)
        print(json.dumps({"step": "digits", "out": str(out), "report": list(res.report)}))
        if res.failures:
            print("ineligible: " + "; ".join(res.failures), file=sys.stderr)
            return 4
        return 0
    except RenderError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except SeviError as e:  # TtsError (non-200), BudgetExceeded: stop, send nothing more
        print(f"error: {e.code.value}: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
