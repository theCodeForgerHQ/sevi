"""Render every spoken prompt and wording clip with Timbre (contracts §8.4, §11.9; B spec §11.4).

For each ``PromptKey`` of one language:

* ``DIGIT_d`` comes from the bank's ``digits/d.wav`` (no Timbre call, no credits).
* every other key renders the selected copy variant (``app.prompts.copy_key_for``:
  ``prompt.<key>`` or ``wording.<CLASS>.spoken``; variants from ``results/copy_selection.json``
  when present, else the reduced-scope outcomes, VOICE = DEMO_ONLY). Texts over ``max_chars``
  are split at sentence ends (``.``, ``?``), a lone long sentence at the last space.
* each part is one ``synthesize_wav(part, lang, voice, model, 48000)`` (Key A, ledger-guarded,
  no retries) whose raw WAV lands in the same content-addressed, write-once cache as the digit
  render (``<cache_dir>/timbre/<sha256(request)>.wav``). A cached part is never sent again.
* build (free, cache only): 48 kHz -> 16 kHz, parts joined with 150 ms of zeros,
  ``core.dsp.prompt_chain_16k`` (band-pass, -20 dBFS, peak -3, 8 kHz, mu-law) ->
  ``<out>/<lang>/<KEY>.ulaw`` + ``manifest.json`` (``PromptManifest``) that
  ``app.prompts.load_prompts`` accepts.

Credit safety: without ``--yes`` nothing is sent; the plan (every request body, no secrets) and
the char / credit totals are printed. ``--only KEY,KEY`` and ``--limit-credits N`` select a
batch; the first error (non-200, budget refusal, a response that is not a 48 kHz mono WAV) stops
the run with nothing retried.

Usage::

    uv run python scripts/render_prompts.py --lang ta                       # dry run (default)
    uv run python scripts/render_prompts.py --lang ta --limit-credits 450 --yes
    uv run python scripts/render_prompts.py --lang ta --offline             # build only

Exit codes: 0 ok / dry run; 1 misses remain and no ``--yes``; 2 cache miss in ``--offline``, bad
WAV, missing bank or client; 5 API or ledger refusal (nothing further sent).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from sevi.app.prompts import MANIFEST_NAME, compute_prompts_hash, copy_key_for
from sevi.calibration import render
from sevi.contracts.enums import Lang, PromptKey
from sevi.contracts.errors import SeviError
from sevi.contracts.stimulus import BankManifest, PromptAsset, PromptManifest, TimbreRenderConfig
from sevi.core import dsp
from sevi.core.bank import read_manifest
from sevi.core.dsp import F64
from sevi.core.wording import DEFAULT_COPY_PATH, load_copy

DEFAULT_MAX_CHARS = 300  # B's adapter MAX_CHARS; ops/timbre_max_chars.txt overrides
MAX_CHARS_FILE = Path("ops/timbre_max_chars.txt")
GAP_SAMPLES_16K = 2400  # 150 ms at 16 kHz (U-TMB-05)
SENTENCE_RE = re.compile(r"(?<=[.?])\s+")
TEST_VOICE = "TESTTONE"


# ---------- text ----------
def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Greedy pack of sentences (split after ``.``/``?``) into parts of <= ``max_chars``; a
    single sentence longer than that is cut at the last space before the limit."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    cur = ""
    for sentence in SENTENCE_RE.split(text):
        s = sentence.strip()
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars + 1)
            if cut <= 0:
                cut = max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        if not s:
            continue
        cand = f"{cur} {s}" if cur else s
        if len(cand) <= max_chars:
            cur = cand
        else:
            parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    return parts


def read_max_chars(path: Path = MAX_CHARS_FILE) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return DEFAULT_MAX_CHARS


# ---------- plan ----------
@dataclass(frozen=True)
class Part:
    text: str
    key: str  # render cache key
    cached: bool


@dataclass(frozen=True)
class KeyPlan:
    key: PromptKey
    copy_key: str
    text: str  # full selected text (or the digit word)
    variant_id: str
    parts: tuple[Part, ...]  # empty for DIGIT_d (from the bank)


@dataclass
class PromptPlan:
    lang: Lang
    config: TimbreRenderConfig
    keys: list[KeyPlan] = field(default_factory=list)

    def jobs(self, only: frozenset[PromptKey] | None = None) -> list[tuple[str, str]]:
        """Uncached ``(cache_key, text)`` parts, de-duplicated, in PromptKey order."""
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for kp in self.keys:
            if only is not None and kp.key not in only:
                continue
            for p in kp.parts:
                if not p.cached and p.key not in seen:
                    seen.add(p.key)
                    out.append((p.key, p.text))
        return out


def timbre_config(lang: Lang, bank: BankManifest | None, model: str, voice: str) -> Any:
    """The bank's Timbre voice/model (B spec §11.4 step 1) unless it is the synthetic bank."""
    if bank is not None and bank.timbre.voice != TEST_VOICE:
        return bank.timbre
    return render.timbre_config(lang, model, voice)


def make_plan(
    lang: Lang,
    cfg: TimbreRenderConfig,
    texts: dict[PromptKey, tuple[str, str, str]],
    cache: render.RenderCache,
    max_chars: int,
) -> PromptPlan:
    """``texts``: key -> (copy_key, variant_id, text) for every non-digit key."""
    plan = PromptPlan(lang, cfg)
    for k in PromptKey:
        if k.value.startswith("DIGIT_"):
            continue
        copy_key, vid, text = texts[k]
        parts = []
        for t in split_text(text, max_chars):
            ck = render.cache_key(t, cfg.voice, cfg.model, cfg.language, cfg.speed)
            parts.append(Part(t, ck, cache.has(ck)))
        plan.keys.append(KeyPlan(k, copy_key, text, vid, tuple(parts)))
    return plan


def selected_texts(
    lang: Lang, copy_path: str, selection: str | None
) -> dict[PromptKey, tuple[str, str, str]]:
    table = load_copy(copy_path, selection)
    out: dict[PromptKey, tuple[str, str, str]] = {}
    for k in PromptKey:
        if k.value.startswith("DIGIT_"):
            continue
        ck = copy_key_for(k)
        v = table.variant(ck)
        out[k] = (ck, v.variant_id, v.text[lang.value])
    return out


def limit_jobs(
    jobs: list[tuple[str, str]], limit_credits: float | None, q: float
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Take parts in order while the cumulative estimate stays <= ``limit_credits``."""
    if limit_credits is None:
        return jobs, []
    take: list[tuple[str, str]] = []
    total = 0.0
    for i, (k, t) in enumerate(jobs):
        if total + len(t) * q > limit_credits:
            return take, jobs[i:]
        take.append((k, t))
        total += len(t) * q
    return take, []


def request_bodies(jobs: Sequence[tuple[str, str]], cfg: TimbreRenderConfig) -> list[Any]:
    return [render.request_body(t, cfg) for _, t in jobs]


# ---------- build (cache only, free) ----------
def decode16(data: bytes) -> F64:
    return dsp.resample_48k_to_16k(render.decode_wav48(data))


def digit16(bank_dir: Path, bank: BankManifest, d: int) -> F64:
    asset = next(a for a in bank.digits if a.digit == d)
    raw = (bank_dir / asset.file).read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset.sha256:
        raise render.RenderError(2, f"bank digit {d}: sha256 mismatch")
    x, fs = sf.read(io.BytesIO(raw), dtype="float64")
    if fs != 16000:
        raise render.RenderError(2, f"bank digit {d}: {fs} Hz")
    return np.asarray(x, dtype=np.float64)


def build(
    plan: PromptPlan,
    cache: render.RenderCache,
    bank_dir: Path,
    bank: BankManifest,
    out_root: Path,
    copy_sha256: str,
) -> PromptManifest:
    out = out_root / plan.lang.value
    out.mkdir(parents=True, exist_ok=True)
    assets: list[PromptAsset] = []
    gap = np.zeros(GAP_SAMPLES_16K)

    def emit(key: PromptKey, copy_key: str, text: str, x16: F64) -> None:
        ulaw = dsp.prompt_chain_16k(x16)
        name = f"{key.value}.ulaw"
        (out / name).write_bytes(ulaw)
        assets.append(
            PromptAsset(
                key=key,
                lang=plan.lang,
                copy_key=copy_key,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                file=name,
                sha256=hashlib.sha256(ulaw).hexdigest(),
                duration_ms=len(ulaw) / 8.0,
            )
        )

    for kp in plan.keys:
        chunks: list[F64] = []
        for i, p in enumerate(kp.parts):
            if not cache.has(p.key):
                raise render.RenderError(2, f"cache miss: {kp.key.value} part {i}")
            if i:
                chunks.append(gap)
            chunks.append(decode16(cache.get(p.key)))
        emit(kp.key, kp.copy_key, kp.text, np.concatenate(chunks))
    for d in bank.digit_set:
        k = PromptKey(f"DIGIT_{d}")
        word = next(a.word for a in bank.digits if a.digit == d)
        emit(k, copy_key_for(k), word, digit16(bank_dir, bank, d))

    assets.sort(key=lambda a: a.key.value)
    draft = PromptManifest(
        lang=plan.lang,
        voice=plan.config.voice,
        timbre_model=plan.config.model,
        copy_sha256=copy_sha256,
        assets=tuple(assets),
        prompts_hash="0" * 64,
    )
    m = draft.model_copy(update={"prompts_hash": compute_prompts_hash(draft)})
    text = m.model_dump_json(indent=2)
    (out / MANIFEST_NAME).write_text(text + "\n", encoding="utf-8")
    return m


# ---------- CLI ----------
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="scripts/render_prompts.py", description=__doc__)
    ap.add_argument("--lang", required=True, choices=[lg.value for lg in Lang])
    ap.add_argument("--bank", default=None, help="bank dir (default stimuli/<lang>)")
    ap.add_argument("--out", default="stimuli/prompts", type=Path)
    ap.add_argument("--copy", default=DEFAULT_COPY_PATH)
    ap.add_argument("--selection", default="results/copy_selection.json")
    ap.add_argument("--model", default=render.DEFAULT_MODEL, choices=["timbre-v3.0", "timbre-v2.5"])
    ap.add_argument("--voice", default=render.DEFAULT_VOICE)
    ap.add_argument(
        "--cache-dir", default=os.environ.get("SEVI_CAL__CACHE_DIR", "var/cache"), type=Path
    )
    ap.add_argument("--credits-per-char", type=float, default=render.DEFAULT_CREDITS_PER_CHAR)
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--only", default="", help="comma-separated PromptKeys to fetch")
    ap.add_argument("--limit-credits", type=float, default=None, help="cap for this batch")
    ap.add_argument("--client", default="sevi.calibration.render:strict_offline_client")
    ap.add_argument("--dry-run", action="store_true", help="plan only (the default w/o --yes)")
    ap.add_argument("--offline", action="store_true", help="cache only; a miss exits 2")
    ap.add_argument("--yes", action="store_true", help="confirm the real (billed) render")
    return ap


def load_client(spec: str) -> render.WavSynthesizer:
    if spec.endswith(":strict_offline_client"):
        mod, _, attr = spec.partition(":")
        factory = getattr(__import__(mod, fromlist=[attr]), attr)
        client: render.WavSynthesizer = factory(purpose="prompt_render")
        return client
    return render.load_client(spec)


def main(argv: list[str] | None = None, client: render.WavSynthesizer | None = None) -> int:
    a = _parser().parse_args(argv)
    lang = Lang(a.lang)
    cache = render.RenderCache(a.cache_dir)
    bank_dir = Path(a.bank) if a.bank else Path("stimuli") / lang.value
    q = a.credits_per_char
    try:
        bank = read_manifest(bank_dir / "manifest.json") if bank_dir.is_dir() else None
        if bank is not None and bank.lang is not lang:
            raise render.RenderError(2, f"bank {bank_dir} is {bank.lang.value}")
        cfg = timbre_config(lang, bank, a.model, a.voice)
        max_chars = a.max_chars or read_max_chars()
        texts = selected_texts(lang, a.copy, a.selection)
        plan = make_plan(lang, cfg, texts, cache, max_chars)
        only = None
        if a.only:
            only = frozenset(PromptKey(s.strip()) for s in a.only.split(",") if s.strip())
        all_jobs = plan.jobs(only)
        jobs, deferred = limit_jobs(all_jobs, a.limit_credits, q)
        chars = sum(len(t) for _, t in jobs)
        summary = {
            "step": "plan",
            "lang": lang.value,
            "model": cfg.model,
            "voice": cfg.voice,
            "max_chars": max_chars,
            "keys": [
                {
                    "key": kp.key.value,
                    "copy_key": kp.copy_key,
                    "variant": kp.variant_id,
                    "chars": sum(len(p.text) for p in kp.parts),
                    "parts": len(kp.parts),
                    "cached_parts": sum(p.cached for p in kp.parts),
                }
                for kp in plan.keys
                if only is None or kp.key in only
            ],
            "digits_from_bank": list(bank.digit_set) if bank else "bank missing",
            "requests": request_bodies(jobs, cfg),
            "chars": chars,
            "credits_per_char": q,
            "credits_estimate": chars * q,
            "deferred_requests": len(deferred),
            "deferred_chars": sum(len(t) for _, t in deferred),
        }
        print(json.dumps(summary, ensure_ascii=False))
        if a.dry_run:
            return 0
        if jobs:
            if a.offline:
                raise render.RenderError(2, "cache miss in --offline mode")
            if not a.yes:
                print(
                    f"{chars} chars (~{chars * q:.2f} credits) would be billed. "
                    "Re-run with --yes to render.",
                    file=sys.stderr,
                )
                return 1
            c = client or load_client(a.client)

            async def run() -> int:
                try:
                    return await render.fetch_texts(jobs, lang, cfg, c, cache)
                finally:
                    await render.aclose_client(c)

            calls = asyncio.run(run())
            print(json.dumps({"step": "rendered", "calls": calls, "chars": chars}))
            plan = make_plan(lang, cfg, texts, cache, max_chars)
        missing = len(plan.jobs())
        if missing:
            print(json.dumps({"step": "partial", "uncached_parts": missing}))
            return 0
        if bank is None:
            raise render.RenderError(2, f"no bank at {bank_dir}: render and freeze digits first")
        copy_sha = hashlib.sha256(Path(a.copy).read_bytes()).hexdigest()
        m = build(plan, cache, bank_dir, bank, a.out, copy_sha)
        print(json.dumps({"step": "built", "assets": len(m.assets), "hash": m.prompts_hash}))
        return 0
    except render.RenderError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    except SeviError as e:  # TtsError (non-200), BudgetExceeded: stop, send nothing more
        print(f"error: {e.code.value}: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
