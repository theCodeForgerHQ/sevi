"""Prisma v3 streaming accuracy on digit triplets vs SNR (EXP-13, issue #61).

The audio is exactly what a session plays: the shipped frozen bank (``stimuli/<lang>``, its
``noise.wav``) rendered by ``BankStimulusProvider.render_trial`` (mix at SNR -> band-pass ->
8 kHz -> mu-law), decoded to 8 kHz s16le and streamed through the REAL adapter
``sevi.adapters.prisma_stream.PrismaStreamClient`` (v3: one locale, 16 kHz wire with the
adapter's internal 2x upsampling). Transcripts are scored with ``core.scoring`` exactly as the
app scores a voice answer.

Plan (deterministic): for each language and SNR (default ``+4 0 -4 -8 -12`` dB) ``N`` (default 6)
triplets drawn without replacement from ``core.triplets.build_pool`` with
``derive_seed("bench-prisma", "<lang>:<snr>")``; noise seed per item
``derive_seed("bench-prisma", "noise:<lang>:<snr>:<i>")``. One stream per (lang, SNR) cell:
open once, play the triplets in sequence with ``--gap-s`` (0.8 s) of silence between, paced in
real time, commit, collect the transcripts, close.

Transcript -> triplet mapping: a final transcript received at time ``R`` belongs to the last
triplet whose final audio frame was sent at or before ``R`` (split segments of one answer are
joined in ``segment_index`` order, as the app does). The adapter's own frame-tag binding is
recorded alongside (``bound_pid``) as a cross-check.

Credit safety (BUILD-BRIEF §5):

* Without ``--yes`` the plan and the credit estimate (billed seconds x
  ``SEVI_LEDGER__CREDITS_PER_STT_S``, default 1.0 when unset) are printed and nothing is sent.
* ``--yes`` requires ``--max-credits``: the client is built on Key A with the persistent ledger
  from ``Settings`` (purpose ``bench``, bucket CAL) wrapped in :class:`CappedLedger`, so a
  reservation that would take THIS run past ``--max-credits`` (or the key past
  ``SEVI_LEDGER__BUDGET_CREDITS_PER_KEY``) raises ``BudgetExceeded`` before any I/O. Note: each
  stream reserves 60 s up front (adapter ``STREAM_RESERVE_S``), so the cap needs ~60 credits of
  headroom over the estimate; only the billed seconds are committed at close.
* The first adapter error (or budget refusal) stops the run; nothing is retried by this tool.
* ``--cells ta:+4,en:0`` runs a subset; a cell whose ``<out>/<lang>_<snr>.json`` exists is
  skipped (resume).

CLI::

    uv run python -m sevi.calibration.bench_prisma                     # plan + estimate only
    infisical run --env=dev -- uv run python -m sevi.calibration.bench_prisma \\
        --yes --max-credits 400                                        # live (Key A)
    uv run python -m sevi.calibration.bench_prisma --report            # markdown from summary

Exit codes: 0 ok; 1 confirmation required (no ``--yes`` / no ``--max-credits``); 2 bad
arguments or bank; 5 the adapter or the ledger refused (run stopped).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from sevi.adapters.deps import BillingLedger, Purpose, Reservation
from sevi.contracts.adapters import (
    Clock,
    FrameTag,
    LedgerEvent,
    SttClosed,
    SttErrorMsg,
    SttSessionConfig,
    SttTranscript,
    VadConfig,
)
from sevi.contracts.constants import PRISMA_FRAME_BYTES
from sevi.contracts.enums import KeyId, Lang, Mode, PresentationKind, WindowCloseReason
from sevi.contracts.errors import BudgetExceeded, SeviError
from sevi.contracts.rng import derive_seed
from sevi.contracts.trial import AnswerEvidence, PresentationSpec, SttSegment
from sevi.core import dsp
from sevi.core.bank import load_bank, read_manifest
from sevi.core.scoring import LexiconScorer, load_lexicons
from sevi.core.stimulus import BankStimulusProvider
from sevi.core.triplets import build_pool

SEED_SCOPE: Final = "bench-prisma"
DEFAULT_SNRS: Final = (4.0, 0.0, -4.0, -8.0, -12.0)
DEFAULT_N: Final = 6
DEFAULT_GAP_S: Final = 0.8
DEFAULT_SETTLE_S: Final = 6.0  # wait after commit for the last transcripts
CONNECT_OVERHEAD_S: Final = 2.0  # estimate: wall seconds billed beyond the audio (connect)
FS: Final = 8000
FRAME_SAMPLES: Final = PRISMA_FRAME_BYTES // 2  # 512 samples = 64 ms at 8 kHz
FRAME_MS: Final = FRAME_SAMPLES * 1000.0 / FS
PURPOSE: Final[Purpose] = "bench"
SUMMARY: Final = "summary.json"


class StreamFailure(Exception):
    """The stream reported a server error or closed early inside a cell."""


class BenchError(Exception):
    def __init__(self, exit_code: int, msg: str) -> None:
        super().__init__(msg)
        self.exit_code = exit_code


# ---------------------------------------------------------------- plan
@dataclass(frozen=True)
class Item:
    index: int  # 1-based within the cell (= FrameTag.presentation_id)
    triplet: tuple[int, int, int]
    pool_index: int
    noise_seed: int


@dataclass(frozen=True)
class Cell:
    lang: Lang
    snr_db: float
    items: tuple[Item, ...]

    @property
    def cell_id(self) -> str:
        return f"{self.lang.value}:{snr_label(self.snr_db)}"

    @property
    def file_name(self) -> str:
        return f"{self.lang.value}_{snr_label(self.snr_db)}.json"


def snr_label(snr: float) -> str:
    return f"{snr:+g}" if snr != 0 else "0"


def parse_snr(s: str) -> float:
    v = float(s)
    if not math.isfinite(v) or not -20.0 <= v <= 10.0:
        raise ValueError(f"SNR {s!r} outside [-20, +10] dB")
    return v


def make_cell(lang: Lang, digit_set: tuple[int, ...], snr: float, n: int) -> Cell:
    pool = build_pool(digit_set)
    if not 1 <= n <= len(pool):
        raise ValueError(f"n must be in 1..{len(pool)}")
    lab = snr_label(snr)
    gen = np.random.Generator(np.random.PCG64(derive_seed(SEED_SCOPE, f"{lang.value}:{lab}")))
    idx = [int(i) for i in gen.permutation(len(pool))[:n]]
    items = tuple(
        Item(
            index=k + 1,
            triplet=pool[i],
            pool_index=i,
            noise_seed=derive_seed(SEED_SCOPE, f"noise:{lang.value}:{lab}:{k + 1}"),
        )
        for k, i in enumerate(idx)
    )
    return Cell(lang, snr, items)


def make_plan(
    digit_sets: dict[Lang, tuple[int, ...]], snrs: Sequence[float], n: int
) -> tuple[Cell, ...]:
    return tuple(make_cell(lg, ds, s, n) for lg, ds in digit_sets.items() for s in snrs)


def parse_cells(spec: str) -> set[str]:
    """``"ta:+4,en:-8"`` -> canonical cell ids."""
    out: set[str] = set()
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        lang, _, snr = part.partition(":")
        out.add(f"{Lang(lang).value}:{snr_label(parse_snr(snr))}")
    return out


# ---------------------------------------------------------------- audio
def spec_for(cell: Cell, item: Item) -> PresentationSpec:
    return PresentationSpec(
        presentation_id=item.index,
        kind=PresentationKind.PRACTICE,
        snr_db=cell.snr_db,
        triplet=item.triplet,
        pool_index=item.pool_index,
        noise_seed=item.noise_seed,
    )


def item_frames(bank: BankStimulusProvider, cell: Cell, item: Item) -> list[bytes]:
    """The session's 8 kHz mu-law clip decoded to s16le, cut into 1,024-byte frames (zero-padded
    last frame)."""
    rt = bank.render_trial(spec_for(cell, item))
    pcm = dsp.ulaw_decode(rt.ulaw).astype("<i2")
    pad = (-pcm.size) % FRAME_SAMPLES
    if pad:
        pcm = np.concatenate([pcm, np.zeros(pad, dtype="<i2")])
    raw = pcm.tobytes()
    return [raw[i : i + PRISMA_FRAME_BYTES] for i in range(0, len(raw), PRISMA_FRAME_BYTES)]


def gap_frames(gap_s: float) -> int:
    return max(1, math.ceil(gap_s * 1000.0 / FRAME_MS))


def cell_audio_s(frames: Sequence[Sequence[bytes]], gap_s: float) -> float:
    n = sum(len(f) + gap_frames(gap_s) for f in frames)
    return n * FRAME_MS / 1000.0


# ---------------------------------------------------------------- credit cap
class CappedLedger:
    """`BillingLedger` wrapper: refuses (BudgetExceeded, before any I/O) once this run's
    committed + outstanding credits would exceed ``max_credits``. The wrapped ledger still
    enforces the per-key budget."""

    def __init__(self, inner: BillingLedger, max_credits: float) -> None:
        if not max_credits > 0:
            raise ValueError("max_credits must be > 0")
        self.inner = inner
        self.max_credits = max_credits
        self.committed = 0.0
        self._open: dict[str, float] = {}

    @property
    def used(self) -> float:
        return self.committed + sum(self._open.values())

    def _check(self, more: float) -> None:
        if self.used + more > self.max_credits:
            raise BudgetExceeded(
                f"--max-credits {self.max_credits:g} would be exceeded",
                details={"run_used": round(self.used, 3), "requested": round(more, 3)},
            )

    def _est(self, stt_s: float, tts_chars: int) -> float:
        credits = getattr(self.inner, "credits", None)
        if callable(credits):
            return float(credits(stt_s=stt_s, tts_chars=tts_chars))
        return float(stt_s)  # conservative 1 credit/s for ledgers without a rate

    def billed(self, key_id: KeyId, purpose: Purpose, **kw: Any) -> Any:
        raise NotImplementedError("bench streams use reserve/top_up/commit only")

    async def reserve(
        self, key_id: KeyId, purpose: Purpose, *, stt_s: float = 0, tts_chars: int = 0
    ) -> Reservation:
        self._check(self._est(stt_s, tts_chars))
        r = await self.inner.reserve(key_id, purpose, stt_s=stt_s, tts_chars=tts_chars)
        self._open[r.reservation_id] = r.credits
        return r

    async def top_up(self, r: Reservation, *, stt_s: float = 0, tts_chars: int = 0) -> None:
        self._check(self._est(stt_s, tts_chars))
        await self.inner.top_up(r, stt_s=stt_s, tts_chars=tts_chars)
        self._open[r.reservation_id] = r.credits

    async def commit(
        self,
        r: Reservation,
        *,
        stt_s: float = 0,
        tts_chars: int = 0,
        stt_rest_calls: int = 0,
        window_id: str | None = None,
    ) -> LedgerEvent:
        ev = await self.inner.commit(
            r, stt_s=stt_s, tts_chars=tts_chars, stt_rest_calls=stt_rest_calls, window_id=window_id
        )
        self._open.pop(r.reservation_id, None)
        self.committed += ev.credits_est
        return ev

    async def release(self, r: Reservation) -> None:
        await self.inner.release(r)
        self._open.pop(r.reservation_id, None)

    async def mark_exhausted(self, key_id: KeyId) -> None:
        await self.inner.mark_exhausted(key_id)

    async def cal_stop(self) -> bool:
        return await self.inner.cal_stop()


# ---------------------------------------------------------------- run one cell
class StreamClient(Protocol):  # structural: PrismaStreamClient
    async def open(self, cfg: SttSessionConfig) -> Any: ...


@dataclass
class SttParams:
    model_version: Any
    format: str = "transcribe"
    keyterms: tuple[str, ...] = ()
    vad: VadConfig = field(default_factory=VadConfig)
    idle_timeout_s: int = 900


def stt_config(cell: Cell, p: SttParams, run_tag: str) -> SttSessionConfig:
    return SttSessionConfig(
        key_id=KeyId.A,
        lang=cell.lang,
        model_version=p.model_version,
        format=p.format,
        keyterms=p.keyterms,
        sample_rate_hz=8000,
        vad=p.vad,
        idle_timeout_s=p.idle_timeout_s,
        request_id=f"bench-{run_tag}-{cell.lang.value}{snr_label(cell.snr_db)}"[:80],
    )


def assign_transcripts(
    ends_ms: Sequence[float], transcripts: Sequence[SttTranscript]
) -> list[list[SttTranscript]]:
    """Each transcript -> the last item whose final frame was sent at or before its arrival
    (before the first end -> item 1)."""
    out: list[list[SttTranscript]] = [[] for _ in ends_ms]
    for t in transcripts:
        k = 0
        for i, e in enumerate(ends_ms):
            if e <= t.received_mono_ms:
                k = i
        out[k].append(t)
    for group in out:
        group.sort(key=lambda t: (t.segment_index is None, t.segment_index or 0))
    return out


def score_item(
    scorer: LexiconScorer, cell: Cell, item: Item, group: Sequence[SttTranscript], clip_ms: float
) -> dict[str, Any]:
    """Score one triplet with ``LexiconScorer.score`` exactly as the app scores a voice
    window (speech present, no early answer, first window)."""
    spec = spec_for(cell, item)
    segs = tuple(
        SttSegment(
            text=t.text[:500],
            segment_id=t.segment_id,
            segment_index=t.segment_index if (t.segment_index or 0) >= 1 else None,
            received_ms=0.0,
            audio_duration_ms=t.audio_duration_ms,
            latency_ms=t.latency_ms,
            after_commit=False,
        )
        for t in group
    )[:20]
    ev = AnswerEvidence(
        presentation_id=item.index,
        lang=cell.lang,
        mode=Mode.VOICE,
        window_index=1,
        window_open_mono_ms=0.0,
        window_close_mono_ms=clip_ms,
        close_reason=WindowCloseReason.SETTLE,
        segments=segs,
        speech_present=True,
        speech_ms=int(clip_ms),
        early_speech=False,
        echo_mode=False,
        reask_used=False,
        replay_used=False,
    )
    r = scorer.score(spec, ev)
    return {
        "status": r.status.value,
        "recognised": list(r.recognised),
        "triplet_correct": r.triplet_correct,
        "position_correct": list(r.position_correct),
    }


async def run_cell(
    client: StreamClient,
    bank: BankStimulusProvider,
    scorer: LexiconScorer,
    cell: Cell,
    params: SttParams,
    clock: Clock,
    *,
    gap_s: float = DEFAULT_GAP_S,
    settle_s: float = DEFAULT_SETTLE_S,
    run_tag: str = "r",
) -> dict[str, Any]:
    """One stream for one (lang, SNR) cell. Any adapter error raises (``SeviError``)."""
    frames = [item_frames(bank, cell, it) for it in cell.items]
    silence = bytes(PRISMA_FRAME_BYTES)
    n_gap = gap_frames(gap_s)
    session = await client.open(stt_config(cell, params, run_tag))
    transcripts: list[SttTranscript] = []
    failure: list[str] = []
    done = asyncio.Event()
    ends_ms: list[float] = []

    async def reader() -> None:
        async for ev in session.events():
            if isinstance(ev, SttTranscript) and ev.text.strip():
                transcripts.append(ev)
                if len(transcripts) >= len(cell.items) and len(ends_ms) == len(cell.items):
                    done.set()
            elif isinstance(ev, SttErrorMsg):
                failure.append(f"server error: {ev.message}")
                done.set()
            elif isinstance(ev, SttClosed):
                if not done.is_set() and ev.reason != "client close":
                    failure.append(f"stream closed early: code={ev.code} reason={ev.reason}")
                done.set()
                return

    rtask = asyncio.create_task(reader())
    try:
        t0 = clock.mono_ms()
        k = 0
        for it, fr in zip(cell.items, frames, strict=True):
            tag = FrameTag(presentation_id=it.index, window_index=1)
            for f in fr:
                await clock.sleep_until(t0 + k * FRAME_MS)
                if failure:
                    raise StreamFailure(failure[0])
                session.send_frame(f, tag)
                k += 1
            ends_ms.append(clock.mono_ms())
            for _ in range(n_gap):
                await clock.sleep_until(t0 + k * FRAME_MS)
                if failure:
                    raise StreamFailure(failure[0])
                session.send_frame(silence, tag)
                k += 1
        await clock.sleep_until(t0 + k * FRAME_MS)
        session.commit()
        if len(transcripts) >= len(cell.items):
            done.set()
        deadline = clock.mono_ms() + settle_s * 1000.0

        async def settle() -> None:
            await clock.sleep_until(deadline)
            done.set()

        stask = asyncio.create_task(settle())
        await done.wait()
        stask.cancel()
        if failure:
            raise StreamFailure(failure[0])
    finally:
        await session.close()
        rtask.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await rtask

    groups = assign_transcripts(ends_ms, transcripts)
    items_out: list[dict[str, Any]] = []
    for it, fr, end, group in zip(cell.items, frames, ends_ms, groups, strict=True):
        clip_ms = len(fr) * FRAME_MS
        sc = score_item(scorer, cell, it, group, clip_ms)
        items_out.append(
            {
                "index": it.index,
                "presented": list(it.triplet),
                "pool_index": it.pool_index,
                "noise_seed": str(it.noise_seed),
                "transcripts": [t.text for t in group],
                "segment_index": [t.segment_index for t in group],
                "bound_pid": [t.bound.presentation_id if t.bound else None for t in group],
                "latency_ms": round(group[0].received_mono_ms - end, 1) if group else None,
                "server_latency_ms": [t.latency_ms for t in group],
                **sc,
                "digit_correct": sc["position_correct"],
            }
        )
    return {
        "cell": cell.cell_id,
        "lang": cell.lang.value,
        "snr_db": cell.snr_db,
        "bank_hash": bank.manifest.bank_hash,
        "model_version": str(params.model_version),
        "format": params.format,
        "keyterms": len(params.keyterms),
        "gap_s": gap_s,
        "audio_s": round(cell_audio_s(frames, gap_s), 3),
        "request_id": getattr(session, "request_id", None),
        "items": items_out,
    }


# ---------------------------------------------------------------- summary / report
def _pct(xs: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(np.asarray(xs, dtype=float), q)), 1) if xs else None


def summarize(cells: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, dict[str, int]]] = {}
    lat: dict[str, list[float]] = {}
    for c in sorted(cells, key=lambda c: (c["lang"], -c["snr_db"])):
        items = c["items"]
        n = len(items)
        trip = sum(1 for i in items if i["triplet_correct"])
        dig = sum(sum(1 for p in i["position_correct"] if p) for i in items)
        cl = [float(i["latency_ms"]) for i in items if i["latency_ms"] is not None]
        lat.setdefault(c["lang"], []).extend(cl)
        statuses: dict[str, int] = {}
        conf = confusion.setdefault(c["lang"], {})
        for i in items:
            statuses[i["status"]] = statuses.get(i["status"], 0) + 1
            rec = i["recognised"]
            for pos, d in enumerate(i["presented"]):
                if len(rec) == 3:
                    got = str(rec[pos])
                else:
                    got = str(d) if i["position_correct"][pos] else "miss"
                row = conf.setdefault(str(d), {})
                row[got] = row.get(got, 0) + 1
        rows.append(
            {
                "lang": c["lang"],
                "snr_db": c["snr_db"],
                "n": n,
                "triplet_acc": round(trip / n, 4) if n else None,
                "digit_acc": round(dig / (3 * n), 4) if n else None,
                "latency_p50_ms": _pct(cl, 50),
                "latency_p95_ms": _pct(cl, 95),
                "status": dict(sorted(statuses.items())),
            }
        )
    return {
        "experiment": "EXP-13",
        "cells": rows,
        "confusion": confusion,
        "latency": {
            lg: {"p50_ms": _pct(v, 50), "p95_ms": _pct(v, 95), "n": len(v)}
            for lg, v in sorted(lat.items())
        },
    }


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Lang | SNR (dB) | n | Triplet correct | Digit correct | Latency p50 / p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["cells"]:
        ta = "-" if r["triplet_acc"] is None else f"{100 * r['triplet_acc']:.0f} %"
        da = "-" if r["digit_acc"] is None else f"{100 * r['digit_acc']:.0f} %"
        lt = f"{r['latency_p50_ms'] or '-'} / {r['latency_p95_ms'] or '-'}"
        lines.append(f"| {r['lang']} | {snr_label(r['snr_db'])} | {r['n']} | {ta} | {da} | {lt} |")
    return "\n".join(lines) + "\n"


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")
    tmp.replace(path)


def load_cells(out: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text("utf-8")) for p in sorted(out.glob("*.json")) if p.name != SUMMARY
    ]


# ---------------------------------------------------------------- live client
ClientFactory = Callable[[float], Awaitable[tuple[StreamClient, Callable[[], Awaitable[None]]]]]


async def live_client(max_credits: float) -> tuple[StreamClient, Callable[[], Awaitable[None]]]:
    """Key A, persistent ledger from ``Settings`` (purpose ``bench``), capped per run."""
    from sevi.adapters.breaker import BreakerRegistry
    from sevi.adapters.ledger import Ledger
    from sevi.adapters.prisma_stream import PrismaOptions, PrismaStreamClient
    from sevi.app.clock import SystemClock
    from sevi.app.settings import Settings
    from sevi.app.store import SqliteStore
    from sevi.fakes.null_telemetry import NullTelemetry

    s = Settings()
    key_a = s.gnani.key_a
    if key_a is None or not key_a.get_secret_value():
        raise BenchError(2, "SEVI_GNANI__KEY_A is not set (run under infisical)")
    clk, tel = SystemClock(), NullTelemetry()
    store = await SqliteStore(s.sqlite_path, clock=clk, telemetry=tel).open()
    led = Ledger(store, s.ledger, alias_b=s.key_b_aliased, clock=clk, telemetry=tel)
    client = PrismaStreamClient(
        PrismaOptions.from_settings(s),
        {KeyId.A: key_a},
        BreakerRegistry(clk, tel, alias_b=s.key_b_aliased),
        CappedLedger(led, max_credits),
        clk,
        tel,
        purpose=PURPOSE,
    )
    return client, store.close


def default_params(scorer: LexiconScorer, lang: Lang, mv: Any) -> SttParams:
    """The app's STT config (orchestrator ``_stt_config``) from ``Settings`` defaults/env."""
    from sevi.app.settings import Settings

    p = Settings().prisma
    return SttParams(
        model_version=mv,
        format=p.format,
        keyterms=scorer.keyterms(lang, mv) if p.keyterms_enabled else (),
        vad=VadConfig(
            threshold=p.vad_threshold,
            min_silence_ms=p.min_silence_ms,
            min_speech_ms=p.min_speech_ms,
            min_segment_ms=p.min_segment_ms,
        ),
        idle_timeout_s=p.idle_timeout_s,
    )


def credits_per_stt_s() -> float:
    from sevi.app.settings import Settings

    rate = Settings().ledger.credits_per_stt_s
    return float(rate) if rate is not None else 1.0


# ---------------------------------------------------------------- CLI
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m sevi.calibration.bench_prisma", description=__doc__
    )
    ap.add_argument("--langs", default="ta,en")
    ap.add_argument("--snrs", default=",".join(snr_label(s) for s in DEFAULT_SNRS))
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="triplets per (lang, SNR) cell")
    ap.add_argument("--cells", default="", help="subset, e.g. ta:+4,en:-8")
    ap.add_argument("--bank-dir", default="stimuli", type=Path)
    ap.add_argument("--out", default="var/bench", type=Path)
    ap.add_argument("--gap-s", type=float, default=DEFAULT_GAP_S)
    ap.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    ap.add_argument("--yes", action="store_true", help="confirm the live (billed) run")
    ap.add_argument("--max-credits", type=float, default=None, help="hard cap for this run")
    ap.add_argument("--report", action="store_true", help="print the markdown table and exit")
    return ap


def load_banks(bank_dir: Path, langs: Sequence[Lang]) -> dict[Lang, BankStimulusProvider]:
    banks: dict[Lang, BankStimulusProvider] = {}
    for lg in langs:
        m = read_manifest(bank_dir / lg.value / "manifest.json")
        if not m.frozen:
            raise BenchError(2, f"{lg.value} bank is not frozen")
        banks[lg] = load_bank(lg, str(bank_dir), {lg: m.bank_hash}, allow_unfrozen=False)
    return banks


async def run(
    cells: Sequence[Cell],
    banks: dict[Lang, BankStimulusProvider],
    scorer: LexiconScorer,
    client: StreamClient,
    clock: Clock,
    out: Path,
    *,
    params: Callable[[Lang], SttParams],
    gap_s: float,
    settle_s: float,
) -> list[str]:
    """Run the cells in order; write each cell's JSON as it completes. Raises on the first
    adapter error or budget refusal (nothing further is sent)."""
    done: list[str] = []
    for c in cells:
        res = await run_cell(
            client,
            banks[c.lang],
            scorer,
            c,
            params(c.lang),
            clock,
            gap_s=gap_s,
            settle_s=settle_s,
        )
        write_json(out / c.file_name, res)
        print(json.dumps({"step": "cell", "cell": c.cell_id, "file": str(out / c.file_name)}))
        done.append(c.cell_id)
    return done


def main(
    argv: list[str] | None = None,
    *,
    client_factory: ClientFactory | None = None,
    clock: Clock | None = None,
    rate: float | None = None,
) -> int:
    a = _parser().parse_args(argv)
    out: Path = a.out
    if a.report:
        p = out / SUMMARY
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            return 2
        print(report_markdown(json.loads(p.read_text("utf-8"))), end="")
        return 0
    try:
        langs = [Lang(x.strip()) for x in a.langs.split(",") if x.strip()]
        snrs = [parse_snr(x) for x in a.snrs.split(",") if x.strip()]
        only = parse_cells(a.cells) if a.cells else None
        banks = load_banks(a.bank_dir, langs)
        plan = make_plan({lg: banks[lg].manifest.digit_set for lg in langs}, snrs, a.n)
    except (ValueError, SeviError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.exit_code
    selected = [c for c in plan if only is None or c.cell_id in only]
    todo = [c for c in selected if not (out / c.file_name).exists()]
    per_rate = rate if rate is not None else credits_per_stt_s()
    audio = {
        c.cell_id: cell_audio_s([item_frames(banks[c.lang], c, i) for i in c.items], a.gap_s)
        for c in todo
    }
    billed_s = sum(v + a.settle_s + CONNECT_OVERHEAD_S for v in audio.values())
    est = {
        "step": "plan",
        "cells": [
            {
                "cell": c.cell_id,
                "triplets": [list(i.triplet) for i in c.items],
                "audio_s": round(audio[c.cell_id], 2) if c.cell_id in audio else None,
                "skip": c not in todo,
            }
            for c in selected
        ],
        "to_run": len(todo),
        "billed_s_max": round(billed_s, 1),
        "credits_per_stt_s": per_rate,
        "credits_estimate": round(billed_s * per_rate, 1),
        "reserve_headroom_credits": 60.0 * per_rate,
    }
    print(json.dumps(est, ensure_ascii=False))
    if todo:
        if not a.yes or a.max_credits is None:
            print(
                f"~{est['credits_estimate']} credits would be billed. Re-run with --yes "
                f"--max-credits <cap> (cap >= estimate + {60.0 * per_rate:g} reserve headroom).",
                file=sys.stderr,
            )
            return 1
        scorer = load_lexicons()
        mvs = {lg: banks[lg].manifest.model_version for lg in langs}
        params = {lg: default_params(scorer, lg, mvs[lg]) for lg in langs}

        async def go() -> None:
            factory = client_factory or live_client
            client, close = await factory(float(a.max_credits))
            try:
                from sevi.app.clock import SystemClock

                await run(
                    todo,
                    banks,
                    scorer,
                    client,
                    clock or SystemClock(),
                    out,
                    params=params.__getitem__,
                    gap_s=a.gap_s,
                    settle_s=a.settle_s,
                )
            finally:
                await close()

        try:
            asyncio.run(go())
        except BenchError as e:
            print(f"error: {e}", file=sys.stderr)
            return e.exit_code
        except (SeviError, StreamFailure) as e:  # BudgetExceeded, SttError, stream failure: stop
            code = getattr(getattr(e, "code", None), "value", "STREAM")
            print(f"error: {code}: {e} (run stopped; completed cells kept)", file=sys.stderr)
            _write_summary(out)
            return 5
    _write_summary(out)
    return 0


def _write_summary(out: Path) -> None:
    cells = load_cells(out) if out.exists() else []
    if cells:
        write_json(out / SUMMARY, summarize(cells))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
