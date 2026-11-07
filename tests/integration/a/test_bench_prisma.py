"""EXP-13 bench tooling (issue #61) end to end against fake_prisma: the real
PrismaStreamClient over a socket, the real shipped bank and the real scorer. No live calls."""

from __future__ import annotations

import json
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from sevi.adapters.prisma_stream import PrismaOptions, PrismaStreamClient
from sevi.calibration import bench_prisma as bp
from sevi.contracts.adapters import SttTranscript
from sevi.contracts.enums import KeyId, Lang
from sevi.core.triplets import build_pool
from sevi.fakes.clock import ScaledClock
from sevi.fakes.fake_prisma import FakePrisma, Rule
from sevi.fakes.ledger_kit import open_ledger
from sevi.fakes.null_telemetry import NullTelemetry
from sevi.fakes.recording_deps import RecordingBreakers

ARGS = ["--langs", "ta", "--snrs", "+4,-8", "--n", "2", "--commit-wait-s", "2", "--quiet-s", "0.3"]
DS = {Lang.TA: (1, 2, 3, 4, 5, 6, 7, 8), Lang.EN: (1, 2, 3, 4, 5, 6, 8, 9)}


def _texts(wrong: set[int] | None = None) -> list[str]:
    """Utterance texts in stream order for ARGS (cell ta:+4 then ta:-8); utterance k in
    `wrong` gets its last digit replaced."""
    out = [
        list(it.triplet)
        for c in bp.make_plan({Lang.TA: DS[Lang.TA]}, [4.0, -8.0], 2)
        for it in c.items
    ]
    res = []
    for k, t in enumerate(out):
        if wrong and k in wrong:
            t = [t[0], t[1], 1 if t[2] != 1 else 2]
        res.append(" ".join(str(d) for d in t))
    return res


def _factory(
    tmp: Path, texts: list[str], *, budget: float = 1000.0
) -> tuple[Callable[[float], Awaitable[Any]], ScaledClock, list[int]]:
    clock = ScaledClock(speed=40.0)
    calls: list[int] = []
    state: dict[str, Any] = {}

    async def factory(max_credits: float) -> Any:
        calls.append(1)
        kit = await open_ledger(tmp / "ledger.sqlite", budget=budget)
        # one connection per cell (2 triplets each): utterance k of connection j -> 2j + k

        def text(k: int) -> str:
            i = 2 * (len(fake.connections) - 1) + k
            return texts[i] if i < len(texts) else ""

        fake = FakePrisma([Rule("", after_voiced_frames=4)], text_source=text, clock=clock)
        url = await fake.start()
        state["kit"] = kit
        client = PrismaStreamClient(
            PrismaOptions(ws_url=url, connect_timeout_s=2.0, send_timeout_s=2.0),
            {KeyId.A: "KA"},
            RecordingBreakers(),
            bp.CappedLedger(kit.ledger, max_credits),
            clock,
            NullTelemetry(),
            purpose="bench",
            rng=random.Random(1),  # noqa: S311
        )

        async def close() -> None:
            await fake.stop()
            await kit.close()

        return client, close

    return factory, clock, calls


def test_U_BEN_plan_deterministic() -> None:
    a = bp.make_plan(DS, bp.DEFAULT_SNRS, 6)
    b = bp.make_plan(DS, bp.DEFAULT_SNRS, 6)
    assert a == b
    assert len(a) == 10
    for c in a:
        pool = build_pool(DS[c.lang])
        assert len({i.pool_index for i in c.items}) == 6
        assert all(pool[i.pool_index] == i.triplet for i in c.items)
    seeds = [i.noise_seed for c in a for i in c.items]
    assert len(set(seeds)) == len(seeds)
    assert a[0].cell_id == "ta:+4"
    assert a[-1].cell_id == "en:-12"
    assert bp.parse_cells("ta:+4, en:-8,ta:0") == {"ta:+4", "en:-8", "ta:0"}


def test_assign_transcripts_by_arrival() -> None:
    def tr(t: float, seg: int, text: str) -> SttTranscript:
        return SttTranscript(
            text=text,
            segment_id=None,
            segment_index=seg,
            audio_duration_ms=None,
            latency_ms=None,
            detected_language_code=None,
            received_mono_ms=t,
        )

    g = bp.assign_transcripts(
        [100.0, 200.0, 300.0], [tr(150, 1, "a"), tr(260, 3, "c"), tr(250, 2, "b"), tr(90, 0, "z")]
    )
    assert [[t.text for t in x] for x in g] == [["z", "a"], ["b", "c"], []]


def test_requires_yes_and_cap(tmp_path: Path) -> None:
    factory, clock, calls = _factory(tmp_path, _texts())
    out = ["--out", str(tmp_path / "out")]
    assert bp.main([*ARGS, *out], client_factory=factory, clock=clock, rate=1.0) == 1
    assert bp.main([*ARGS, *out, "--yes"], client_factory=factory, clock=clock, rate=1.0) == 1
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_end_to_end_fake_scores_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory, clock, calls = _factory(tmp_path, _texts(wrong={3}))
    out = tmp_path / "out"
    argv = [*ARGS, "--out", str(out), "--yes", "--max-credits", "500"]
    assert bp.main(argv, client_factory=factory, clock=clock, rate=1.0) == 0
    assert calls == [1]
    hi = json.loads((out / "ta_+4.json").read_text())
    lo = json.loads((out / "ta_-8.json").read_text())
    assert [i["triplet_correct"] for i in hi["items"]] == [True, True]
    assert [i["triplet_correct"] for i in lo["items"]] == [True, False]
    bad = lo["items"][1]
    assert bad["status"] == "OK"
    assert bad["digit_correct"] == [True, True, False]
    assert bad["transcripts"]
    assert bad["latency_ms"] is not None
    first = bp.make_plan({Lang.TA: DS[Lang.TA]}, [4.0], 2)[0].items[0]
    assert hi["items"][0]["presented"] == list(first.triplet)
    s = json.loads((out / "summary.json").read_text())
    rows = {(r["lang"], r["snr_db"]): r for r in s["cells"]}
    assert rows[("ta", 4.0)]["triplet_acc"] == 1.0
    assert rows[("ta", -8.0)]["triplet_acc"] == 0.5
    assert rows[("ta", -8.0)]["digit_acc"] == round(5 / 6, 4)
    assert s["latency"]["ta"]["n"] == 4
    # resume: every cell exists -> nothing to run, the factory is not called again
    assert bp.main(argv, client_factory=factory, clock=clock, rate=1.0) == 0
    assert calls == [1]
    capsys.readouterr()
    assert bp.main(["--report", "--out", str(out)]) == 0
    md = capsys.readouterr().out
    assert "| ta | +4 | 2 | 100 % | 100 % |" in md
    assert "| ta | -8 | 2 | 50 % | 83 % |" in md


def test_cells_subset_runs_only_that_cell(tmp_path: Path) -> None:
    factory, clock, _ = _factory(tmp_path, _texts())
    out = tmp_path / "out"
    argv = [*ARGS, "--out", str(out), "--yes", "--max-credits", "500", "--cells", "ta:+4"]
    assert bp.main(argv, client_factory=factory, clock=clock, rate=1.0) == 0
    assert sorted(p.name for p in out.iterdir()) == ["summary.json", "ta_+4.json"]


@pytest.mark.parametrize(("budget", "cap"), [(30.0, 500.0), (1000.0, 30.0)])
def test_budget_refusal_stops_run(tmp_path: Path, budget: float, cap: float) -> None:
    """Key budget (ledger) or --max-credits (CappedLedger) below the 60 s stream reservation:
    BudgetExceeded before any I/O, exit 5, no cell written."""
    factory, clock, _ = _factory(tmp_path, _texts(), budget=budget)
    out = tmp_path / "out"
    argv = [*ARGS, "--out", str(out), "--yes", "--max-credits", str(cap)]
    assert bp.main(argv, client_factory=factory, clock=clock, rate=1.0) == 5
    assert not out.exists() or not list(out.glob("ta_*.json"))
