"""Commit-bound transcript mapping for the EXP-13 bench (issue #61, schema 2).

Live Prisma (lead's ta:+4 run) split triplets into several segments, finalised some after the
next triplet had started and dropped/emptied others; the v1 arrival-time mapping then gave
triplet 1 the tail of triplet 2 and triplet 4 the head of triplet 5. These tests pin that exact
pattern and check the commit-bound mapping, end to end against a fake that reproduces it."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from sevi.adapters.prisma_stream import PrismaOptions, PrismaStreamClient
from sevi.calibration import bench_prisma as bp
from sevi.contracts.adapters import FrameTag, SttTranscript
from sevi.contracts.enums import KeyId, Lang
from sevi.core.scoring import load_lexicons
from sevi.fakes.clock import ScaledClock
from sevi.fakes.fake_prisma import FakePrisma, FakePrismaConnection, _UttState
from sevi.fakes.ledger_kit import open_ledger
from sevi.fakes.null_telemetry import NullTelemetry
from sevi.fakes.recording_deps import RecordingBreakers

# The live ta:+4 cell: presented triplets and the segments Prisma returned.
TA4 = [(5, 3, 1), (3, 4, 5), (7, 4, 6), (5, 8, 6), (7, 1, 5), (8, 3, 6)]
STEP, SPEECH = 3700.0, 2900.0  # v1 continuous stream: triplet i starts at STEP*(i-1)
# (segment_index, text, received_ms, adapter-bound pid); segments 2, 3 never arrived
TA4_SEGMENTS = [
    (1, "5", 1287.0, 1),
    (4, "4 5", 6500.0, 2),
    (5, "7, 4, 6.", 10500.0, 3),
    (6, "5, 8, 6.", 14300.0, 4),
    (7, "ஏழு.", 15500.0, 5),
    (8, "5", 16500.0, 5),
    (9, "எட்டு.", 19500.0, 6),
]


def _tr(seg: int, text: str, t: float, pid: int | None) -> SttTranscript:
    return SttTranscript(
        text=text,
        segment_id=f"s{seg}",
        segment_index=seg,
        audio_duration_ms=600,
        latency_ms=300,
        detected_language_code=None,
        received_mono_ms=t,
        bound=FrameTag(presentation_id=pid, window_index=1) if pid else None,
    )


def test_ta4_live_pattern_old_mapping_wrong_new_mapping_right() -> None:
    trs = [_tr(*s) for s in TA4_SEGMENTS]
    ends = [STEP * i + SPEECH for i in range(6)]
    old = [[t.text for t in g] for g in bp.assign_transcripts(ends, trs)]
    # what the lead saw with v1
    assert old[0] == ["5", "4 5"]
    assert old[1] == []
    assert old[3] == ["5, 8, 6.", "ஏழு.", "5"]
    assert old[4] == ["எட்டு."]
    windows = [bp.Window(i + 1, STEP * i, STEP * i + 3200, STEP * (i + 1)) for i in range(6)]
    b = bp.bind_windows(windows, trs)
    new = {pid: [t.text for t in g] for pid, g in b.groups.items()}
    assert new == {
        1: ["5"],
        2: ["4 5"],
        3: ["7, 4, 6."],
        4: ["5, 8, 6."],
        5: ["ஏழு.", "5"],
        6: ["எட்டு."],
    }
    assert b.late == []
    assert bp.missing_segment_indices(trs) == [2, 3]


def test_late_final_of_previous_triplet_is_not_given_to_the_next() -> None:
    windows = [bp.Window(1, 0, 3000, 4000), bp.Window(2, 4000, 7000, 8000)]
    trs = [
        _tr(1, "5 3", 3500, 1),
        _tr(2, "1", 4300, 1),
        _tr(3, "", 7200, 2),
        _tr(4, "3 4 5", 7300, 2),
    ]
    b = bp.bind_windows(windows, trs)
    assert [t.text for t in b.groups[1]] == ["5 3"]
    assert [t.text for t in b.groups[2]] == ["3 4 5"]
    assert b.empty == {1: 0, 2: 1}
    assert b.late == [{"text": "1", "segment_index": 2, "bound_pid": 1, "window_pid": 2}]


class SplittingPrisma(FakePrisma):
    """Emits nothing on VAD; on the k-th commit sends the scripted segments of triplet k, each
    after its own delay (split answers, empty segments, finals delayed past later triplets)."""

    def __init__(self, plan: list[list[tuple[str, int]]], **kw: Any) -> None:
        super().__init__(**kw)
        self.plan = plan
        self.n_commits = 0
        self.seg = 0

    async def _on_frame(
        self, ws: ServerConnection, c: FakePrismaConnection, st: _UttState, frame: bytes
    ) -> None:
        c.frames.append((c.stream_ms, frame))

    async def _on_text(
        self, ws: ServerConnection, c: FakePrismaConnection, st: _UttState, raw: str
    ) -> None:
        if json.loads(raw).get("type") != "commit":
            return
        c.commits.append(c.stream_ms)
        k = self.n_commits
        self.n_commits += 1
        for text, delay in self.plan[k] if k < len(self.plan) else []:
            self._spawn(self._seg(ws, c, text, delay))

    async def _seg(self, ws: ServerConnection, c: FakePrismaConnection, text: str, d: int) -> None:
        with contextlib.suppress(ConnectionClosed):
            await self._wait_ms(d)
            self.seg += 1
            msg = {
                "type": "transcript",
                "text": text,
                "audio_duration_ms": 400,
                "segment_id": f"seg-{self.seg}",
                "segment_index": self.seg,
                "latency": d,  # Prisma reports its real finalisation latency
                "is_final": True,
            }
            await self._send(ws, c, msg)


def test_commit_bound_end_to_end_split_delayed_empty(tmp_path: Path) -> None:
    """Fake reproduces the live failure modes; the run maps, flags and scores them."""
    banks = bp.load_banks(Path("stimuli"), [Lang.TA])
    cell = bp.Cell(
        Lang.TA,
        4.0,
        tuple(bp.Item(i + 1, t, 0, 1000 + i) for i, t in enumerate(TA4[:4])),
    )
    plan = [
        [("5", 100), ("", 300), ("3 1", 1500)],  # split + empty, 2nd half after 1.5 s
        [("3", 200), ("4", 600), ("5", 900)],  # split into three
        [("7 4 6", 5000)],  # final delayed past the next triplet start -> late
        [("5, 8, 6.", 300)],
    ]

    async def go() -> dict[str, Any]:
        clock = ScaledClock(speed=20.0)
        kit = await open_ledger(tmp_path / "l.sqlite")
        fake = SplittingPrisma(plan, clock=clock)
        url = await fake.start()
        client = PrismaStreamClient(
            PrismaOptions(ws_url=url, connect_timeout_s=2.0, send_timeout_s=2.0),
            {KeyId.A: "KA"},
            RecordingBreakers(),
            bp.CappedLedger(kit.ledger, 500.0),
            clock,
            NullTelemetry(),
            purpose="bench",
            rng=random.Random(1),  # noqa: S311
        )
        try:
            return await bp.run_cell(
                client,
                banks[Lang.TA],
                load_lexicons(),
                cell,
                bp.SttParams(model_version=banks[Lang.TA].manifest.model_version),
                clock,
                commit_wait_s=4.0,
                quiet_s=1.0,
            )
        finally:
            await fake.stop()
            await kit.close()

    res = asyncio.run(go())
    assert res["mapping"] == "commit-bound"
    assert res["schema"] == 2
    it = res["items"]
    assert it[0]["transcripts"] == ["5", "3 1"]
    assert it[0]["empty_segments"] == 1
    assert it[0]["triplet_correct"] is True
    assert it[1]["transcripts"] == ["3", "4", "5"]
    assert it[1]["triplet_correct"] is True
    assert it[2]["no_response"] is True
    assert it[2]["triplet_correct"] is False
    assert it[3]["transcripts"] == ["5, 8, 6."]
    assert it[3]["triplet_correct"] is True
    assert res["segments_empty"] == 1
    assert [s["text"] for s in res["segments_late"]] == ["7 4 6"]


def test_resume_ignores_v1_cell_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "ta_+4.json").write_text(json.dumps({"cell": "ta:+4", "items": []}))
    assert not bp.is_current(out / "ta_+4.json")
    assert bp.load_cells(out) == []
    rc = bp.main(["--langs", "ta", "--snrs", "+4", "--n", "6", "--out", str(out)], rate=1.0)
    assert rc == 1  # the old file is not reused: the cell is planned again (and needs --yes)
    (out / "ta_+4.json").write_text(json.dumps({"mapping": "commit-bound", "items": []}))
    assert bp.is_current(out / "ta_+4.json")
