"""EXP-13 bench report (issue #61): latency definition and CAPTURE_MISS accounting. Fixtures
copy the structure of a real live cell file (var/bench-ms800/en_+4.json, schema 2)."""

from __future__ import annotations

import copy
from typing import Any

from sevi.calibration import bench_prisma as bp
from sevi.contracts.adapters import FrameTag, SttTranscript


def _item(index: int, *, ok: bool, server_ms: int, old_latency: float | None) -> dict[str, Any]:
    it: dict[str, Any] = {
        "bound_pid": [index],
        "digit_correct": [True, True, ok],
        "empty_segments": 0,
        "index": index,
        "latency_ms": old_latency,  # v2 commit-relative field: negative in the live run
        "no_response": False,
        "noise_seed": "8321022901613988126",
        "pool_index": 62,
        "position_correct": [True, True, ok],
        "presented": [8, 1, 5],
        "recognised": [8, 1, 5] if ok else [8, 1, 6],
        "segment_index": [index],
        "server_latency_ms": [server_ms],
        "status": "OK" if ok else "WRONG",
        "transcripts": ["815." if ok else "816."],
        "triplet_correct": ok,
    }
    return it


def _miss(index: int) -> dict[str, Any]:
    it = _item(index, ok=False, server_ms=0, old_latency=None)
    it.update(
        bound_pid=[],
        digit_correct=[False] * 3,
        position_correct=[False] * 3,
        recognised=[],
        segment_index=[],
        server_latency_ms=[],
        status="CAPTURE_MISS",
        transcripts=[],
        no_response=True,
    )
    return it


def _cell() -> dict[str, Any]:
    return {
        "schema": 2,
        "mapping": "commit-bound",
        "cell": "en:+4",
        "lang": "en",
        "snr_db": 4.0,
        "segments_empty": 0,
        "segments_late": [],
        "items": [
            _item(1, ok=True, server_ms=83, old_latency=-478.2),
            _item(2, ok=True, server_ms=99, old_latency=-605.4),
            _item(3, ok=False, server_ms=45, old_latency=-398.2),
            _miss(4),
        ],
    }


def test_old_cell_file_negative_latency_falls_back_to_server_latency() -> None:
    """The live v2 files carry commit-relative latency_ms < 0 (Prisma's VAD finalises in the
    trailing gap silence, before the commit). The report must never print those."""
    s = bp.summarize([_cell()])
    row = s["cells"][0]
    assert row["latency_source"] == "server"
    assert row["latency_p50_ms"] == 83.0
    assert row["latency_p95_ms"] > 0
    assert s["latency"]["en"]["n"] == 3
    md = bp.report_markdown(s)
    assert "-478" not in md
    assert "server_latency_ms" in md


def test_capture_miss_counted_apart_from_misrecognition() -> None:
    row = bp.summarize([_cell()])["cells"][0]
    assert (row["capture_miss"], row["misrecognised"]) == (1, 1)
    assert "| 1 | 1 |" in bp.report_markdown(bp.summarize([_cell()]))


def test_answer_latency_used_when_present() -> None:
    c = copy.deepcopy(_cell())
    for i, v in zip(c["items"], [900.0, 1100.0, 1000.0, None], strict=True):
        del i["latency_ms"]
        i["answer_latency_ms"] = v
    s = bp.summarize([c])
    assert s["latency_source"] == "answer"
    assert s["cells"][0]["latency_p50_ms"] == 1000.0
    assert "Answer latency" in bp.report_markdown(s)


def test_answer_latency_from_speech_end_is_positive_when_final_precedes_commit() -> None:
    """Final received at 1400 ms; speech ended at 1000 ms; commit (end of the 0.8 s gap) at
    1800 ms. Commit-relative would be -400; speech-end-relative is +400."""
    t = SttTranscript(
        text="815.",
        segment_id=None,
        segment_index=1,
        audio_duration_ms=900,
        latency_ms=83,
        detected_language_code="en-IN",
        received_mono_ms=1400.0,
        bound=FrameTag(presentation_id=1, window_index=1),
    )
    assert bp.answer_latency_ms([t], 1000.0) == 400.0
    assert bp.answer_latency_ms([], 1000.0) is None
