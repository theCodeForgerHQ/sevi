"""U-UPL-01..10 (B spec §6.3): decode, gating of every played clip, FIFO, gap fill, tags."""

from __future__ import annotations

import numpy as np
from b_fakes import RecordingTelemetry

from sevi.contracts.adapters import FrameTag, TMedia
from sevi.core.capture import CaptureParams, CaptureTracker
from sevi.core.dsp import ulaw_decode, ulaw_encode
from sevi.transport.uplink import Uplink

LOUD = ulaw_encode(
    (8000 * np.sin(2 * np.pi * 440 * np.arange(160) / 8000)).astype(np.int16)
)  # 160 B, non-zero PCM
TAG = FrameTag(presentation_id=3, window_index=1)


def media(ts: int, cap: float | None = None, payload: bytes = LOUD) -> TMedia:
    return TMedia(
        ulaw=payload,
        seq=ts // 20 + 1,
        ts_ms=ts,
        received_mono_ms=1000.0 + ts,
        capture_mono_ms=(1000.0 + ts) if cap is None else cap,
    )


def make(
    gates: list[tuple[float, float]] | None = None, windows=None
) -> tuple[Uplink, RecordingTelemetry]:  # type: ignore[no-untyped-def]
    tel = RecordingTelemetry()
    gates = gates or []
    windows = windows or []

    def gated_at(t: float) -> bool:
        return any(a <= t <= b for a, b in gates)

    def tag_at(t: float) -> FrameTag | None:
        for a, b, tag in windows:
            if a <= t <= b:
                return tag  # type: ignore[no-any-return]
        return None

    return Uplink(ulaw_decode, tel, gated_at, tag_at), tel


def test_U_UPL_01_fifo_framing() -> None:
    up, _ = make()
    frames = []
    for i in range(16):
        caps, pf = up.on_media(media(20 * i))
        assert len(caps) == 1
        frames += pf
    assert len(frames) == 5
    assert all(len(f.pcm) == 1024 for f in frames)
    assert up.pending_bytes == 0
    up2, _ = make()
    for i in range(3):
        assert up2.on_media(media(20 * i))[1] == []
    assert up2.pending_bytes == 960


def test_U_UPL_02_gated_chunk_zero_to_prisma_real_energy_to_capture() -> None:
    """Speech captured while one of our clips plays is zeros to Prisma (never scored)."""
    up, _ = make(gates=[(0.0, 10_000.0)])
    frames = []
    caps_all = []
    for i in range(16):
        caps, pf = up.on_media(media(20 * i))
        caps_all += caps
        frames += pf
    assert all(f.pcm == bytes(1024) and f.gated_zero for f in frames)
    assert all(c.dbfs > -30 for c in caps_all)


def test_U_UPL_03_gap_fill_ge_20() -> None:
    up, tel = make()
    out = []
    for ts in (0, 20, 80):
        out += up.on_media(media(ts))[1]
    assert len(out) * 1024 + up.pending_bytes == (3 * 160 + 40 * 8) * 2
    tail = up.flush_tail()
    assert tail is not None
    pcm = np.frombuffer(b"".join(f.pcm for f in [*out, tail]), dtype="<i2")
    assert not pcm[320:640].any()
    assert pcm[640:800].any()
    assert ("uplink.gap", "INFO", {"gap_ms": 40}) in tel.events
    up2, _ = make()
    n = 0
    for ts in (0, 20, 60):
        n += len(up2.on_media(media(ts))[1])
    assert n * 1024 + up2.pending_bytes == (3 * 160 + 20 * 8) * 2


def test_U_UPL_04_gap_fill_capped() -> None:
    up, _ = make()
    up.on_media(media(20))
    _, pf = up.on_media(media(3000))
    total = len(pf) * 1024 + up.pending_bytes
    assert total == (160 + 2000 * 8 + 160) * 2


def test_U_UPL_05_dup_and_out_of_order_dropped() -> None:
    up, tel = make()
    up.on_media(media(20))
    assert up.on_media(media(20)) == ([], [])
    up.on_media(media(40))
    assert up.on_media(media(0)) == ([], [])
    assert [e for e in tel.events if e[2].get("reason") == "dup"].__len__() == 2
    assert up.pending_bytes == 2 * 320


def test_U_UPL_06_flush_tail() -> None:
    up, _ = make()
    assert up.flush_tail() is None
    for i in range(2):
        up.on_media(media(20 * i))
    up.on_media(media(40, payload=LOUD[:30]))  # 640 + 60 = 700 B pending
    assert up.pending_bytes == 700
    f = up.flush_tail()
    assert f is not None
    assert len(f.pcm) == 1024
    assert f.pcm.endswith(bytes(324))
    assert up.flush_tail() is None


def test_U_UPL_07_multi_chunk_capture_times() -> None:
    up, _ = make()
    caps, _ = up.on_media(media(0, cap=5000.0, payload=LOUD * 3))
    assert [c.mono_ms for c in caps] == [5000.0, 5020.0, 5040.0]


def test_U_UPL_08_burst_after_gate_end_uses_capture_time() -> None:
    """Frames arriving after gate_end but captured inside the gate are zeros and untagged."""
    up, _ = make(gates=[(1000.0, 2100.0)], windows=[(2100.1, 9999.0, TAG)])
    frames = []
    for i in range(50):  # captured 1000..1980, all arrive late
        m = TMedia(
            ulaw=LOUD, seq=i, ts_ms=20 * i, received_mono_ms=5000.0, capture_mono_ms=1000.0 + 20 * i
        )
        frames += up.on_media(m)[1]
    assert frames
    assert all(f.pcm == bytes(1024) and f.gated_zero and f.tag is None for f in frames)


def test_U_UPL_10_tag_at_middle_sample_and_flush_tail_tag() -> None:
    # window opens at capture 1032 ms; frame 0 covers 1000..1064, middle sample at 1032
    up, _ = make(windows=[(1032.0, 9999.0, TAG)])
    frames = []
    for i in range(4):
        frames += up.on_media(media(20 * i))[1]
    assert frames[0].tag == TAG
    assert not frames[0].gated_zero
    up2, _ = make(windows=[(1032.1, 9999.0, TAG)])
    f0 = []
    for i in range(4):
        f0 += up2.on_media(media(20 * i))[1]
    assert f0[0].tag is None
    tail = up2.flush_tail()
    assert tail is not None
    assert tail.tag == TAG


def test_gate_end_to_end_with_capture_tracker_reask_not_scored() -> None:
    """IT-GATE-02 (unit slice): speech during a played clip -> Prisma sees zeros (not scored)
    and CaptureTracker reports the early onset that drives REASK_SAY_AGAIN."""
    gate = (1200.0, 1800.0)
    up, _ = make(gates=[gate])
    tr = CaptureTracker(CaptureParams(12.0, 300, 10.0))
    silent = ulaw_encode(np.zeros(160, dtype=np.int16))
    sent = []
    for i in range(60):
        ts = 20 * i
        cap = 1000.0 + ts
        payload = LOUD if 1300 <= cap < 1700 else silent
        caps, pf = up.on_media(media(ts, cap=cap, payload=payload))
        for c in caps:
            tr.feed(c.dbfs, c.mono_ms, gated=gate[0] <= c.mono_ms <= gate[1], playing=False)
        sent += pf
    pcm = np.frombuffer(b"".join(f.pcm for f in sent), dtype="<i2")
    assert not pcm.any()  # loud audio only inside the gate -> nothing reaches Prisma
    assert tr.early_onset(1250.0, gate[1]) == 1300.0
