"""U-CAP-01..07: core/capture.py (B spec §6.4)."""

from __future__ import annotations

import numpy as np

from sevi.core.capture import CaptureParams, CaptureTracker, frame_energy_dbfs

P = CaptureParams(speech_db_above_floor=12.0, speech_min_ms=300, echo_high_db=10.0)


def _quiet(tr: CaptureTracker, n: int, t0: float = 0.0, db: float = -60.0) -> float:
    for i in range(n):
        tr.feed(db, t0 + 20 * i, gated=False, playing=False)
    return t0 + 20 * n


def test_U_CAP_01_frame_energy() -> None:
    assert frame_energy_dbfs(np.zeros(160, dtype=np.int16)) == -120.0
    sq = np.where(np.arange(160) % 2 == 0, 32768.0, -32768.0)
    assert abs(frame_energy_dbfs(sq) - 0.0) <= 0.01
    amp = 32768.0 * 10 ** (-20 / 20) * np.sqrt(2)
    sine = amp * np.sin(2 * np.pi * 1000 * np.arange(160) / 8000)
    assert abs(frame_energy_dbfs(sine) - (-20.0)) <= 0.05


def test_U_CAP_02_floor() -> None:
    tr = CaptureTracker(P)
    _quiet(tr, 24, db=-30.0)
    assert tr.floor_dbfs() == -60.0
    tr2 = CaptureTracker(P)
    for i in range(100):
        tr2.feed(-50.0 if i % 2 else -70.0, 20 * i, gated=False, playing=False)
    for i in range(100, 200):  # playing frames never enter the ring
        tr2.feed(0.0, 20 * i, gated=True, playing=True)
    assert tr2.floor_dbfs() == -70.0


def test_U_CAP_03_speech_present_boundary() -> None:
    tr = CaptureTracker(P)
    for k in (14, 15):
        t = 10_000.0 * k
        for i in range(k):
            tr.feed(-48.0, t + 20 * i, gated=False, playing=True)  # floor stays -60 (default)
        st = tr.speech_in((t, t + 20 * k))
        assert st.speech_ms == 20 * k
        assert st.present is (k == 15)
        assert st.first_mono_ms == t
    assert not tr.is_speech(-48.1)


def test_U_CAP_04_early_onset() -> None:
    tr = CaptureTracker(P)
    t = _quiet(tr, 30)
    gate_end = t + 200
    tr.feed(-30.0, t, False, False)
    tr.feed(-30.0, t + 20, False, False)  # run of 2
    _quiet(tr, 3, t0=t + 40)
    assert tr.early_onset(t - 1000, gate_end) is None
    s = gate_end - 10  # run of 3 starting 10 ms before gate end
    for i in range(3):
        tr.feed(-30.0, s + 20 * i, False, False)
    assert tr.early_onset(t + 100, gate_end) == s
    # the window starts after the run's first frame -> not counted
    assert tr.early_onset(s + 1, gate_end + 100) is None


def test_U_CAP_04_burst_cannot_fake_run() -> None:
    """Consecutive = capture times <= 30 ms apart, so frames 100 ms apart are no run."""
    tr = CaptureTracker(P)
    t = _quiet(tr, 30)
    for i in range(3):
        tr.feed(-30.0, t + 100 * i, False, False)
    assert tr.early_onset(t - 1, t + 1000) is None


def test_U_CAP_05_echo_union() -> None:
    tr = CaptureTracker(P)
    _quiet(tr, 50, db=-55.0)  # floor -55
    tr.add_echo_interval(5_000, 5_380)  # 20 frames
    tr.add_echo_interval(8_000, 8_380)  # 20 frames
    for i in range(20):
        tr.feed(-35.0, 5_000 + 20 * i, True, True)
        tr.feed(-35.0, 8_000 + 20 * i, True, True)
    assert tr.echo_coupling_db() == 20.0
    assert tr.echo_mode()
    tr2 = CaptureTracker(P)
    _quiet(tr2, 50, db=-55.0)
    tr2.add_echo_interval(5_000, 5_230)  # 12 frames
    tr2.add_echo_interval(8_000, 8_230)  # 12 frames -> 24 total
    for i in range(12):
        tr2.feed(-35.0, 5_000 + 20 * i, True, True)
        tr2.feed(-35.0, 8_000 + 20 * i, True, True)
    assert tr2.echo_coupling_db() is None
    assert not tr2.echo_mode()


def test_U_CAP_06_any_speech_after_strict() -> None:
    tr = CaptureTracker(P)
    _quiet(tr, 30)
    tr.feed(-30.0, 1000.0, False, False)
    assert not tr.any_speech_after(1000.0, 2000.0)
    assert tr.any_speech_after(999.0, 1000.0)


def test_U_CAP_07_playing_by_capture_time_excluded_from_floor() -> None:
    """Loud frames captured during our clip (playing=True) never move the floor, whatever
    order they arrive in."""
    tr = CaptureTracker(P)
    _quiet(tr, 40, db=-62.0)
    for i in range(200):
        tr.feed(-10.0, 50_000 + 20 * i, gated=True, playing=True)
    assert tr.floor_dbfs() == -62.0
