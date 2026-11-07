"""Capture analysis C12 (B spec §6.4, contracts §5.3, §6.1): rx energy, noise floor, speech
presence, early-answer onsets and echo coupling. Pure and deterministic given the fed sequence.

All times are capture times (`TMedia.capture_mono_ms` based), never arrival times, so a network
burst can neither fake nor hide a speech run, and frames captured while our clip played never
enter the floor even when they arrive after the clip's mark.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FRAME_MS: Final = 20.0
FLOOR_MIN_FRAMES: Final = 25
FLOOR_DEFAULT_DBFS: Final = -60.0
FLOOR_PERCENTILE: Final = 20.0
FLOOR_RING_FRAMES: Final = 3000  # last 60 s of non-playing frames
EARLY_RUN_FRAMES: Final = 3
ECHO_MIN_FRAMES: Final = 25
KEEP_FRAMES: Final = 45000  # 15 min
CONSECUTIVE_MAX_GAP_MS: Final = 30.0
SILENT_FRAME_DBFS: Final = -120.0


def frame_energy_dbfs(pcm: npt.NDArray[np.int16] | npt.NDArray[np.float64]) -> float:
    """20*log10(rms/32768) of int16-scaled samples; -120.0 for all-zero (or empty)."""
    x = np.asarray(pcm, dtype=np.float64)
    if x.size == 0:
        return SILENT_FRAME_DBFS
    r = float(np.sqrt(np.mean(x * x)))
    if r <= 0.0:
        return SILENT_FRAME_DBFS
    return max(SILENT_FRAME_DBFS, 20.0 * float(np.log10(r / 32768.0)))


@dataclass(frozen=True)
class CaptureParams:
    speech_db_above_floor: float
    speech_min_ms: int
    echo_high_db: float


@dataclass(frozen=True)
class SpeechStats:
    present: bool
    speech_ms: int
    first_mono_ms: float | None
    last_mono_ms: float | None


@dataclass(frozen=True)
class _Frame:
    mono_ms: float
    dbfs: float
    playing: bool
    gated: bool


class CaptureTracker:
    def __init__(self, params: CaptureParams) -> None:
        self._p = params
        self._frames: deque[_Frame] = deque(maxlen=KEEP_FRAMES)
        self._ring: deque[float] = deque(maxlen=FLOOR_RING_FRAMES)
        self._floor: float | None = None
        self._new_ring = 0
        self._echo: list[tuple[float, float]] = []

    def feed(self, frame_dbfs: float, mono_ms: float, gated: bool, playing: bool) -> None:
        self._frames.append(_Frame(mono_ms, frame_dbfs, playing, gated))
        if not playing:
            self._ring.append(frame_dbfs)
            self._new_ring += 1

    def floor_dbfs(self) -> float:
        if len(self._ring) < FLOOR_MIN_FRAMES:
            return FLOOR_DEFAULT_DBFS
        if self._floor is None or self._new_ring >= FLOOR_MIN_FRAMES:
            arr = np.fromiter(self._ring, dtype=np.float64)
            self._floor = float(np.percentile(arr, FLOOR_PERCENTILE, method="linear"))
            self._new_ring = 0
        return self._floor

    def is_speech(self, frame_dbfs: float) -> bool:
        return frame_dbfs >= self.floor_dbfs() + self._p.speech_db_above_floor

    def speech_in(self, window: tuple[float, float]) -> SpeechStats:
        a, b = window
        thr = self.floor_dbfs() + self._p.speech_db_above_floor
        times = [f.mono_ms for f in self._frames if a <= f.mono_ms <= b and f.dbfs >= thr]
        speech_ms = int(FRAME_MS) * len(times)
        return SpeechStats(
            present=speech_ms >= self._p.speech_min_ms,
            speech_ms=speech_ms,
            first_mono_ms=times[0] if times else None,
            last_mono_ms=times[-1] if times else None,
        )

    def any_speech_after(self, t_ms: float, until_ms: float) -> bool:
        thr = self.floor_dbfs() + self._p.speech_db_above_floor
        return any(t_ms < f.mono_ms <= until_ms and f.dbfs >= thr for f in self._frames)

    def early_onset(self, start_ms: float, end_ms: float) -> float | None:
        """Capture time of the first frame of the first run of >= 3 consecutive speech frames
        (adjacent capture times <= 30 ms apart) whose first frame lies in [start_ms, end_ms)."""
        thr = self.floor_dbfs() + self._p.speech_db_above_floor
        run_start: float | None = None
        run_len = 0
        prev_t: float | None = None
        for f in sorted(self._frames, key=lambda fr: fr.mono_ms):
            if f.dbfs >= thr:
                if run_len and prev_t is not None and f.mono_ms - prev_t <= CONSECUTIVE_MAX_GAP_MS:
                    run_len += 1
                else:
                    run_start, run_len = f.mono_ms, 1
                prev_t = f.mono_ms
                if (
                    run_len == EARLY_RUN_FRAMES
                    and run_start is not None
                    and start_ms <= run_start < end_ms
                ):
                    return run_start
            else:
                run_len, run_start, prev_t = 0, None, None
        return None

    def add_echo_interval(self, start_ms: float, end_ms: float) -> None:
        self._echo.append((start_ms, end_ms))

    def echo_coupling_db(self) -> float | None:
        """median(dbfs of frames in the union of echo intervals) - floor; None if < 25 frames."""
        vals = [f.dbfs for f in self._frames if any(a <= f.mono_ms <= b for a, b in self._echo)]
        if len(vals) < ECHO_MIN_FRAMES:
            return None
        return float(np.median(np.asarray(vals, dtype=np.float64))) - self.floor_dbfs()

    def echo_mode(self) -> bool:
        c = self.echo_coupling_db()
        return c is not None and c >= self._p.echo_high_db
