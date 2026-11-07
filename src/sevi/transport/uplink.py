"""Browser uplink: mu-law decode, capture frames, gating and 1,024-byte Prisma framing
(B spec §6.3, contracts §6.1 "Gate", §4.3 audio chain).

Every decision uses capture time (`TMedia.capture_mono_ms`), never arrival time:
- `gated_at(t)`: `t` lies in a gate interval. Every clip we play opens one, so speech captured
  while a clip plays reaches Prisma as zeros (capture still sees the real audio).
- `tag_at(t)`: the `FrameTag` of the answer window open at `t` (else None). Transcripts bind to a
  trial by this tag, never by arrival time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from sevi.contracts.adapters import FrameTag, Telemetry, TMedia
from sevi.contracts.constants import PRISMA_FRAME_BYTES
from sevi.contracts.telemetry import EventName
from sevi.core.capture import frame_energy_dbfs

SAMPLES_PER_MS: Final = 8  # 8 kHz
CHUNK_SAMPLES: Final = 160  # 20 ms
CHUNK_MS: Final = 20.0
FRAME_SAMPLES: Final = PRISMA_FRAME_BYTES // 2  # 512 s16le samples = 64 ms
GAP_FILL_MIN_MS: Final = 20
GAP_FILL_MAX_MS: Final = 2000


@dataclass(frozen=True)
class CaptureFrame:
    dbfs: float
    mono_ms: float  # capture time of the chunk's first sample


@dataclass(frozen=True)
class PrismaFrame:
    pcm: bytes  # PRISMA_FRAME_BYTES of s16le
    tag: FrameTag | None
    gated_zero: bool  # every real sample came from gated chunks or gated fill


class Uplink:
    def __init__(
        self,
        decode: Callable[[bytes], npt.NDArray[np.int16]],
        telemetry: Telemetry,
        gated_at: Callable[[float], bool],
        tag_at: Callable[[float], FrameTag | None],
    ) -> None:
        self._decode = decode
        self._tel = telemetry
        self._gated_at = gated_at
        self._tag_at = tag_at
        self._prev_ts: int | None = None
        self._prev_len_ms = 0.0
        self._pcm: list[npt.NDArray[np.int16]] = []
        self._t: list[npt.NDArray[np.float64]] = []  # capture time per sample
        self._gated: list[npt.NDArray[np.bool_]] = []
        self._n = 0

    @property
    def pending_bytes(self) -> int:
        return self._n * 2

    def _append(self, pcm: npt.NDArray[np.int16], t0: float, gated: bool) -> None:
        n = len(pcm)
        self._pcm.append(pcm)
        self._t.append(t0 + np.arange(n, dtype=np.float64) / SAMPLES_PER_MS)
        self._gated.append(np.full(n, gated, dtype=np.bool_))
        self._n += n

    def _pop_frames(self) -> list[PrismaFrame]:
        if self._n < FRAME_SAMPLES:
            return []
        pcm = np.concatenate(self._pcm)
        t = np.concatenate(self._t)
        g = np.concatenate(self._gated)
        out: list[PrismaFrame] = []
        k = 0
        while self._n - k >= FRAME_SAMPLES:
            sl = slice(k, k + FRAME_SAMPLES)
            out.append(
                PrismaFrame(
                    pcm=pcm[sl].astype("<i2").tobytes(),
                    tag=self._tag_at(float(t[k + FRAME_SAMPLES // 2])),
                    gated_zero=bool(g[sl].all()),
                )
            )
            k += FRAME_SAMPLES
        self._pcm, self._t, self._gated = [pcm[k:]], [t[k:]], [g[k:]]
        self._n -= k
        return out

    def on_media(self, m: TMedia) -> tuple[list[CaptureFrame], list[PrismaFrame]]:
        cap0 = m.capture_mono_ms if m.capture_mono_ms is not None else m.received_mono_ms
        len_ms = len(m.ulaw) / SAMPLES_PER_MS
        if m.ts_ms is not None:
            if self._prev_ts is not None:
                if m.ts_ms <= self._prev_ts:
                    self._tel.emit(EventName.UPLINK_GAP, "DEBUG", reason="dup", ts_ms=m.ts_ms)
                    return [], []
                gap = m.ts_ms - (self._prev_ts + self._prev_len_ms)
                if gap >= GAP_FILL_MIN_MS:
                    fill_ms = min(gap, GAP_FILL_MAX_MS)
                    self._fill(cap0 - fill_ms, fill_ms)
                    self._tel.emit(EventName.UPLINK_GAP, "INFO", gap_ms=int(gap))
            self._prev_ts = m.ts_ms
            self._prev_len_ms = len_ms

        pcm = self._decode(m.ulaw)
        captures: list[CaptureFrame] = []
        for i in range(0, len(pcm), CHUNK_SAMPLES):
            chunk = pcm[i : i + CHUNK_SAMPLES]
            t_i = cap0 + CHUNK_MS * (i // CHUNK_SAMPLES)
            captures.append(CaptureFrame(frame_energy_dbfs(chunk), t_i))
            gated = self._gated_at(t_i)
            self._append(np.zeros_like(chunk) if gated else chunk, t_i, gated)
        return captures, self._pop_frames()

    def _fill(self, t0: float, fill_ms: float) -> None:
        """Zero fill (Prisma FIFO only); each 20 ms carries the gate state of its capture time."""
        total = round(fill_ms * SAMPLES_PER_MS)
        for i in range(0, total, CHUNK_SAMPLES):
            n = min(CHUNK_SAMPLES, total - i)
            t_i = t0 + i / SAMPLES_PER_MS
            self._append(np.zeros(n, dtype=np.int16), t_i, self._gated_at(t_i))

    def flush_tail(self) -> PrismaFrame | None:
        """Zero-pad the partial frame to 1,024 bytes. Tag = tag at the last real chunk's
        capture time (U-UPL-10). None if the FIFO is empty."""
        if self._n == 0:
            return None
        pcm = np.concatenate(self._pcm)
        t = np.concatenate(self._t)
        g = np.concatenate(self._gated)
        padded = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        padded[: len(pcm)] = pcm
        frame = PrismaFrame(
            pcm=padded.astype("<i2").tobytes(),
            tag=self._tag_at(float(t[-1])),
            gated_zero=bool(g.all()),
        )
        self.drop_pending()
        return frame

    def drop_pending(self) -> None:
        """Reconnect / mode switch: discard the partial frame (ordering state is kept)."""
        self._pcm, self._t, self._gated = [], [], []
        self._n = 0
