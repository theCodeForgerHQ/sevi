"""``BankStimulusProvider``: renders one trial from a loaded bank (A spec §5, contracts §5.1).

Built only by :func:`sevi.core.bank.load_bank`. Immutable after construction (thread-safe): the
digit, noise and floor arrays are made read-only. Only the SHIPPED variant exists (BUILD-BRIEF
reduced scope: the human panel and ``render_panel_trial`` are cut).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np

from sevi.contracts.constants import MIX_RMS_DBFS, PEAK_MAX_DBFS
from sevi.contracts.enums import Condition
from sevi.contracts.stimulus import BankManifest, DigitSpan, RenderedTrial, TimingMap
from sevi.contracts.trial import PresentationSpec
from sevi.core import dsp
from sevi.core.dsp import F64

NOISE_SAMPLES: Final = dsp.FLOOR_BUFFER_SAMPLES  # 960000 = 60 s at 16 kHz
SAMPLES_PER_MS: Final = 16  # 16 kHz mix timeline; /16 is exact in binary floating point


@dataclass(frozen=True, slots=True)
class PreChain:
    """Everything ``render_trial`` computes before the phone chain (A spec §5.2 steps 2-6)."""

    y: F64  # 16 kHz float after the condition and the step-6 limiter
    start: int  # noise_start_sample
    spans: tuple[tuple[int, int], ...]  # digit (onset, offset) samples at 16 kHz
    lim1: bool  # step-5 limiter acted
    lim2: bool  # step-6 limiter acted


def _span(position: Literal[1, 2, 3], digit: int, span: tuple[float, float]) -> DigitSpan:
    return DigitSpan(position=position, digit=digit, onset_ms=span[0], offset_ms=span[1])


def _readonly(x: F64) -> F64:
    y = np.array(x, dtype=np.float64, copy=True)
    y.setflags(write=False)
    return y


class BankStimulusProvider:
    """One language's SHIPPED bank. Implements ``contracts.ports.StimulusProvider``."""

    __slots__ = ("_digits", "_floors", "_manifest", "_noise", "_offset_db", "_ref_rms")

    def __init__(self, manifest: BankManifest, digits: Mapping[int, F64], noise: F64) -> None:
        if sorted(digits) != sorted(manifest.digit_set):
            raise ValueError("digits must cover manifest.digit_set exactly")
        if noise.size != NOISE_SAMPLES:
            raise ValueError(f"noise must have {NOISE_SAMPLES} samples")
        self._manifest = manifest
        self._digits: Mapping[int, F64] = MappingProxyType(
            {d: _readonly(digits[d]) for d in manifest.digit_set}
        )
        self._offset_db: Mapping[int, float] = MappingProxyType(
            {a.digit: a.offset_db for a in manifest.digits}
        )
        self._noise = _readonly(noise)
        mean_dbfs = float(np.mean([a.rms_dbfs for a in manifest.digits]))
        self._ref_rms = float(10.0 ** (mean_dbfs / 20.0))
        self._floors: Mapping[Condition, F64] = MappingProxyType(
            {
                c: _readonly(dsp.floor_buffer(manifest.lang.value, c))
                for c in (Condition.N2, Condition.N3)
            }
        )

    @property
    def manifest(self) -> BankManifest:
        return self._manifest

    @property
    def ref_rms(self) -> float:
        return self._ref_rms

    def floor(self, condition: Literal[Condition.N2, Condition.N3]) -> F64:
        """The read-only 60 s floor buffer of an N2/N3 condition."""
        return self._floors[condition]

    def prechain(self, spec: PresentationSpec) -> PreChain:
        """A spec §5.2 steps 1-6: the 16 kHz float signal that enters the phone chain."""
        triplet = spec.triplet
        if any(t not in self._digits for t in triplet):  # step 1
            raise ValueError(f"triplet {triplet} has a digit outside {self._manifest.digit_set}")
        speech, spans = dsp.assemble_triplet(  # step 2
            [self._digits[t] for t in triplet], [self._offset_db[t] for t in triplet]
        )
        n = speech.size
        if n > NOISE_SAMPLES:
            raise ValueError("clip longer than the noise buffer")
        gen = np.random.Generator(np.random.PCG64(spec.noise_seed))  # step 3 (R-11)
        start = int(gen.integers(0, NOISE_SAMPLES - n + 1))
        seg = self._noise[start : start + n]
        mix, _, _ = dsp.mix_at_snr(speech, seg, spec.snr_db, self._ref_rms)  # step 4
        mix = dsp.scale_to_rms_dbfs(mix, MIX_RMS_DBFS)  # step 5
        mix, lim1 = dsp.peak_limit(mix, PEAK_MAX_DBFS)
        cond = spec.condition  # step 6 [A-D17]
        floor_seg = (
            self._floors[cond][start : start + n] if cond in (Condition.N2, Condition.N3) else None
        )
        y = dsp.apply_condition(mix, cond, floor_seg)
        y, lim2 = dsp.peak_limit(y, PEAK_MAX_DBFS)
        return PreChain(y=y, start=start, spans=spans, lim1=lim1, lim2=lim2)

    def render_trial(self, spec: PresentationSpec) -> RenderedTrial:
        pre = self.prechain(spec)
        n = pre.y.size
        ulaw, _ = dsp.phone_chain(pre.y)  # step 7
        if len(ulaw) != n // 2:  # pragma: no cover - guaranteed by resample_poly(1, 2), n even
            raise ValueError("phone chain length mismatch")
        spans_ms = [(s / SAMPLES_PER_MS, e / SAMPLES_PER_MS) for s, e in pre.spans]
        timing = TimingMap(  # step 8
            clip_ms=n / SAMPLES_PER_MS,
            noise_off_ms=n / SAMPLES_PER_MS,
            digits=(
                _span(1, spec.triplet[0], spans_ms[0]),
                _span(2, spec.triplet[1], spans_ms[1]),
                _span(3, spec.triplet[2], spans_ms[2]),
            ),
            last_digit_offset_ms=spans_ms[2][1],
        )
        return RenderedTrial(  # step 9
            presentation_id=spec.presentation_id,
            lang=self._manifest.lang,
            triplet=spec.triplet,
            snr_db=spec.snr_db,
            condition=spec.condition,
            noise_seed=spec.noise_seed,
            noise_start_sample=pre.start,
            bank_hash=self._manifest.bank_hash,
            ulaw=ulaw,
            n_bytes=n // 2,
            timing=timing,
            level_limited=pre.lim1 or pre.lim2,
        )


__all__ = ["NOISE_SAMPLES", "BankStimulusProvider", "PreChain"]
