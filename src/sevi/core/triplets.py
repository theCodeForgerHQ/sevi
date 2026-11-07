"""Balanced digit-triplet design (A spec §2, contracts §4.3, D08).

``pool[p*n + a] = (ds[a], ds[(a+d1) % n], ds[(a+d2) % n])`` for each offset pair ``(d1, d2)``.
Each pair contributes every digit exactly once per position.
"""

from __future__ import annotations

import itertools
from typing import Literal, cast

from sevi.contracts.constants import CAL_SUBSET_SIZE, OFFSET_PAIRS, POOL_SIZE
from sevi.contracts.stimulus import TripletDesign

Triplet = tuple[int, int, int]


def _validate(digit_set: tuple[int, ...]) -> int:
    n = len(digit_set)
    if n not in (8, 9):
        raise ValueError(f"digit_set must have 8 or 9 digits, got {n}")
    if any(not isinstance(d, int) or isinstance(d, bool) for d in digit_set):
        raise ValueError("digit_set must contain ints")
    if any(d < 1 or d > 9 for d in digit_set):
        raise ValueError("digit_set values must be 1-9")
    if any(b <= a for a, b in itertools.pairwise(digit_set)):
        raise ValueError("digit_set must be strictly ascending")
    return n


def build_pool(digit_set: tuple[int, ...]) -> tuple[Triplet, ...]:
    n = _validate(digit_set)
    ds = digit_set
    pool = tuple(
        (ds[a], ds[(a + d1) % n], ds[(a + d2) % n]) for d1, d2 in OFFSET_PAIRS[n] for a in range(n)
    )
    if len(pool) != POOL_SIZE[n]:  # pragma: no cover - guarded by the constants
        raise ValueError("pool size mismatch")
    return pool


def calibration_subset(pool: tuple[Triplet, ...], n_digits: int) -> tuple[Triplet, ...]:
    if n_digits not in CAL_SUBSET_SIZE:
        raise ValueError(f"n_digits must be 8 or 9, got {n_digits}")
    if len(pool) != POOL_SIZE[n_digits]:
        raise ValueError(f"pool must have {POOL_SIZE[n_digits]} triplets for n={n_digits}")
    return pool[: CAL_SUBSET_SIZE[n_digits]]


def design(digit_set: tuple[int, ...]) -> TripletDesign:
    n = _validate(digit_set)
    return TripletDesign(
        n_digits=cast(Literal[8, 9], n),
        offset_pairs=OFFSET_PAIRS[n],
        pool_size=cast(Literal[120, 126], POOL_SIZE[n]),
        calibration_subset_size=cast(Literal[54, 56], CAL_SUBSET_SIZE[n]),
    )


__all__ = ["Triplet", "build_pool", "calibration_subset", "design"]
