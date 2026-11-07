"""Unit tests for core/triplets.py (A spec §2, U-TRP-01..03)."""

from __future__ import annotations

from collections import Counter

import pytest

from sevi.contracts.constants import CAL_SUBSET_SIZE, OFFSET_PAIRS
from sevi.contracts.stimulus import TripletDesign
from sevi.core.triplets import build_pool, calibration_subset, design

TA = (1, 2, 3, 4, 5, 6, 7, 8)
EN = (1, 2, 3, 4, 5, 6, 8, 9)
NINE = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _per_position(triplets: tuple[tuple[int, int, int], ...]) -> list[Counter[int]]:
    return [Counter(t[pos] for t in triplets) for pos in range(3)]


@pytest.mark.parametrize(
    ("ds", "pool_n", "sub_n", "per_pos_pool", "per_pos_sub"),
    [(TA, 120, 56, 15, 7), (EN, 120, 56, 15, 7), (NINE, 126, 54, 14, 6)],
    ids=["ta", "en", "nine"],
)
def test_U_TRP_01_sizes_distinct_and_balanced(
    ds: tuple[int, ...], pool_n: int, sub_n: int, per_pos_pool: int, per_pos_sub: int
) -> None:
    pool = build_pool(ds)
    sub = calibration_subset(pool, len(ds))
    assert len(pool) == pool_n
    assert len(sub) == sub_n
    assert len(set(pool)) == pool_n
    assert all(len(set(t)) == 3 for t in pool)
    for counts in _per_position(pool):
        assert counts == Counter({d: per_pos_pool for d in ds})
    for counts in _per_position(sub):
        assert counts == Counter({d: per_pos_sub for d in ds})
    total = Counter(d for t in pool for d in t)
    assert set(total.values()) == {3 * per_pos_pool}


@pytest.mark.parametrize("ds", [TA, EN, NINE], ids=["ta", "en", "nine"])
def test_U_TRP_01_each_pair_contributes_each_digit_once_per_position(
    ds: tuple[int, ...],
) -> None:
    n = len(ds)
    pool = build_pool(ds)
    for p in range(len(OFFSET_PAIRS[n])):
        block = pool[p * n : (p + 1) * n]
        for counts in _per_position(block):
            assert counts == Counter({d: 1 for d in ds})


def test_U_TRP_02_ta_index_values() -> None:
    pool = build_pool(TA)
    assert pool[0] == (1, 2, 3)
    assert pool[7] == (8, 1, 2)
    assert pool[8] == (1, 3, 5)
    assert pool[56] == (1, 3, 7)
    assert pool[119] == (8, 5, 1)


def test_U_TRP_02_en_index_values() -> None:
    pool = build_pool(EN)
    assert pool[3] == (4, 5, 6)
    assert pool[7] == (9, 1, 2)
    assert pool[19] == (4, 8, 2)  # video hook triplet
    assert pool[56] == (1, 3, 8)
    assert pool[119] == (9, 5, 1)


@pytest.mark.parametrize(
    "bad",
    [
        (2, 1, 3, 4, 5, 6, 7, 8),
        (1, 1, 2, 3, 4, 5, 6, 7),
        (1, 2, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 4, 5, 6, 7),
        (1, 2, 3, 4, 5, 6, 7, 10),
        (),
    ],
    ids=["unsorted", "duplicate", "seven", "zero", "ten", "empty"],
)
def test_U_TRP_03_invalid_digit_sets(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"digit_set"):
        build_pool(bad)
    with pytest.raises(ValueError, match=r"digit_set"):
        design(bad)


def test_U_TRP_03_calibration_subset_validates() -> None:
    with pytest.raises(ValueError, match=r"pool must have"):
        calibration_subset(build_pool(TA), 9)
    with pytest.raises(ValueError, match=r"n_digits"):
        calibration_subset(build_pool(TA), 7)


def test_design_block_for_ta() -> None:
    d = design(TA)
    assert isinstance(d, TripletDesign)
    assert d.n_digits == 8
    assert d.offset_pairs == OFFSET_PAIRS[8]
    assert d.pool_size == 120
    assert d.calibration_subset_size == CAL_SUBSET_SIZE[8]
    assert d.model_dump(mode="json")["offset_pairs"][:2] == [[1, 2], [2, 4]]
