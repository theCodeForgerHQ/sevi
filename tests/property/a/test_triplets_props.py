"""Property tests for core/triplets.py (A spec §2, P-TRP-01)."""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from sevi.contracts.constants import CAL_SUBSET_SIZE, OFFSET_PAIRS, POOL_SIZE
from sevi.contracts.stimulus import TripletDesign
from sevi.core.triplets import build_pool, calibration_subset, design

digit_sets = st.integers(8, 9).flatmap(
    lambda n: st.sets(st.integers(1, 9), min_size=n, max_size=n).map(lambda s: tuple(sorted(s)))
)


@settings(deadline=None)
@given(ds=digit_sets)
def test_P_TRP_01_balance_and_design(ds: tuple[int, ...]) -> None:
    n = len(ds)
    pool = build_pool(ds)
    sub = calibration_subset(pool, n)
    assert len(pool) == POOL_SIZE[n] == len(set(pool))
    assert len(sub) == CAL_SUBSET_SIZE[n]
    n_pairs_sub = CAL_SUBSET_SIZE[n] // n
    for pos in range(3):
        assert Counter(t[pos] for t in pool) == Counter({d: len(OFFSET_PAIRS[n]) for d in ds})
        assert Counter(t[pos] for t in sub) == Counter({d: n_pairs_sub for d in ds})
    for p in range(len(OFFSET_PAIRS[n])):
        block = pool[p * n : (p + 1) * n]
        for pos in range(3):
            assert sorted(t[pos] for t in block) == list(ds)
    d = design(ds)
    assert TripletDesign.model_validate(d.model_dump()) == d
    assert d.offset_pairs == OFFSET_PAIRS[n]
