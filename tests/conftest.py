"""Shared pytest hooks: live-test gating and credit declarations."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sevi.core.stimulus import BankStimulusProvider

pytest_plugins = ["pytester"]

LIVE_ENV = "SEVI_ALLOW_LIVE"


def live_allowed() -> bool:
    return os.environ.get(LIVE_ENV) == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason=f"live test: set {LIVE_ENV}=1 to run")
    for item in items:
        if item.get_closest_marker("live") is None:
            continue
        credits = item.get_closest_marker("max_credits")
        if credits is None or not credits.args or not isinstance(credits.args[0], int | float):
            raise pytest.UsageError(
                f"{item.nodeid}: live tests must declare @pytest.mark.max_credits(<number>)"
            )
        if not live_allowed():
            item.add_marker(skip_live)


# --------------------------------------------------------------------------- A: synthetic test bank

TEST_BANK_DIR = Path(__file__).resolve().parents[1] / "stimuli" / "test-bank"
# Pinned bank_hash of stimuli/test-bank (A spec §3.4). Regenerate only in a reviewed PR:
#   uv run python -m sevi.calibration.testbank stimuli/test-bank
TEST_BANK_HASH = "2e2e98670db6df246ada022e0df5c378fd52c8fd90b0d1f02ed4f3d8a71938fb"


@pytest.fixture(scope="session")
def test_bank() -> BankStimulusProvider:
    """The committed test bank, loaded through the production path with its pinned hash."""
    from sevi.contracts.enums import Lang  # lazy: keep numpy/scipy out of B-only sessions
    from sevi.core.bank import load_bank

    return load_bank(Lang.TA, str(TEST_BANK_DIR), {Lang.TA: TEST_BANK_HASH}, allow_unfrozen=False)


@pytest.fixture(scope="session")
def test_bank_dir() -> Path:
    return TEST_BANK_DIR


@pytest.fixture(scope="session")
def test_bank_hash() -> str:
    return TEST_BANK_HASH
