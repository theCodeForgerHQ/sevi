"""Shared pytest hooks: live-test gating and credit declarations."""

from __future__ import annotations

import os

import pytest

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
