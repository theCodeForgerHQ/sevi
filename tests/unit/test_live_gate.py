"""The live-test gate in tests/conftest.py: skip by default, demand max_credits."""

from pathlib import Path

import pytest

CONFTEST = (Path(__file__).parents[1] / "conftest.py").read_text()
INI = (
    "[pytest]\nasyncio_default_fixture_loop_scope = function\n"
    "markers =\n    live: x\n    max_credits(n): x\n"
)


def _setup(pytester: pytest.Pytester, body: str) -> None:
    pytester.makeconftest(CONFTEST)
    pytester.makeini(INI)
    pytester.makepyfile(body)


def test_live_without_max_credits_is_an_error(pytester: pytest.Pytester) -> None:
    _setup(pytester, "import pytest\n@pytest.mark.live\ndef test_x():\n    pass\n")
    result = pytester.runpytest()
    assert "must declare" in result.stdout.str() + result.stderr.str()
    assert result.ret != 0


def test_live_skipped_without_env(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SEVI_ALLOW_LIVE", raising=False)
    _setup(
        pytester,
        "import pytest\n@pytest.mark.live\n@pytest.mark.max_credits(5)\n"
        "def test_x():\n    raise AssertionError\n",
    )
    pytester.runpytest().assert_outcomes(skipped=1)
