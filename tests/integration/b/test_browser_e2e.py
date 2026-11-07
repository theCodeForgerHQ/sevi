"""Real-browser E2E (scripts/browser_e2e.py) against the app in demo mode (fakes, test bank).

Chromium (Playwright) drives the shipped web client: landing -> test -> 24 scored trials ->
result -> report. Voice mode injects the bank's digit recordings through an overridden
`getUserMedia`; the demo fake Prisma "hears" the presented digits once it detects voiced
frames, so the full capture worklet -> uplink -> fake Prisma path is exercised offline.
No keys, no network beyond 127.0.0.1. Skipped when Chromium is not installed
(`uv run playwright install chromium`). Run with `make browser-e2e` (about 2-3 min per mode).
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests" / "integration" / "a"))
sys.path.insert(0, str(REPO / "scripts"))

pytestmark = pytest.mark.e2e


def _chromium_missing() -> str | None:
    if importlib.util.find_spec("playwright") is None:
        return "playwright not installed"
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            pw.chromium.launch(headless=True).close()
    except Exception as e:  # browser binary missing
        return f"chromium not available: {str(e).splitlines()[0]}"
    return None


@pytest.fixture(scope="module")
def chromium() -> None:
    why = _chromium_missing()
    if why:
        pytest.skip(f"{why} (uv run playwright install chromium)")


@contextmanager
def demo_server(tmp: Path) -> Iterator[str]:
    """`make demo` in-process on an ephemeral 127.0.0.1 port, real time (browser audio plays
    in real time, so a scaled clock would close voice windows early)."""
    import uvicorn
    from e2e_browser import demo_settings

    from sevi.app.main import create_app
    from sevi.fakes.clock import ScaledClock

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    app = create_app(
        demo_settings(tmp, pages_origin=origin, public_base_url=origin), clock=ScaledClock(1.0)
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, lifespan="on")
    )
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        for _ in range(1500):
            if server.started:
                break
            if not th.is_alive():
                raise RuntimeError("uvicorn failed to start")
            threading.Event().wait(0.02)
        else:
            raise RuntimeError("uvicorn did not start")
        yield f"{origin}/sevi/"
    finally:
        server.should_exit = True
        th.join(timeout=15)


def _run(base: str, tmp: Path, *extra: str) -> dict[str, Any]:
    import browser_e2e

    args = browser_e2e.build_parser().parse_args(
        ["--base-url", base, "--bank-dir", str(REPO / "stimuli" / "test-bank"), *extra]
    )
    out = browser_e2e.run(args)
    assert out.ok, out.failures
    return out.summary


def test_browser_e2e_keypad(chromium: None, tmp_path: Path) -> None:
    """Keypad: result band + report link; every scored trial answered on the page keypad."""
    with demo_server(tmp_path) as base:
        s = _run(base, tmp_path, "--mode", "keypad")
    assert s["scored_answered"] == 24
    assert s["scored_by_voice"] == 0
    assert s["result"]["band"]
    assert s["report"]["rows"] == 24
    assert all(f["correct"] for f in s["practice_feedback"])  # the triplet plan matches


def test_browser_e2e_voice_injected_audio(chromium: None, tmp_path: Path) -> None:
    """Voice: the injected stream reaches the uplink and >= 20 of 24 trials score by voice."""
    with demo_server(tmp_path) as base:
        s = _run(base, tmp_path, "--mode", "voice")
    assert s["media_frames_out"] > 0
    assert s["scored_by_voice"] >= 20
    assert not s["voice_fallback"]
    assert s["result"]["band"]


def test_browser_e2e_staircase_wrong_after(chromium: None, tmp_path: Path) -> None:
    """Keypad, wrong from scored trial 6 on: the staircase climbs, the SRT is not at floor."""
    with demo_server(tmp_path) as base:
        s = _run(base, tmp_path, "--mode", "keypad", "--answers", "wrong-after-6")
    assert s["result"]["band"]
    assert s["result"]["srt_display"] != "\u221220.0"  # floor; U+2212 minus
