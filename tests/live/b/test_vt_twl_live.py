"""Live Twilio trial checks (contracts §8.7 VT-TWL-01/05). NEVER run by CI or builder agents:
``@pytest.mark.live`` is skipped unless SEVI_ALLOW_LIVE=1, and the Twilio credentials come only
from ``infisical run`` (SEVI_TWILIO__*). No Gnani credits are spent (``max_credits(0)``); each
call uses at most one trial minute of the trial balance.

VT-TWL-01 needs the app running behind the public ingress with ``SEVI_PHONE__ENABLED=true``:
it dials the first verified number with the real IVR webhook URL and waits for the call to end.
"""

from __future__ import annotations

import asyncio
import os
import secrets

import pytest

from sevi.adapters.twilio_rest import TwilioOptions, TwilioRest
from sevi.app.settings import Settings
from sevi.fakes.clock import ScaledClock
from sevi.fakes.null_telemetry import RecordingTelemetry


def _settings() -> Settings:
    if not os.environ.get("SEVI_TWILIO__AUTH_TOKEN"):
        pytest.skip("SEVI_TWILIO__* not set (run under infisical)")
    s = Settings()
    if not s.twilio.verified_numbers():
        pytest.skip("SEVI_TWILIO__VERIFIED_TO_NUMBERS is empty")
    return s


@pytest.mark.live
@pytest.mark.max_credits(0)
@pytest.mark.asyncio
async def test_VT_TWL_01_trial_can_ring_a_verified_number() -> None:
    """One outbound call to the first verified caller ID; it must ring (Twilio accepts it)."""
    s = _settings()
    tw = TwilioRest(TwilioOptions.from_settings(s.twilio), ScaledClock(1.0), RecordingTelemetry())
    to = next(iter(s.twilio.verified_numbers().values()))
    base = str(s.public_base_url).rstrip("/")
    sid = await tw.create_call(
        to_e164=to,
        session_id="vt-twl-01",
        base=base,
        t=secrets.token_urlsafe(16),
    )
    assert sid.startswith("CA")
    await asyncio.sleep(20)
    await tw.update_call(sid, "completed")


@pytest.mark.live
@pytest.mark.max_credits(0)
@pytest.mark.asyncio
async def test_VT_TWL_05_delete_completed_call() -> None:
    """A completed call can be deleted on the trial (2xx or 404 = OK)."""
    s = _settings()
    sid = os.environ.get("SEVI_VT_CALL_SID", "")
    if not sid:
        pytest.skip("set SEVI_VT_CALL_SID to a completed call SID")
    tw = TwilioRest(TwilioOptions.from_settings(s.twilio), ScaledClock(1.0), RecordingTelemetry())
    await tw.delete_call(sid)
