"""Adapters read the merged `Settings` (PR #26): option factories."""

from __future__ import annotations

from sevi.adapters.evon import EvonOptions
from sevi.adapters.prisma_stream import PrismaOptions
from sevi.adapters.twilio_rest import TwilioOptions
from sevi.app.settings import Settings


def test_options_from_settings_defaults() -> None:
    s = Settings()
    p = PrismaOptions.from_settings(s)
    assert p.ws_url == "wss://api.vachana.ai/stt/v3/stream"
    assert p.keyterms_headers == ("keyterms", "x-keyterms", "x-keywords")
    assert p.send_timeout_s == s.session.send_timeout_s
    e = EvonOptions.from_settings(s.evon)
    assert e.base_url.endswith("/v1")
    assert e.timeout_s == 6.0
    t = TwilioOptions.from_settings(s.twilio)
    assert t.time_limit_s == 600
