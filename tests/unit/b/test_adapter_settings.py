"""Adapters read the merged `Settings` (PR #26): option factories and offline_client."""

from __future__ import annotations

import httpx
import pytest

from sevi.adapters.evon import EvonOptions
from sevi.adapters.prisma_stream import PrismaOptions
from sevi.adapters.timbre import offline_client
from sevi.adapters.twilio_rest import TwilioOptions
from sevi.app.settings import Settings
from sevi.contracts.enums import KeyId, Lang
from sevi.fakes.fake_clock import FakeClock
from sevi.fakes.fake_timbre import FakeTimbre
from sevi.fakes.memory_ledger import MemoryLedger


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


@pytest.mark.asyncio
async def test_offline_client_reads_key_a_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEVI_GNANI__KEY_A", "env-key-a")
    fake = FakeTimbre()
    led = MemoryLedger()
    client = offline_client(
        ledger=led, http=httpx.AsyncClient(transport=fake.transport()), clock=FakeClock()
    )
    await client.synthesize_wav("ஒன்று", Lang.TA, "Chetana", "timbre-v3.0")
    assert fake.headers[0]["x-api-key-id"] == "env-key-a"
    assert led.events[0].key_id is KeyId.A
