"""PrismaStreamClient against fake_prisma over a real socket (B spec §13.4 F-PRS-*).

The orchestrator-level halves of these IDs (keypad switch, S51, scoring) belong to the
orchestrator PR; here we check the adapter's half: codes, breaker effects, events, ledger.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from sevi.adapters.prisma_stream import PrismaOptions, PrismaStreamClient
from sevi.contracts.adapters import (
    FrameTag,
    SttClosed,
    SttCommitSent,
    SttProcessing,
    SttSessionConfig,
    SttTranscript,
    VadConfig,
)
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.contracts.errors import BreakerOpenError, BudgetExceeded, ErrorCode, SttError
from sevi.contracts.telemetry import EventName
from sevi.fakes.fake_clock import ScaledClock
from sevi.fakes.fake_prisma import FakePrisma, Rule, silent_frame, voiced_frame
from sevi.fakes.memory_ledger import MemoryBreakers, MemoryLedger
from sevi.fakes.null_telemetry import RecordingTelemetry

CFG = SttSessionConfig(
    key_id=KeyId.B,
    lang=Lang.TA,
    model_version=ModelVersion.V3,
    format="transcribe",
    keyterms=("ஒன்று", "one"),
    sample_rate_hz=8000,
    vad=VadConfig(),
    idle_timeout_s=900,
    request_id="01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b-s1",
)
TAG = FrameTag(presentation_id=3, window_index=1)


def _client(
    url: str, *, ledger: MemoryLedger | None = None, timeout: float = 1.0
) -> tuple[PrismaStreamClient, MemoryBreakers, MemoryLedger, RecordingTelemetry]:
    breakers, led, tel = MemoryBreakers(), ledger or MemoryLedger(), RecordingTelemetry()
    c = PrismaStreamClient(
        PrismaOptions(ws_url=url, connect_timeout_s=timeout, send_timeout_s=1.0),
        {KeyId.A: "KA", KeyId.B: "KB"},
        breakers,
        led,
        ScaledClock(speed=100.0),  # pacing and backoff run 100x faster than real time
        tel,
        rng=random.Random(1),  # noqa: S311
    )
    return c, breakers, led, tel


async def _collect(s: Any, until: type, timeout: float = 3.0) -> list[Any]:
    out: list[Any] = []

    async def run() -> None:
        async for ev in s.events():
            out.append(ev)
            if isinstance(ev, until):
                return

    await asyncio.wait_for(run(), timeout)
    return out


@pytest.mark.asyncio
async def test_end_to_end_transcript_bound_to_window_and_ledger_commit() -> None:
    async with FakePrisma([Rule("4 7 2", latency_ms=20)]) as fp:
        c, _, led, tel = _client(await fp.start())
        s = await c.open(CFG)
        assert fp.connections[0].headers["x-api-key-id"] == "KB"
        for _ in range(8):
            s.send_frame(voiced_frame(), TAG)
        for _ in range(6):
            s.send_frame(silent_frame(), None, gated_zero=True)
        evs = await _collect(s, SttTranscript)
        tr = evs[-1]
        assert tr.text == "4 7 2"
        assert any(isinstance(e, SttProcessing) for e in evs)
        await s.close()
        assert len(led.events) == 1
        assert led.events[0].stt_s > 0
        assert led.events[0].key_id is KeyId.B
        assert tel.of(EventName.STT_OPEN)[0]["keyterm_headers"] == [
            "keyterms",
            "x-keyterms",
            "x-keywords",
        ]


@pytest.mark.asyncio
async def test_F_PRS_01_429_every_attempt_rate_limited_breaker_plus_one() -> None:
    async with FakePrisma(http_status=429) as fp:
        c, breakers, led, _ = _client(await fp.start())
        with pytest.raises(SttError) as e:
            await c.open(CFG)
        assert e.value.code is ErrorCode.STT_RATE_LIMITED
        assert len(fp.rejected) == 3
        assert breakers.get("prisma", KeyId.B).failures == 1
        assert led.reservations == {}
        assert led.events == []  # nothing billed: the socket never opened


@pytest.mark.asyncio
async def test_F_PRS_02_error_and_1008_handshake_rejected_forces_breaker_open() -> None:
    async with FakePrisma(handshake_error="policy violation") as fp:
        c, breakers, _, _ = _client(await fp.start())
        with pytest.raises(SttError) as e:
            await c.open(CFG)
        assert e.value.code is ErrorCode.STT_HANDSHAKE_REJECTED
        b = breakers.get("prisma", KeyId.B)
        assert b.forced == 1
        assert len(fp.connections) == 1  # not retried
        with pytest.raises(BreakerOpenError):
            await c.open(CFG)


@pytest.mark.asyncio
async def test_F_PRS_04_garbage_json_skipped_session_continues() -> None:
    async with FakePrisma([Rule("5 6 1", latency_ms=10)], garbage_json=True) as fp:
        c, _, _, tel = _client(await fp.start())
        s = await c.open(CFG)
        for _ in range(8):
            s.send_frame(voiced_frame(), TAG)
        for _ in range(6):
            s.send_frame(silent_frame(), None)
        evs = await _collect(s, SttTranscript)
        assert evs[-1].text == "5 6 1"
        assert len(tel.of(EventName.WIRE_UNKNOWN_MESSAGE)) == 2
        await s.close()


@pytest.mark.asyncio
async def test_F_PRS_05_never_final_processing_without_transcript() -> None:
    async with FakePrisma([Rule("1 2 3", latency_ms=10)], never_final=True) as fp:
        c, _, _, _ = _client(await fp.start())
        s = await c.open(CFG)
        for _ in range(8):
            s.send_frame(voiced_frame(), TAG)
        s.commit()
        evs = await _collect(s, SttProcessing)
        assert any(isinstance(e, SttCommitSent) for e in evs)
        await asyncio.sleep(0.1)
        await s.close()
        rest = [e async for e in s.events()]
        assert not any(isinstance(e, SttTranscript) for e in rest)


@pytest.mark.asyncio
async def test_F_PRS_06_commit_written_after_queued_frames() -> None:
    async with FakePrisma([Rule("4 7 2", latency_ms=10)]) as fp:
        c, _, _, _ = _client(await fp.start())
        s = await c.open(CFG)
        for _ in range(4):
            s.send_frame(voiced_frame(), TAG)
        s.commit()
        evs = await _collect(s, SttTranscript)
        conn = fp.connections[0]
        assert len(conn.frames) == 4
        assert conn.commits == [4 * 64.0]  # the commit arrived after the 4th frame
        assert evs[-1].text == "4 7 2"
        await s.close()


@pytest.mark.asyncio
async def test_F_PRS_07_split_segments_consecutive_indices() -> None:
    async with FakePrisma([Rule("4 7 2", latency_ms=10)], split_segments=True) as fp:
        c, _, _, _ = _client(await fp.start())
        s = await c.open(CFG)
        for _ in range(8):
            s.send_frame(voiced_frame(), TAG)
        for _ in range(6):
            s.send_frame(silent_frame(), None)
        first = (await _collect(s, SttTranscript))[-1]
        s.commit()
        second = (await _collect(s, SttTranscript))[-1]
        assert (first.text, second.text) == ("4", "7 2")
        assert (first.segment_index, second.segment_index) == (1, 2)
        await s.close()


@pytest.mark.asyncio
async def test_F_PRS_08_error_message_mid_session_is_an_event_only() -> None:
    async with FakePrisma(error_after_commit=True) as fp:
        c, breakers, _, _ = _client(await fp.start())
        s = await c.open(CFG)
        s.commit()
        from sevi.contracts.adapters import SttErrorMsg

        evs = await _collect(s, SttErrorMsg)
        assert evs[-1].message == "STT engine failed"
        assert breakers.get("prisma", KeyId.B).failures == 0
        await s.close()


@pytest.mark.parametrize(
    ("status", "code"), [(401, ErrorCode.STT_AUTH), (402, ErrorCode.STT_CREDITS)]
)
@pytest.mark.asyncio
async def test_F_PRS_09_10_auth_and_credits_force_open(status: int, code: ErrorCode) -> None:
    async with FakePrisma(http_status=status) as fp:
        c, breakers, led, _ = _client(await fp.start())
        with pytest.raises(SttError) as e:
            await c.open(CFG)
        assert e.value.code is code
        assert len(fp.rejected) == 1  # never retried
        assert breakers.get("prisma", KeyId.B).forced == 1
        assert (KeyId.B in led.exhausted) == (status == 402)
        if status == 402:  # the ledger now refuses the key outright
            breakers.get("prisma", KeyId.B).open = False
            with pytest.raises(BudgetExceeded):
                await c.open(CFG)


@pytest.mark.asyncio
async def test_F_PRS_11_connected_delay_connect_timeout_after_3_attempts() -> None:
    async with FakePrisma(connected_delay_ms=400) as fp:
        c, breakers, _, _ = _client(await fp.start(), timeout=0.1)
        with pytest.raises(SttError) as e:
            await c.open(CFG)
        assert e.value.code is ErrorCode.STT_CONNECT_TIMEOUT
        assert len(fp.connections) == 3
        assert breakers.get("prisma", KeyId.B).failures == 1


@pytest.mark.asyncio
async def test_F_PRS_15_close_1001_session_limit() -> None:
    async with FakePrisma(drop_after_frames=2, close_code=1001) as fp:
        c, _, led, tel = _client(await fp.start())
        s = await c.open(CFG)
        s.send_frame(silent_frame(), None)
        s.send_frame(silent_frame(), None)
        evs = await _collect(s, SttClosed)
        assert evs[-1].code == 1001
        assert tel.of(EventName.STT_CLOSED)[-1]["error"] == "STT_SESSION_LIMIT"
        await s.close()
        assert len(led.events) == 1


@pytest.mark.asyncio
async def test_F_PRS_16_config_echo_mismatch_no_breaker_count() -> None:
    async with FakePrisma(config_echo={"sample_rate": 16000}) as fp:
        c, breakers, _, _ = _client(await fp.start())
        with pytest.raises(SttError) as e:
            await c.open(CFG)
        assert e.value.code is ErrorCode.STT_CONFIG_MISMATCH
        b = breakers.get("prisma", KeyId.B)
        assert (b.failures, b.forced) == (0, 0)


@pytest.mark.asyncio
async def test_budget_refusal_makes_no_connection() -> None:
    async with FakePrisma() as fp:
        c, _, _, _ = _client(await fp.start(), ledger=MemoryLedger(budget=1.0))
        with pytest.raises(BudgetExceeded):
            await c.open(CFG)
        assert fp.connections == []
        assert fp.rejected == []
