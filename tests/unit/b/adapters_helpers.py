"""Shared builders for B's adapter tests (no network)."""

from __future__ import annotations

import asyncio
from typing import Any

from sevi.contracts.adapters import SttSessionConfig, VadConfig
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.fakes.clock import FakeClock

KEYTERMS_TA = ("ஒன்று", "onru", "one", "இரண்டு", "irandu", "two")
RID = "01933b6e-8f1a-7c2d-9e4b-2a1f0c9d8e7b-s1"


def stt_cfg(**kw: Any) -> SttSessionConfig:
    base: dict[str, Any] = {
        "key_id": KeyId.B,
        "lang": Lang.TA,
        "model_version": ModelVersion.V3,
        "format": "transcribe",
        "keyterms": KEYTERMS_TA,
        "sample_rate_hz": 8000,
        "vad": VadConfig(),
        "idle_timeout_s": 900,
        "request_id": RID,
    }
    base.update(kw)
    return SttSessionConfig.model_validate(base)


class FakeWs:
    """Minimal stand-in for a websockets ClientConnection."""

    def __init__(self, clock: FakeClock, *, stall: bool = False) -> None:
        self.clock = clock
        self.stall = stall
        self.sent: list[tuple[float, bytes | str]] = []
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.closed = False

    async def send(self, data: bytes | str) -> None:
        if self.stall:
            await asyncio.Event().wait()
        self.sent.append((self.clock.mono_ms(), data))

    async def recv(self) -> str:
        m = await self.incoming.get()
        assert m is not None
        return m

    def __aiter__(self) -> FakeWs:
        return self

    async def __anext__(self) -> str:
        m = await self.incoming.get()
        if m is None:
            raise StopAsyncIteration
        return m

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if not self.closed:
            self.closed = True
            self.close_code = code
            self.incoming.put_nowait(None)
