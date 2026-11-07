"""Phone IVR rig: real A core, SqliteStore, FakeTwilioRest and a FastAPI app over ASGI."""

from __future__ import annotations

import io
import re
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from orch_harness import RecordingTelemetry, shared_core

from sevi.adapters.ledger import Ledger, TwilioLimits
from sevi.app.ivr import PhoneCall, PhoneDeps, PhoneService, Transcriber
from sevi.app.prompts import placeholder_prompts
from sevi.app.routes.twilio import install_phone, operator_router
from sevi.app.settings import Settings
from sevi.app.store import SqliteStore
from sevi.contracts.enums import Lang
from sevi.core.result import t8_decision
from sevi.core.staircase import PROCEDURE
from sevi.core.stimulus import BankStimulusProvider
from sevi.fakes.clock import FakeClock
from sevi.fakes.fake_twilio_rest import TEST_AUTH_TOKEN, FakeTwilioRest
from sevi.fakes.sim_twilio import SimPhone

BASE = "https://sevi.example.test"
VERIFIED = "P-01:+919800000000"


def phone_settings(*, enabled: bool = True, **twilio: Any) -> Settings:
    tw: dict[str, Any] = {
        "account_sid": "AC" + "0" * 32,
        "auth_token": TEST_AUTH_TOKEN,
        "from_number": "+15005550006",
        "verified_to_numbers": [VERIFIED],
        **twilio,
    }
    return Settings(env="test", public_base_url=BASE, phone={"enabled": enabled}, twilio=tw)


def wav_pcm16(seconds: float = 1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x01" * int(8000 * seconds))
    return buf.getvalue()


@dataclass
class PhoneRig:
    svc: PhoneService
    client: httpx.AsyncClient
    twilio: FakeTwilioRest
    clock: FakeClock
    store: SqliteStore
    tm: RecordingTelemetry
    bank: BankStimulusProvider

    async def dial(self, pid: str = "P-01", key: str = "k1", **body: Any) -> httpx.Response:
        return await self.client.post(
            "/op/api/calls",
            json={"participant_id": pid, "participant_type": "PILOT", **body},
            headers={"Idempotency-Key": key},
        )

    def call(self, sid: str) -> PhoneCall:
        return self.svc.calls[sid]

    def perfect(self, sid: str) -> Any:
        """Answerer that keys the presented triplet for every presentation."""

        def answer(action: str, n: int) -> str:
            if "/c1" in action or "/c2" in action:
                return "1"
            pending = self.call(sid).pending
            assert pending is not None
            return "".join(str(d) for d in pending.triplet)

        return answer

    async def sim(self, **kw: Any) -> SimPhone:
        return SimPhone(self.client, self.twilio, self.twilio.created[-1], **kw)

    async def run_call(self, answer: Any = None, **kw: Any) -> tuple[str, SimPhone]:
        r = await self.dial()
        assert r.status_code == 201, r.text
        sid = r.json()["session_id"]
        phone = await self.sim(answer=answer or self.perfect(sid), **kw)
        await phone.run()
        return sid, phone


@asynccontextmanager
async def phone_rig(
    bank: BankStimulusProvider,
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    transcribe: Transcriber | None = None,
    scorer: Any = None,
) -> AsyncIterator[PhoneRig]:
    real_scorer, classifier, copy = shared_core()
    clock = FakeClock()
    s = settings or phone_settings()
    tm = RecordingTelemetry()
    store = await SqliteStore(tmp_path / "sevi.sqlite", clock=clock).open()
    fake = FakeTwilioRest()
    ledger = Ledger(
        store,
        s.ledger,
        alias_b=True,
        clock=clock,
        telemetry=tm,  # type: ignore[arg-type]
        twilio=TwilioLimits(
            trial_minutes_total=s.twilio.trial_minutes_total,
            trial_minutes_stop=s.twilio.trial_minutes_stop,
        ),
    )
    deps = PhoneDeps(
        settings=s,
        clock=clock,
        telemetry=tm,  # type: ignore[arg-type]
        store=store,
        procedure=PROCEDURE,
        scorer=scorer or real_scorer,
        classifier=classifier,
        t8_decision=t8_decision,
        banks={Lang.TA: bank},
        prompts={Lang.TA: placeholder_prompts(Lang.TA, bank.manifest.digit_set)},
        twilio=fake,
        minutes=ledger,
        copy_text=copy.text,
        transcribe=transcribe,
    )
    svc = PhoneService(deps, arm_dial_timer=False, step_deadline_s=30.0, record_deadline_s=30.0)
    app = FastAPI()
    install_phone(app, svc)
    app.include_router(operator_router())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as client:
        try:
            yield PhoneRig(svc, client, fake, clock, store, tm, bank)
        finally:
            await svc.close()
            await store.close()


def step_pid(action: str) -> int | None:
    m = re.search(r"/step/[^/]+/[pr](\d+)", action)
    return int(m.group(1)) if m else None
