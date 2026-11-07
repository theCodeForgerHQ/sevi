"""Simulated Twilio phone channel (B spec §12, E2E-SIM-01). No network, no real Twilio.

``SimPhone`` plays the callee *and* Twilio: it answers the call created by ``FakeTwilioRest``,
POSTs signed webhooks (``X-Twilio-Signature`` computed with the fake's auth token, exactly as
Twilio signs them), follows the returned TwiML verb by verb, fetches every ``<Play>`` URL from
``/audio`` and answers each ``<Gather>`` with digits from a callback (or each ``<Record>`` with
a scripted recording). When the TwiML hangs up (or ends) it sends the ``completed`` status
callback with a ``CallDuration``.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from sevi.fakes.fake_twilio_rest import CreatedCall, FakeTwilioRest

# (action URL, number of digits asked) -> keys pressed ("" = silence)
Answerer = Callable[[str, int], str]
# action URL -> (RecordingSid, RecordingUrl, RecordingDuration) or None for silence
Recorder = Callable[[str], tuple[str, str, float] | None]


@dataclass
class Step:
    url: str
    form: dict[str, str]
    status: int
    twiml: str


@dataclass
class SimPhone:
    client: httpx.AsyncClient
    twilio: FakeTwilioRest
    call: CreatedCall
    answer: Answerer = field(default=lambda _url, _n: "")
    record: Recorder | None = None
    steps: list[Step] = field(default_factory=list)
    played: list[str] = field(default_factory=list)  # audio names in play order
    audio: dict[str, bytes] = field(default_factory=dict)  # name -> WAV body
    audio_status: list[int] = field(default_factory=list)
    hung_up: bool = False
    _seq: itertools.count[int] = field(default_factory=itertools.count)
    max_steps: int = 200

    def _base_form(self) -> dict[str, str]:
        return {
            "AccountSid": "AC" + "0" * 32,
            "CallSid": self.call.call_sid,
            "From": "+15005550006",
            "To": "+919800000000",
            "Direction": "outbound-api",
        }

    async def post(self, url: str, extra: dict[str, str] | None = None) -> Step:
        form = {**self._base_form(), **(extra or {})}
        r = await self.twilio.emit_webhook(self.client, url, form)
        st = Step(url, form, r.status_code, r.text)
        self.steps.append(st)
        return st

    async def status(self, call_status: str, duration_s: int | None = None) -> httpx.Response:
        form = {
            **self._base_form(),
            "CallStatus": call_status,
            "SequenceNumber": str(next(self._seq)),
        }
        if duration_s is not None:
            form["CallDuration"] = str(duration_s)
        return await self.twilio.emit_webhook(self.client, self.call.status_callback, form)

    async def _play(self, url: str) -> None:
        path = urlsplit(url).path
        name = path.rsplit("/", 1)[-1].removesuffix(".wav")
        r = await self.client.get(url)
        self.audio_status.append(r.status_code)
        self.played.append(name)
        if r.status_code == 200:
            self.audio[name] = r.content

    async def run(self, *, duration_s: int = 240) -> None:
        """Answer, follow TwiML to the end, then send the ``completed`` status callback."""
        await self.status("ringing")
        st = await self.post(self.call.url)
        for _ in range(self.max_steps):
            nxt = await self._follow(st)
            if nxt is None:
                break
            st = nxt
        await self.status("completed", duration_s)

    async def _follow(self, st: Step) -> Step | None:
        if st.status != 200:
            return None
        root = ET.fromstring(st.twiml)  # noqa: S314 (our own TwiML, test-only)
        for verb in root:
            if verb.tag == "Play":
                await self._play((verb.text or "").strip())
            elif verb.tag == "Gather":
                for child in verb:
                    if child.tag == "Play":
                        await self._play((child.text or "").strip())
                action = verb.attrib["action"]
                n = int(verb.attrib.get("numDigits", "1"))
                digits = self.answer(action, n)
                return await self.post(action, {"Digits": digits} if digits else {})
            elif verb.tag == "Record":
                action = verb.attrib["action"]
                rec = self.record(action) if self.record is not None else None
                extra: dict[str, str] = {}
                if rec is not None:
                    sid, url, dur = rec
                    extra = {
                        "RecordingSid": sid,
                        "RecordingUrl": url,
                        "RecordingDuration": str(dur),
                    }
                return await self.post(action, extra)
            elif verb.tag == "Hangup":
                self.hung_up = True
                return None
        self.hung_up = True  # end of TwiML: Twilio hangs up
        return None


__all__ = ["Answerer", "Recorder", "SimPhone", "Step"]
