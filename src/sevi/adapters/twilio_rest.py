"""Twilio trial REST adapter (B spec §5.7, contracts §8.6, §8.8). Demo only, behind
`SEVI_PHONE__ENABLED=false` by default.

All SDK calls run in a worker thread under `asyncio.timeout(5.0)`. `create_call` is never
retried; a client timeout raises `TWILIO_DIAL_FAILED` with `maybe_created=True` (the session
stays DIALING, I02). Update/delete retry twice (0.5 s, 1.0 s) on 5xx/connection errors; 404 on
delete = done. Before dialling, the optional `dial_guard` (the ledger's `check_dial`) refuses
once the trial-minute stop is reached (`TWILIO_TRIAL_MINUTES_STOP`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import httpx
from twilio.base.exceptions import TwilioRestException  # type: ignore[import-untyped]
from twilio.request_validator import RequestValidator  # type: ignore[import-untyped]

from sevi.adapters.deps import secret
from sevi.contracts.adapters import Clock, Telemetry
from sevi.contracts.errors import ErrorCode, TwilioError
from sevi.contracts.telemetry import EventName

T = TypeVar("T")
SDK_TIMEOUT_S = 5.0
RETRY_DELAYS_S = (0.5, 1.0)
STATUS_EVENTS = ["initiated", "ringing", "answered", "completed"]


@dataclass(frozen=True)
class TwilioOptions:
    account_sid: str
    auth_token: object
    from_number: str
    ring_timeout_s: int = 30
    time_limit_s: int = 600

    @classmethod
    def from_settings(cls, s: Any) -> TwilioOptions:
        """`s` = `Settings.twilio`."""
        return cls(s.account_sid, s.auth_token, s.from_number, s.ring_timeout_s, s.time_limit_s)


def _transient(e: BaseException) -> bool:
    if isinstance(e, TwilioRestException):
        return int(getattr(e, "status", 0) or 0) >= 500
    return isinstance(e, ConnectionError | OSError | httpx.TransportError)


class TwilioRest:
    def __init__(
        self,
        opts: TwilioOptions,
        clock: Clock,
        telemetry: Telemetry,
        *,
        client: Any = None,
        http: httpx.AsyncClient | None = None,
        dial_guard: Callable[[], Awaitable[None]] | None = None,
        sdk_timeout_s: float = SDK_TIMEOUT_S,
        retry_delays_s: tuple[float, ...] = RETRY_DELAYS_S,
    ) -> None:
        self.opts = opts
        self.clock = clock
        self.telemetry = telemetry
        if client is None:
            from twilio.rest import Client  # type: ignore[import-untyped]

            client = Client(opts.account_sid, secret(opts.auth_token))
        self.client = client
        self.http = http
        self.dial_guard = dial_guard
        self.sdk_timeout_s = sdk_timeout_s
        self.retry_delays_s = retry_delays_s
        self._validator = RequestValidator(secret(opts.auth_token))

    async def _sdk(self, fn: Callable[[], T]) -> T:
        async with asyncio.timeout(self.sdk_timeout_s):
            return await asyncio.to_thread(fn)

    async def _with_retries(self, fn: Callable[[], Any], what: str) -> Any:
        for attempt in range(len(self.retry_delays_s) + 1):
            try:
                return await self._sdk(fn)
            except TwilioRestException as e:
                if int(e.status or 0) == 404 and what.startswith("delete"):
                    return None
                if not _transient(e) or attempt == len(self.retry_delays_s):
                    raise TwilioError(
                        ErrorCode.TWILIO_REST_FAILED,
                        f"{what} failed",
                        details={"twilio_code": e.code, "status": e.status},
                    ) from None
            except (TimeoutError, ConnectionError, OSError) as e:
                if attempt == len(self.retry_delays_s):
                    raise TwilioError(
                        ErrorCode.TWILIO_REST_FAILED, f"{what}: {type(e).__name__}"
                    ) from None
            await self.clock.sleep_until(self.clock.mono_ms() + self.retry_delays_s[attempt] * 1000)
        raise AssertionError("unreachable")

    async def create_call(self, *, to_e164: str, session_id: str, base: str, t: str) -> str:
        """C§8.6.1 exactly. The number is never logged."""
        if self.dial_guard is not None:
            await self.dial_guard()
        o = self.opts

        def call() -> Any:
            return self.client.calls.create(
                to=to_e164,
                from_=o.from_number,
                url=f"{base}/twilio/voice/{session_id}?t={t}",
                method="POST",
                status_callback=f"{base}/twilio/status/{session_id}?t={t}",
                status_callback_event=STATUS_EVENTS,
                status_callback_method="POST",
                timeout=o.ring_timeout_s,
                time_limit=o.time_limit_s,
                record=False,
            )

        try:
            created = await self._sdk(call)
        except TimeoutError:
            raise TwilioError(
                ErrorCode.TWILIO_DIAL_FAILED,
                "calls.create timed out",
                details={"maybe_created": True},
            ) from None
        except TwilioRestException as e:
            raise TwilioError(
                ErrorCode.TWILIO_DIAL_FAILED,
                "calls.create rejected",
                details={"twilio_code": e.code, "status": e.status},
            ) from None
        except (ConnectionError, OSError) as e:
            raise TwilioError(ErrorCode.TWILIO_REST_FAILED, f"dial: {type(e).__name__}") from None
        sid = str(created.sid)
        self.telemetry.emit(EventName.TWILIO_DIAL, "INFO", session_id=session_id, call_sid=sid)
        return sid

    async def update_call(self, call_sid: str, status: Literal["canceled", "completed"]) -> None:
        await self._with_retries(
            lambda: self.client.calls(call_sid).update(status=status), "update"
        )

    async def delete_call(self, call_sid: str) -> None:
        await self._with_retries(lambda: self.client.calls(call_sid).delete(), "delete call")
        self.telemetry.emit(EventName.TWILIO_DELETED, "INFO", call_sid=call_sid)

    async def delete_recording(self, recording_sid: str) -> None:
        await self._with_retries(
            lambda: self.client.recordings(recording_sid).delete(), "delete rec"
        )

    async def fetch_recording_wav(self, recording_url: str) -> bytes:
        """GET {url}.wav with basic auth, 5 s, no retry."""
        http = self.http or httpx.AsyncClient(follow_redirects=True, trust_env=False)
        try:
            async with asyncio.timeout(self.sdk_timeout_s):
                r = await http.get(
                    f"{recording_url}.wav",
                    auth=(self.opts.account_sid, secret(self.opts.auth_token)),
                    timeout=httpx.Timeout(self.sdk_timeout_s, connect=2.0),
                )
        except (TimeoutError, httpx.HTTPError) as e:
            raise TwilioError(
                ErrorCode.TWILIO_REST_FAILED, f"recording: {type(e).__name__}"
            ) from None
        finally:
            if self.http is None:
                await http.aclose()
        if r.status_code != 200 or not r.content:
            raise TwilioError(
                ErrorCode.TWILIO_REST_FAILED,
                "recording fetch failed",
                details={"status": r.status_code},
            )
        return r.content

    def validate(self, url: str, form: Mapping[str, str], signature: str) -> bool:
        return bool(self._validator.validate(url, dict(form), signature))
