"""Drop-in fake for `adapters.twilio_rest.TwilioRest` (contracts §11.8, B spec §12.1).

Records `create_call` / `update_call` / `delete_call` / recording calls; `fail_create(code)` and
`timeout_create()` script the next dial; `emit_webhook()` POSTs a form to the app signed with the
test auth token (`RequestValidator.compute_signature`), exactly as Twilio would.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import httpx
from twilio.request_validator import RequestValidator  # type: ignore[import-untyped]

from sevi.contracts.errors import ErrorCode, TwilioError

TEST_AUTH_TOKEN = "0123456789abcdef0123456789abcdef"  # noqa: S105 (test-only token)


def new_call_sid() -> str:
    return "CA" + secrets.token_hex(16)


@dataclass
class CreatedCall:
    call_sid: str
    to_e164: str
    session_id: str
    url: str
    status_callback: str


@dataclass
class FakeTwilioRest:
    auth_token: str = TEST_AUTH_TOKEN
    created: list[CreatedCall] = field(default_factory=list)
    updated: list[tuple[str, str]] = field(default_factory=list)
    deleted_calls: list[str] = field(default_factory=list)
    deleted_recordings: list[str] = field(default_factory=list)
    fetched_recordings: list[str] = field(default_factory=list)
    recordings: dict[str, bytes] = field(default_factory=dict)  # url -> wav bytes
    _next_create_error: TwilioError | None = None

    # ---------- scripting ----------
    def fail_create(self, code: int, status: int = 400) -> None:
        self._next_create_error = TwilioError(
            ErrorCode.TWILIO_DIAL_FAILED,
            f"twilio error {code}",
            details={"twilio_code": code, "status": status},
        )

    def timeout_create(self) -> None:
        self._next_create_error = TwilioError(
            ErrorCode.TWILIO_DIAL_FAILED, "calls.create timed out", details={"maybe_created": True}
        )

    # ---------- TwilioRest API ----------
    async def create_call(self, *, to_e164: str, session_id: str, base: str, t: str) -> str:
        if self._next_create_error is not None:
            err, self._next_create_error = self._next_create_error, None
            raise err
        sid = new_call_sid()
        self.created.append(
            CreatedCall(
                sid,
                to_e164,
                session_id,
                f"{base}/twilio/voice/{session_id}?t={t}",
                f"{base}/twilio/status/{session_id}?t={t}",
            )
        )
        return sid

    async def update_call(self, call_sid: str, status: Literal["canceled", "completed"]) -> None:
        self.updated.append((call_sid, status))

    async def delete_call(self, call_sid: str) -> None:
        self.deleted_calls.append(call_sid)

    async def fetch_recording_wav(self, recording_url: str) -> bytes:
        self.fetched_recordings.append(recording_url)
        try:
            return self.recordings[recording_url]
        except KeyError:
            raise TwilioError(ErrorCode.TWILIO_REST_FAILED, "recording not found") from None

    async def delete_recording(self, recording_sid: str) -> None:
        self.deleted_recordings.append(recording_sid)

    def validate(self, url: str, form: Mapping[str, str], signature: str) -> bool:
        return bool(RequestValidator(self.auth_token).validate(url, dict(form), signature))

    # ---------- webhooks ----------
    def sign(self, url: str, form: Mapping[str, str]) -> str:
        return str(RequestValidator(self.auth_token).compute_signature(url, dict(form)))

    async def emit_webhook(
        self, client: httpx.AsyncClient, url: str, form: Mapping[str, str]
    ) -> httpx.Response:
        """POST `form` to the absolute `url` (as Twilio sees it) with a valid signature."""
        return await client.post(
            url, data=dict(form), headers={"X-Twilio-Signature": self.sign(url, form)}
        )
