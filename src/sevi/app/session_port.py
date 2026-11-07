"""The port between the HTTP/WS layer and the session runtime (B spec §2.4, §2.5, §7).

Admission, the WS route and the operator API only need a small surface of the orchestrator:
create a session (with an id and token chosen by admission, so a QUEUED voice session keeps the
id and token it was given), look one up, list them, end or abort one. `SessionPort` is that
surface. The app runs the real orchestrator through `app.runtime.OrchestratorPort`.
`StubSessionPort` remains for tests that need a runtime without the orchestrator (inject it with
`create_app(port_factory=...)`): it accepts the socket, checks the `start` token and ends the
session on `stop`/close. It never calls an upstream API.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from sevi.contracts.adapters import CallTransport, Clock, TClosed, Telemetry, TStart, TStop
from sevi.contracts.enums import EndReason, KeyId, Lang, Mode, SessionState
from sevi.contracts.errors import WS_CLOSE_CODE, ErrorCode
from sevi.contracts.telemetry import EventName
from sevi.contracts.wire import CreateSessionRequest


class PermitLike(Protocol):
    def release(self) -> None: ...


class SessionHandle(Protocol):
    """What the app layer reads from a live session."""

    @property
    def session_id(self) -> str: ...
    @property
    def state(self) -> SessionState: ...
    @property
    def mode_requested(self) -> Mode: ...
    @property
    def lang(self) -> Lang: ...
    @property
    def connected(self) -> bool: ...
    @property
    def created_at(self) -> datetime: ...
    @property
    def ended_at(self) -> datetime | None: ...
    @property
    def end_reason(self) -> EndReason | None: ...

    def token_ok(self, token: str) -> bool: ...
    async def accept(self, transport: CallTransport) -> None: ...
    async def run(self) -> None: ...


class SessionPort(Protocol):
    def create(
        self,
        req: CreateSessionRequest,
        *,
        session_id: str,
        ws_token: str,
        permit: PermitLike | None,
        key_id: KeyId,
    ) -> SessionHandle: ...
    def get(self, session_id: str) -> SessionHandle | None: ...
    def all(self) -> Iterable[SessionHandle]: ...
    def remove(self, session_id: str) -> None: ...
    async def end(self, session_id: str, reason: EndReason) -> bool: ...
    async def abort(self, session_id: str, reason: str) -> bool: ...


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def token_matches(token: str, token_sha256: str) -> bool:
    """Constant-time compare of a presented token against its stored sha256."""
    return hmac.compare_digest(sha256_hex(token).encode(), token_sha256.encode())


# ---------- stub runtime (tests only) ----------


class StubSession:
    def __init__(
        self,
        req: CreateSessionRequest,
        *,
        session_id: str,
        ws_token: str,
        permit: PermitLike | None,
        key_id: KeyId,
        clock: Clock,
        telemetry: Telemetry,
    ) -> None:
        self._sid = session_id
        self._mode = req.mode
        self._lang = req.lang
        self.key_id = key_id
        self.permit = permit
        self._token_sha = sha256_hex(ws_token)
        self._clock = clock
        self._tel = telemetry
        self._state = SessionState.AWAIT_START
        self._created_at = clock.now_utc()
        self._ended_at: datetime | None = None
        self._end_reason: EndReason | None = None
        self.transport: CallTransport | None = None

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def mode_requested(self) -> Mode:
        return self._mode

    @property
    def lang(self) -> Lang:
        return self._lang

    @property
    def connected(self) -> bool:
        return self.transport is not None and self._state is not SessionState.ENDED

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def ended_at(self) -> datetime | None:
        return self._ended_at

    @property
    def end_reason(self) -> EndReason | None:
        return self._end_reason

    def token_ok(self, token: str) -> bool:
        return token_matches(token, self._token_sha)

    async def accept(self, transport: CallTransport) -> None:
        self.transport = transport
        self._tel.emit(EventName.WS_ACCEPTED, session_id=self._sid)

    async def run(self) -> None:
        t = self.transport
        if t is None:
            return
        try:
            async for ev in t.events():
                if isinstance(ev, TStart):
                    if not self.token_ok(ev.custom.get("token", "")):
                        await t.close(WS_CLOSE_CODE[ErrorCode.SESSION_TOKEN_INVALID], "bad token")
                        self.finish(EndReason.QUEUE_ABANDONED)
                        return
                    self._state = SessionState.CONSENT
                    self._tel.emit(EventName.SESSION_STARTED, session_id=self._sid)
                elif isinstance(ev, TStop):
                    self.finish(EndReason.CALLER_HANGUP)
                    return
                elif isinstance(ev, TClosed):
                    self.finish(EndReason.CALLER_HANGUP if ev.clean else EndReason.SOCKET_LOST)
                    return
        finally:
            self.finish(EndReason.SOCKET_LOST)

    def finish(self, reason: EndReason) -> None:
        if self._state is SessionState.ENDED:
            return
        self._state = SessionState.ENDED
        self._end_reason = reason
        self._ended_at = self._clock.now_utc()
        if self.permit is not None:
            self.permit.release()
        self._tel.emit(EventName.SESSION_ENDED, session_id=self._sid, end_reason=reason.value)

    async def close(self) -> None:
        if self.transport is not None:
            with contextlib.suppress(Exception):
                await self.transport.close(1000, "")


class StubSessionPort:
    """In-memory `SessionPort` for tests (the app default is `runtime.OrchestratorPort`)."""

    def __init__(self, clock: Clock, telemetry: Telemetry) -> None:
        self.clock = clock
        self.telemetry = telemetry
        self.sessions: dict[str, StubSession] = {}

    def create(
        self,
        req: CreateSessionRequest,
        *,
        session_id: str,
        ws_token: str,
        permit: PermitLike | None,
        key_id: KeyId,
    ) -> StubSession:
        s = StubSession(
            req,
            session_id=session_id,
            ws_token=ws_token,
            permit=permit,
            key_id=key_id,
            clock=self.clock,
            telemetry=self.telemetry,
        )
        self.sessions[session_id] = s
        return s

    def get(self, session_id: str) -> StubSession | None:
        return self.sessions.get(session_id)

    def all(self) -> Iterable[StubSession]:
        return list(self.sessions.values())

    def remove(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def end(self, session_id: str, reason: EndReason) -> bool:
        s = self.sessions.get(session_id)
        if s is None or s.state is SessionState.ENDED:
            return False
        s.finish(reason)
        await s.close()
        return True

    async def abort(self, session_id: str, reason: str) -> bool:
        return await self.end(session_id, EndReason.OPERATOR_ABORT)
