"""Operator API (B spec §9.5; contracts §9.1 `L=O` rows, §9.3). API only, no console HTML.

Auth (operator listener only; never funnelled or tunnelled):
1. peer not in `SEVI_OPERATOR__ALLOWED_CIDRS`, or the request carries `X-Forwarded-For`,
   `CF-Connecting-IP` or `Cf-Ray` -> 403 `FORBIDDEN_NETWORK`;
2. CSRF: an `Origin` other than the operator listener's own -> 403; every POST also needs the
   header `X-Sevi-Op: 1` (forces a CORS preflight the operator app never answers);
3. HTTP Basic: user == `SEVI_OPERATOR__USERNAME` and scrypt(password) equals the stored
   `SEVI_OPERATOR__PASSWORD_SCRYPT` (`hmac.compare_digest`) -> else 401 with
   `WWW-Authenticate`. No configured hash -> every request is 401 (fail closed). Successful
   credentials are cached 300 s by sha256(user:password).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import secrets
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import Field

from sevi.app.admission import http_error, new_session_id
from sevi.app.settings import SCRYPT_RE, OperatorSettings
from sevi.contracts.adapters import LedgerEvent
from sevi.contracts.enums import (
    AllowlistKind,
    EndReason,
    KeyId,
    Lang,
    Mode,
    RelationLabel,
    SessionState,
)
from sevi.contracts.errors import ErrorCode
from sevi.contracts.telemetry import EventName
from sevi.contracts.types import ParticipantId, SessionId, SeviModel
from sevi.contracts.wire import ClientHello, CreateSessionRequest, ReadyzResponse, SwitchRequest

if TYPE_CHECKING:
    from sevi.app.session_port import SessionHandle
    from sevi.app.state import AppState

FORWARDING_HEADERS: Final = ("x-forwarded-for", "cf-connecting-ip", "cf-ray")
CSRF_HEADER: Final = "x-sevi-op"
REALM: Final = 'Basic realm="sevi-op"'
AUTH_CACHE_MS: Final = 300_000.0
SCRYPT_N, SCRYPT_R, SCRYPT_P = 16384, 8, 1
SCRYPT_MAXMEM: Final = 64 * 1024 * 1024
STATION_IDEM_TTL_S: Final = 86_400


# ---------- password hashing (`sevi op-password`) ----------


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt if salt is not None else secrets.token_bytes(16)
    h = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
        dklen=32,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not SCRYPT_RE.match(stored):
        return False
    *_, salt_hex, hash_hex = stored.split("$")
    want = bytes.fromhex(hash_hex)
    got = hashlib.scrypt(
        password.encode(),
        salt=bytes.fromhex(salt_hex),
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=SCRYPT_MAXMEM,
        dklen=len(want),
    )
    return hmac.compare_digest(got, want)


# ---------- auth ----------


class OperatorAuth:
    def __init__(self, cfg: OperatorSettings, own_origin: str, now_ms: Any) -> None:
        self.cfg = cfg
        self.own_origin = own_origin.rstrip("/").lower()
        self._now_ms = now_ms
        self._cache: dict[str, float] = {}

    def _peer_allowed(self, peer: str | None) -> bool:
        try:
            ip = ipaddress.ip_address(peer or "")
        except ValueError:
            return False
        return any(ip.version == n.version and ip in n for n in self.cfg.allowed_cidrs)

    def _basic(self, authorization: str | None) -> tuple[str, str] | None:
        if not authorization or not authorization.lower().startswith("basic "):
            return None
        try:
            raw = base64.b64decode(authorization[6:].strip(), validate=True).decode()
        except (binascii.Error, UnicodeDecodeError):
            return None
        user, sep, pw = raw.partition(":")
        return (user, pw) if sep else None

    def check(self, peer: str | None, headers: Mapping[str, str], method: str) -> None:
        """Raises `AdmissionError` (403 / 401) or returns None."""
        if not self._peer_allowed(peer) or any(h in headers for h in FORWARDING_HEADERS):
            raise http_error(ErrorCode.FORBIDDEN_NETWORK, "operator network only")
        origin = headers.get("origin")
        if origin is not None and origin.rstrip("/").lower() != self.own_origin:
            raise http_error(ErrorCode.FORBIDDEN_NETWORK, "cross-origin request")
        if method.upper() not in ("GET", "HEAD", "OPTIONS") and headers.get(CSRF_HEADER) != "1":
            raise http_error(ErrorCode.FORBIDDEN_NETWORK, "missing X-Sevi-Op header")
        creds = self._basic(headers.get("authorization"))
        stored = self.cfg.password_scrypt.get_secret_value() if self.cfg.password_scrypt else ""
        if creds is None or not stored:
            raise http_error(ErrorCode.UNAUTHORIZED, "operator credentials required")
        user, pw = creds
        ck = hashlib.sha256(f"{user}:{pw}".encode()).hexdigest()
        now = self._now_ms()
        hit = self._cache.get(ck)
        if hit is not None and now - hit < AUTH_CACHE_MS:
            return
        user_ok = hmac.compare_digest(user.encode(), self.cfg.username.encode())
        pw_ok = verify_password(pw, stored)  # always computed: no user-name timing oracle
        if not (user_ok and pw_ok):
            raise http_error(ErrorCode.UNAUTHORIZED, "operator credentials required")
        self._cache[ck] = now


# ---------- B-internal views (§9.5) ----------


class ParticipantView(SeviModel):
    participant_id: ParticipantId
    kind: AllowlistKind
    relation_label: RelationLabel
    default_lang: Lang
    active: bool
    expires_at: datetime


class SessionView(SeviModel):
    session_id: SessionId
    channel: Literal["BROWSER"] = "BROWSER"
    mode_requested: Mode
    lang: Lang
    state: SessionState
    created_at: datetime
    ended_at: datetime | None
    end_reason: EndReason | None


class StationRequest(SeviModel):
    """Supervised station session (validation station is cut; a minimal operator path)."""

    lang: Lang = Lang.TA
    mode: Literal[Mode.KEYPAD, Mode.VOICE] = Mode.KEYPAD


class StationResponse(SeviModel):
    session_id: SessionId
    test_url: str


class ReconcileResponse(SeviModel):
    inserted: int
    buckets: dict[str, float] = Field(default_factory=dict)


def session_view(h: SessionHandle) -> SessionView:
    return SessionView(
        session_id=h.session_id,
        mode_requested=h.mode_requested,
        lang=h.lang,
        state=h.state,
        created_at=h.created_at,
        ended_at=h.ended_at,
        end_reason=h.end_reason,
    )


_STATION_CLIENT = ClientHello(
    browser_family="Other",
    os_family="Other",
    ua="station",
    audio_worklet=True,
    sample_rate_hz=48000,
    page_version="station",
)


class OperatorService:
    def __init__(self, app: AppState) -> None:
        self.app = app

    async def participants(self) -> list[ParticipantView]:
        return [
            ParticipantView(
                participant_id=e.participant_id,
                kind=e.kind,
                relation_label=e.relation_label,
                default_lang=e.default_lang,
                active=e.active,
                expires_at=e.expires_at,
            )
            for e in await self.app.store.allowlist_list()
        ]

    def sessions(self, limit: int = 50) -> list[SessionView]:
        hs = sorted(self.app.port.all(), key=lambda h: h.created_at, reverse=True)
        return [session_view(h) for h in hs[: max(1, min(limit, 500))]]

    async def abort(self, session_id: str) -> SessionView:
        h = self.app.port.get(session_id)
        if h is None:
            raise http_error(ErrorCode.NOT_FOUND, "unknown session")
        if h.state is not SessionState.ENDED:  # repeat-safe
            await self.app.port.abort(session_id, "operator")
        return session_view(h)

    async def set_switch(self, req: SwitchRequest) -> ReadyzResponse:
        sw = self.app.switches
        if req.name == "friends":
            value: bool = req.value in (True, "ALLOWED")
        elif isinstance(req.value, bool):
            value = req.value
        else:
            raise http_error(ErrorCode.VALIDATION_FAILED, f"{req.name} takes a boolean")
        setattr(sw, req.name, value)
        self.app.telemetry.emit(EventName.SWITCH_CHANGED, "WARNING", name=req.name, value=value)
        if hasattr(self.app.credit, "invalidate"):
            self.app.credit.invalidate()
        return await self.app.readiness.full()

    async def ledger_show(self) -> dict[str, Any]:
        return await self.app.ledger.show()

    async def reconcile(self, ev: LedgerEvent) -> ReconcileResponse:
        if ev.origin != "manual" or ev.dashboard_used is None:
            raise http_error(ErrorCode.VALIDATION_FAILED, "need origin=manual and dashboard_used")
        n = await self.app.ledger.reconcile(ev)
        if hasattr(self.app.credit, "invalidate"):
            self.app.credit.invalidate()
        shown = await self.app.ledger.show()
        key = self.app.ledger.resolve(ev.key_id).value
        buckets = shown["keys"].get(key, {}).get("buckets", {})
        return ReconcileResponse(inserted=n, buckets={k: float(v) for k, v in buckets.items()})

    async def create_station(self, req: StationRequest, idem_key: str | None) -> StationResponse:
        if not idem_key:
            raise http_error(ErrorCode.VALIDATION_FAILED, "Idempotency-Key header required")
        store = self.app.store
        cached = await store.idempotency_get("station", idem_key)
        if cached is not None:
            return StationResponse.model_validate(json.loads(cached))
        if req.lang in self.app.settings.result.experimental_langs or req.lang not in (
            self.app.banks
        ):
            raise http_error(ErrorCode.LANG_UNAVAILABLE, "language unavailable for stations")
        permit = None
        key_id = KeyId.NONE
        if req.mode is Mode.VOICE:
            permit = self.app.sem.try_take_stream(KeyId.A)
            if permit is None:
                raise http_error(ErrorCode.VOICE_UNAVAILABLE, "no Key A stream permit")
            key_id = KeyId.A
        sid = new_session_id()
        token = secrets.token_urlsafe(16)
        self.app.port.create(
            CreateSessionRequest(
                lang=req.lang, mode=req.mode, earphones=True, client=_STATION_CLIENT
            ),
            session_id=sid,
            ws_token=token,
            permit=permit,
            key_id=key_id,
        )
        base = str(self.app.settings.pages_origin).rstrip("/")
        resp = StationResponse(session_id=sid, test_url=f"{base}/test.html#s={sid}&k={token}")
        await store.idempotency_put("station", idem_key, resp.model_dump_json(), STATION_IDEM_TTL_S)
        return resp
