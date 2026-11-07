"""Admission: spike ladder, waiting room, per-IP limits (contracts §9.2.1, §9.3; B spec §9.1, §9.2).

- `keypad_full` = active keypad sessions >= `MAX_KEYPAD_SESSIONS` (120).
- `voice_offered` = browser_voice and prisma switches on, the public key's Prisma breaker closed,
  not `public_key_low`, not `exposure_paused`, voice queue length <= `VOICE_QUEUE_VISIBLE_MAX`.
- Level = max of L4 (readiness down / draining), L3 (credit stop, exposure, breaker, switch off),
  L2 (keypad full), L1 (queue > 3), L0.
- Per-IP hourly limits (keypad 20, voice 3) apply only when a trusted client IP exists; with
  none, per-session limits and the global caps hold and pre-`start` sockets are capped globally.
- Voice needs a stream permit on the public key (Key B, or Key A when B is aliased); without one
  the session waits in the queue (max `VOICE_QUEUE_VISIBLE_MAX` entries) and polls.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math
import secrets
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlsplit

from sevi.contracts.enums import AdmissionLevel, EndReason, KeyId, Mode, SessionState
from sevi.contracts.errors import AdmissionError, ErrorCode
from sevi.contracts.telemetry import EventName
from sevi.contracts.wire import (
    CreateSessionRequest,
    CreateSessionResponse,
    LangStatus,
    PublicStatusResponse,
    SessionStatusResponse,
)

if TYPE_CHECKING:
    from sevi.adapters.ledger import Ledger
    from sevi.app.session_port import PermitLike
    from sevi.app.settings import Settings
    from sevi.app.state import AppState

IPAddress = IPv4Address | IPv6Address

# §9.1: explicit list (`is_private` also flags documentation ranges such as 203.0.113.0/24).
NON_PUBLIC_NETS: Final = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)
IOS_SPEAKER_VOICE_OK: Final = False  # flipped by PR only after the iOS speaker row passes
LEVEL_L1_QUEUE: Final = 3
RATE_LIMIT_RETRY_S: Final = 600
DRAIN_RETRY_S: Final = 60
AHEAD_WINDOW_MS: Final = 60_000.0
STATUS_CACHE_MS: Final = 5_000.0
CREDIT_CACHE_MS: Final = 5_000.0
ENDED_KEEP_MS: Final = 5_000.0
IP_WINDOW_S: Final = 3600
UNRECONCILED_EXPOSURE_CAP_CREDITS: Final = 400.0
_LEVELS: Final = (
    AdmissionLevel.L0,
    AdmissionLevel.L1,
    AdmissionLevel.L2,
    AdmissionLevel.L3,
    AdmissionLevel.L4,
)
_LIVE: Final = frozenset(
    {
        SessionState.AWAIT_START,
        SessionState.CONSENT,
        SessionState.INSTRUCTIONS,
        SessionState.TEST,
        SessionState.RESULT,
        SessionState.CLOSING,
    }
)


def http_error(
    code: ErrorCode,
    message: str = "",
    *,
    retry_after_s: int | None = None,
    **details: str | int | float | bool,
) -> AdmissionError:
    d: dict[str, str | int | float | bool] = dict(details)
    if retry_after_s is not None:
        d["retry_after_s"] = retry_after_s
    return AdmissionError(code, message, details=d)


# ---------- client IP (§9.1, contracts §9.3) ----------


def is_public_ip(ip: IPAddress) -> bool:
    return not any(ip.version == n.version and ip in n for n in NON_PUBLIC_NETS)


def _parse_ip(raw: str) -> IPAddress | None:
    try:
        return ipaddress.ip_address(raw.strip())
    except ValueError:
        return None


def peer_trusted(
    peer: str | None, trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network]
) -> bool:
    ip = _parse_ip(peer) if peer else None
    return ip is not None and any(ip.version == n.version and ip in n for n in trusted)


def client_ip(
    peer: str | None,
    headers: Mapping[str, str],
    trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    header_name: str,
) -> IPAddress | None:
    """Untrusted peer -> the peer (headers ignored). Trusted peer -> the configured header
    (X-Forwarded-For: rightmost entry; CF-Connecting-IP: whole value), which must be public."""
    if not peer_trusted(peer, trusted):
        return _parse_ip(peer) if peer else None
    if not header_name:
        return None
    raw = headers.get(header_name.lower()) or headers.get(header_name)
    if not raw:
        return None
    value = raw.split(",")[-1] if header_name.lower() == "x-forwarded-for" else raw
    ip = _parse_ip(value)
    return ip if ip is not None and is_public_ip(ip) else None


def client_net(ip: IPAddress) -> IPAddress:
    if isinstance(ip, IPv4Address):
        return ip
    return ipaddress.ip_network(f"{ip}/64", strict=False).network_address


def ip_hmac(key: bytes, ip: IPAddress | None) -> str | None:
    if ip is None:
        return None
    return hmac.new(key, str(client_net(ip)).encode(), hashlib.sha256).hexdigest()


def ws_url(public_base_url: str, host: str | None, trusted_peer: bool, session_id: str) -> str:
    """U-ADM-04: the request Host is used only from a trusted ingress peer and only when it is
    the configured public host or a `trycloudflare.com` host (fallback tunnel)."""
    base = urlsplit(public_base_url.rstrip("/"))
    scheme = "wss" if base.scheme == "https" else "ws"
    netloc = base.netloc
    if trusted_peer and host:
        h = host.strip().lower()
        if h == base.netloc.lower():
            netloc = h
        elif h.endswith(".trycloudflare.com"):
            netloc, scheme = h, "wss"
    return f"{scheme}://{netloc}/web/stream?sid={session_id}"


def eta_s(position: int, voice_eta_s: int, permits: int) -> int:
    """U-ADM-03: ceil(position x VOICE_ETA_S / permits)."""
    return math.ceil(position * voice_eta_s / max(permits, 1))


def new_session_id() -> str:
    """UUIDv7-shaped id (time-ordered prefix not needed here; version/variant nibbles set)."""
    h = uuid.uuid4().hex
    return f"{h[:8]}-{h[8:12]}-7{h[13:16]}-8{h[17:20]}-{h[20:32]}"


# ---------- credit inputs (§9.2.1) ----------


class CreditGate(Protocol):
    async def public_key_low(self) -> bool: ...
    async def exposure_paused(self) -> bool: ...


class LedgerCreditGate:
    """`public_key_low` from the ledger; `exposure_paused` = unreconciled stream seconds on the
    public key x the configured STT rate >= 400 credits (a manual reconcile clears it). Both
    cached 5 s. Observed-rate refinements (r_obs, est_live) are not built (ledger scope)."""

    def __init__(self, ledger: Ledger, public_key: KeyId, clock_ms: Callable[[], float]) -> None:
        self.ledger = ledger
        self.public_key = public_key
        self._now = clock_ms
        self._cache: dict[str, tuple[float, bool]] = {}

    async def _cached(self, name: str, compute: Callable[[], Awaitable[bool]]) -> bool:
        now = self._now()
        hit = self._cache.get(name)
        if hit is not None and now - hit[0] < CREDIT_CACHE_MS:
            return hit[1]
        v = await compute()
        self._cache[name] = (now, v)
        return v

    def invalidate(self) -> None:
        self._cache.clear()

    async def public_key_low(self) -> bool:
        return await self._cached("low", lambda: self.ledger.public_key_low(self.public_key))

    async def exposure_paused(self) -> bool:
        return await self._cached("exposure", self._exposure)

    async def _exposure(self) -> bool:
        rate = self.ledger.cfg.credits_per_stt_s
        if rate is None:
            return False  # every billed call is refused by the ledger anyway
        physical = self.ledger.resolve(self.public_key)
        unreconciled = 0.0
        for ev in await self.ledger.store.ledger_events(physical):
            if ev.origin == "manual":
                unreconciled = 0.0
            else:
                unreconciled += ev.stt_s
        return unreconciled * rate >= UNRECONCILED_EXPOSURE_CAP_CREDITS


# ---------- admission ----------


@dataclass
class QueueEntry:
    session_id: str
    token_sha256: str
    req: CreateSessionRequest
    last_poll_mono: float
    ip_hmac: str | None = None


@dataclass(frozen=True)
class ClientContext:
    """What the route derived from the request (B spec §9.1)."""

    ip_hmac: str | None
    ws_base_host: str | None = None
    trusted_peer: bool = False


@dataclass
class _Expired:
    token_sha256: str


class Admission:
    def __init__(self, app: AppState) -> None:
        self.app = app
        self.queue: list[QueueEntry] = []
        self._deadline: dict[str, float] = {}  # admitted, not yet connected -> mono deadline
        self._expired: dict[str, _Expired] = {}
        self._wait_rejects: dict[str, float] = {}
        self._ended_seen: dict[str, float] = {}
        self._level: AdmissionLevel | None = None
        self._status_cache: tuple[float, PublicStatusResponse] | None = None
        self._no_ip_warned = False
        self._socket_bucket: dict[str, str | None] = {}

    # ----- small helpers -----
    @property
    def _s(self) -> Settings:
        return self.app.settings

    def _now(self) -> float:
        return self.app.clock.mono_ms()

    def public_key(self) -> KeyId:
        return self.app.settings.public_key

    def _permits(self) -> int:
        g = self.app.settings.gnani
        return g.key_b_permits if self.public_key() is KeyId.B else g.key_a_permits

    def _count(self, mode: Mode) -> int:
        return sum(1 for h in self.app.port.all() if h.mode_requested is mode and h.state in _LIVE)

    def active_keypad(self) -> int:
        return self._count(Mode.KEYPAD)

    def active_voice(self) -> int:
        return self._count(Mode.VOICE)

    def keypad_full(self) -> bool:
        return self.active_keypad() >= self._s.admission.max_keypad_sessions

    def warn_no_client_ip(self, header: str) -> None:
        if not self._no_ip_warned:
            self._no_ip_warned = True
            self.app.telemetry.emit(
                EventName.ADMISSION_NO_CLIENT_IP, "WARNING", ingress="trusted", header=header
            )

    # ----- C§9.2.1 -----
    def _breaker_open(self) -> bool:
        return self.app.breakers.get("prisma", self.public_key()).is_open()

    async def voice_offered(self) -> bool:
        sw = self.app.switches
        if not (sw.browser_voice and sw.prisma) or self._breaker_open():
            return False
        if await self.app.credit.public_key_low() or await self.app.credit.exposure_paused():
            return False
        return len(self.queue) <= self._s.admission.voice_queue_visible_max

    async def level(self) -> AdmissionLevel:
        lvl = AdmissionLevel.L0
        if self.app.draining or self.app.readiness.is_down():
            lvl = AdmissionLevel.L4
        elif (
            not (self.app.switches.browser_voice and self.app.switches.prisma)
            or self._breaker_open()
            or await self.app.credit.public_key_low()
            or await self.app.credit.exposure_paused()
        ):
            lvl = AdmissionLevel.L3
        elif self.keypad_full():
            lvl = AdmissionLevel.L2
        elif len(self.queue) > LEVEL_L1_QUEUE:
            lvl = AdmissionLevel.L1
        if lvl is not self._level:
            prev = self._level
            self._level = lvl
            self.app.telemetry.emit(
                EventName.ADMISSION_LEVEL, "INFO", to=lvl.value, prev=prev and prev.value
            )
        return lvl

    @staticmethod
    def level_rank(lvl: AdmissionLevel) -> int:
        return _LEVELS.index(lvl)

    # ----- POST /api/v1/sessions (S01-S05) -----
    def _reject(self, code: ErrorCode, reason: str) -> None:
        self.app.telemetry.emit(
            EventName.ADMISSION_REJECTED, "INFO", code=code.value, reason=reason
        )

    async def create(self, req: CreateSessionRequest, ctx: ClientContext) -> CreateSessionResponse:
        a = self._s.admission
        voice = req.mode is Mode.VOICE
        if req.lang not in self.app.banks:  # 1
            self._reject(ErrorCode.LANG_UNAVAILABLE, "lang")
            raise http_error(ErrorCode.LANG_UNAVAILABLE, "language unavailable")
        if self.app.draining or self.app.readiness.is_down():  # 2
            self._reject(ErrorCode.SERVICE_DEGRADED, "down")
            raise http_error(
                ErrorCode.SERVICE_DEGRADED, "service unavailable", retry_after_s=DRAIN_RETRY_S
            )
        if self.keypad_full():  # 3 (keypad and voice)
            now = self._now()
            self._wait_rejects[ctx.ip_hmac or f"anon:{secrets.token_hex(8)}"] = now
            self._wait_rejects = {
                k: t for k, t in self._wait_rejects.items() if now - t < AHEAD_WINDOW_MS
            }
            self._reject(ErrorCode.ADMISSION_WAIT, "keypad_full")
            raise http_error(
                ErrorCode.ADMISSION_WAIT,
                "please wait",
                retry_after_s=a.waiting_room_retry_s,
                ahead=max(1, len(self._wait_rejects)),
            )
        if voice:  # 4
            ios_speaker = req.client.os_family == "iOS" and not req.earphones
            if not await self.voice_offered() or (ios_speaker and not IOS_SPEAKER_VOICE_OK):
                self._reject(ErrorCode.VOICE_UNAVAILABLE, "voice")
                raise http_error(ErrorCode.VOICE_UNAVAILABLE, "voice unavailable")
        if ctx.ip_hmac is not None:  # 5 (per-IP only with a trusted client IP)
            prefix = "v:" if voice else "k:"
            limit = a.per_ip_voice_per_hour if voice else a.per_ip_keypad_per_hour
            if await self.app.store.ip_bucket_hit(prefix + ctx.ip_hmac, IP_WINDOW_S) > limit:
                self._reject(ErrorCode.RATE_LIMITED_IP, "per_ip")
                raise http_error(
                    ErrorCode.RATE_LIMITED_IP, "too many sessions", retry_after_s=RATE_LIMIT_RETRY_S
                )

        sid = new_session_id()
        token = secrets.token_urlsafe(16)
        url = ws_url(str(self._s.public_base_url), ctx.ws_base_host, ctx.trusted_peer, sid)
        tel = self.app.telemetry
        tel.emit(EventName.SESSION_CREATED, session_id=sid, mode=req.mode.value)
        if not voice:  # S01
            self._admit(req, sid, token, None, KeyId.NONE, a.admitted_connect_timeout_keypad_s)
            return CreateSessionResponse(
                session_id=sid,
                status="ADMITTED",
                ws_url=url,
                ws_token=token,
                expires_in_s=a.admitted_connect_timeout_keypad_s,
            )
        permit = None if self.queue else self.app.sem.try_take_stream(self.public_key())
        if permit is not None:  # S02
            self._admit(req, sid, token, permit, self.public_key(), a.admitted_connect_timeout_s)
            return CreateSessionResponse(
                session_id=sid,
                status="ADMITTED",
                ws_url=url,
                ws_token=token,
                expires_in_s=a.admitted_connect_timeout_s,
            )
        if len(self.queue) >= a.voice_queue_visible_max:  # queue holds at most 3 (IT-ADM-02)
            self._reject(ErrorCode.VOICE_UNAVAILABLE, "queue_full")
            raise http_error(ErrorCode.VOICE_UNAVAILABLE, "voice queue full")
        self.queue.append(  # S03
            QueueEntry(
                sid, hashlib.sha256(token.encode()).hexdigest(), req, self._now(), ctx.ip_hmac
            )
        )
        pos = len(self.queue)
        tel.emit(EventName.SESSION_QUEUED, session_id=sid, position=pos)
        return CreateSessionResponse(
            session_id=sid,
            status="QUEUED",
            ws_url=url,
            ws_token=token,
            queue_position=pos,
            eta_s=eta_s(pos, a.voice_eta_s, self._permits()),
            expires_in_s=a.queue_poll_timeout_s,
        )

    def _admit(
        self,
        req: CreateSessionRequest,
        sid: str,
        token: str,
        permit: PermitLike | None,
        key_id: KeyId,
        timeout_s: int,
    ) -> None:
        self.app.port.create(req, session_id=sid, ws_token=token, permit=permit, key_id=key_id)
        self._deadline[sid] = self._now() + timeout_s * 1000.0
        self.app.telemetry.emit(EventName.SESSION_ADMITTED, session_id=sid, mode=req.mode.value)

    # ----- GET /api/v1/sessions/{id} (S06) -----
    async def status(self, session_id: str, bearer: str | None) -> SessionStatusResponse:
        tok_sha = hashlib.sha256((bearer or "").encode()).hexdigest()

        def ok(sha: str) -> bool:
            return bearer is not None and hmac.compare_digest(tok_sha.encode(), sha.encode())

        for i, e in enumerate(self.queue):
            if e.session_id != session_id:
                continue
            if not ok(e.token_sha256):
                raise http_error(ErrorCode.UNAUTHORIZED, "bad token")
            e.last_poll_mono = self._now()
            if i == 0:
                permit = self.app.sem.try_take_stream(self.public_key())
                if permit is not None:
                    self.queue.pop(0)
                    a = self._s.admission
                    self.app.port.create(
                        e.req,
                        session_id=e.session_id,
                        ws_token=bearer or "",
                        permit=permit,
                        key_id=self.public_key(),
                    )
                    self._deadline[e.session_id] = self._now() + a.admitted_connect_timeout_s * 1e3
                    self.app.telemetry.emit(
                        EventName.SESSION_ADMITTED, session_id=e.session_id, mode=Mode.VOICE.value
                    )
                    return SessionStatusResponse(
                        session_id=session_id, status="ADMITTED", queue_position=None, eta_s=None
                    )
            pos = i + 1
            return SessionStatusResponse(
                session_id=session_id,
                status="QUEUED",
                queue_position=pos,
                eta_s=eta_s(pos, self._s.admission.voice_eta_s, self._permits()),
            )
        exp = self._expired.get(session_id)
        if exp is not None:
            if not ok(exp.token_sha256):
                raise http_error(ErrorCode.UNAUTHORIZED, "bad token")
            return SessionStatusResponse(
                session_id=session_id, status="EXPIRED", queue_position=None, eta_s=None
            )
        h = self.app.port.get(session_id)
        if h is None:
            raise http_error(ErrorCode.NOT_FOUND, "unknown session")
        if bearer is None or not h.token_ok(bearer):
            raise http_error(ErrorCode.UNAUTHORIZED, "bad token")
        st = h.state
        status = (
            "ENDED"
            if st is SessionState.ENDED
            else "ADMITTED"
            if st is SessionState.AWAIT_START and not h.connected
            else "ACTIVE"
        )
        return SessionStatusResponse(
            session_id=session_id, status=status, queue_position=None, eta_s=None
        )

    # ----- sweeper (every 1 s) -----
    async def sweep(self) -> None:
        now = self._now()
        poll_ms = self._s.admission.queue_poll_timeout_s * 1000.0
        keep: list[QueueEntry] = []
        for e in self.queue:  # S07
            if now - e.last_poll_mono > poll_ms:
                self._expired[e.session_id] = _Expired(e.token_sha256)
                self.app.telemetry.emit(
                    EventName.SESSION_ENDED,
                    session_id=e.session_id,
                    end_reason=EndReason.QUEUE_ABANDONED.value,
                )
            else:
                keep.append(e)
        self.queue = keep
        for sid, deadline in list(self._deadline.items()):  # S14
            h = self.app.port.get(sid)
            if h is None or h.connected or h.state is not SessionState.AWAIT_START:
                self._deadline.pop(sid, None)
            elif now > deadline:
                self._deadline.pop(sid, None)
                await self.app.port.end(sid, EndReason.QUEUE_ABANDONED)
        for h in list(self.app.port.all()):  # registry: drop 5 s after ENDED
            if h.state is SessionState.ENDED:
                seen = self._ended_seen.setdefault(h.session_id, now)
                if now - seen >= ENDED_KEEP_MS:
                    self.app.port.remove(h.session_id)
                    self._ended_seen.pop(h.session_id, None)
                    self._socket_bucket.pop(h.session_id, None)
        await self.level()

    # ----- pre-start sockets (S13) -----
    def pre_start_count(self, bucket: str | None) -> int:
        n = 0
        for sid, b in self._socket_bucket.items():
            h = self.app.port.get(sid)
            if h is None or h.state is not SessionState.AWAIT_START:
                continue
            if bucket is None or b == bucket:
                n += 1
        return n

    def pre_start_cap(self, bucket: str | None) -> int:
        per_ip = self._s.admission.pre_start_sockets_per_ip
        return per_ip if bucket is not None else 10 * per_ip

    def socket_opened(self, session_id: str, bucket: str | None) -> None:
        self._socket_bucket[session_id] = bucket
        self._deadline.pop(session_id, None)

    # ----- GET /api/v1/status (cached 5 s) -----
    async def public_status(self) -> PublicStatusResponse:
        now = self._now()
        if self._status_cache is not None and now - self._status_cache[0] < STATUS_CACHE_MS:
            return self._status_cache[1]
        from sevi import __version__

        r = self._s.result
        offered = await self.voice_offered()
        a = self._s.admission
        resp = PublicStatusResponse(
            admission_level=await self.level(),
            voice_offered=offered,
            voice_queue_len=len(self.queue),
            voice_eta_s=eta_s(len(self.queue) + 1, a.voice_eta_s, self._permits())
            if offered
            else None,
            languages=tuple(
                LangStatus(
                    lang=lang,
                    experimental=lang in r.experimental_langs,
                    available=lang in self.app.banks,
                )
                for lang in r.enabled_langs
            ),
            phone_pilot=False,
            version=__version__,
        )
        self._status_cache = (now, resp)
        return resp
