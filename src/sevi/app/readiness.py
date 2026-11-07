"""Readiness: /healthz, /readyz, /op/api/readyz, /version (B spec §9.3; contracts §9.1, §9.2).

`down` iff the startup retention pass has not finished, no bank loaded, the store health write
failed, or the process is draining. Else `degraded` iff the admission level is >= L1, a
language is unavailable, a Prisma breaker is open, Evon is switched on but its breaker is open,
or a startup factory came from `fakes/stub_core.py`. Else `ok`. Refreshed every 30 s and on
demand when older than 30 s; `readyz.status` on change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from sevi import __version__
from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.enums import AdmissionLevel, KeyId
from sevi.contracts.telemetry import EventName
from sevi.contracts.wire import PublicReadyz, ReadyzResponse, VersionResponse

if TYPE_CHECKING:
    from sevi.app.state import AppState

Status = Literal["ok", "degraded", "down"]
REFRESH_MS: Final = 30_000.0
HEALTH_TTL_S: Final = 60


class Readiness:
    def __init__(self, app: AppState) -> None:
        self.app = app
        self._status: Status | None = None
        self._at: float | None = None
        self._store_ok = True

    def is_down(self) -> bool:
        a = self.app
        return a.draining or not a.startup_retention_done or not a.banks or not self._store_ok

    async def _store_health(self) -> bool:
        try:
            await self.app.store.idempotency_put(
                "health", "ping", self.app.clock.now_utc().isoformat(), HEALTH_TTL_S
            )
        except Exception:
            return False
        return True

    def _prisma_open(self) -> bool:
        keys = [KeyId.A] if self.app.settings.key_b_aliased else [KeyId.A, KeyId.B]
        return any(self.app.breakers.get("prisma", k).is_open() for k in keys)

    def _evon(self) -> Literal["up", "down", "disabled"]:
        if not self.app.switches.evon:
            return "disabled"
        return "down" if self.app.breakers.get("evon", None).is_open() else "up"

    async def refresh(self, *, force: bool = True) -> Status:
        now = self.app.clock.mono_ms()
        fresh = self._at is not None and now - self._at < REFRESH_MS
        if not force and self._status is not None and fresh and not self.is_down():
            return self._status
        self._store_ok = await self._store_health()
        status: Status
        if self.is_down():
            status = "down"
        else:
            lvl = await self.app.admission.level()
            degraded = (
                lvl is not AdmissionLevel.L0
                or bool(self.app.languages_unavailable)
                or self._prisma_open()
                or self._evon() == "down"
                or bool(self.app.stub_factories)
            )
            status = "degraded" if degraded else "ok"
        self._at = now
        if status != self._status:
            self.app.telemetry.emit(
                EventName.READYZ_STATUS,
                "INFO" if status == "ok" else "WARNING",
                status=status,
                prev=self._status,
            )
            self._status = status
        return status

    async def public(self) -> PublicReadyz:
        status = "down" if self.is_down() else await self.refresh(force=False)
        return PublicReadyz(status=status)

    async def _credits(self) -> dict[str, dict[str, float | bool]]:
        led = self.app.ledger
        shown = await led.show()
        reserve = led.cfg.reserve_frac
        out: dict[str, dict[str, float | bool]] = {}
        for key, info in shown["keys"].items():
            total = float(info["total"])
            spent = float(info["spent"]) + float(info["reserved"])
            d: dict[str, float | bool] = {
                "used_frac": spent / total if total else 0.0,
                "reserve_ok": spent < total * (1 - reserve),
                "budget_remaining": float(info["remaining"]),
                "exhausted": bool(info["exhausted"]),
            }
            d.update({f"bucket_{b}": float(v) for b, v in info["buckets"].items()})
            out[key] = d
        public = self.app.settings.public_key
        pk = led.resolve(public).value
        if pk in out:
            out[pk]["exposure_paused"] = await self.app.credit.exposure_paused()
            out[pk]["public_key_low"] = await self.app.credit.public_key_low()
        return out

    async def full(self) -> ReadyzResponse:
        a = self.app
        status = await self.refresh(force=False)
        banks: dict[str, dict[str, str | bool]] = {}
        for lang in a.settings.result.enabled_langs:
            p = a.banks.get(lang)
            if p is None:
                banks[lang.value] = {"bank_hash": "", "frozen": False, "ok": False}
            else:
                m = p.manifest
                banks[lang.value] = {"bank_hash": m.bank_hash, "frozen": m.frozen, "ok": True}
        prisma: dict[str, Literal["closed", "open", "disabled"]] = {}
        for k in (KeyId.A, KeyId.B):
            if not a.switches.prisma:
                prisma[k.value] = "disabled"
            else:
                prisma[k.value] = "open" if a.breakers.get("prisma", k).is_open() else "closed"
        sw = a.switches
        return ReadyzResponse(
            status=status,
            admission_level=await a.admission.level(),
            banks=banks,
            prisma=prisma,
            evon=self._evon(),
            twilio="disabled" if not sw.phone else "ok",
            credits=await self._credits(),
            active_sessions={
                "keypad": a.admission.active_keypad(),
                "voice": a.admission.active_voice(),
                "phone": 0,
            },
            voice_queue_len=len(a.admission.queue),
            switches={
                "evon": sw.evon,
                "prisma": sw.prisma,
                "timbre_runtime": sw.timbre_runtime,
                "phone": sw.phone,
                "browser_voice": sw.browser_voice,
                "demo_mode": sw.demo_mode,
                "friends": "ALLOWED" if sw.friends else "DISALLOWED",
            },
        )

    def version(self) -> VersionResponse:
        a = self.app
        return VersionResponse(
            app_version=__version__,
            contracts_version=CONTRACTS_VERSION,
            git_sha=a.git_sha,
            banks={lang.value: p.manifest.bank_hash for lang, p in a.banks.items()},
            prompts=dict(a.prompts_hashes),
            copy_sha256=a.copy_sha256,
        )
