"""Prisma REST STT client (B spec §5.4, contracts §8.3; cache policy owner A).

Flow: content-addressed cache hit -> `cached=True` (no permit, no ledger, no HTTP). Miss ->
`ledger.cal_stop()` for CAL purposes (CREDIT_STOP) -> `ledger.billed(A, purpose, stt_s=audio_s,
stt_rest_calls=1)` (refuses with `BudgetExceeded` before any I/O) -> `sem.acquire(A, "rest")` ->
multipart POST with up to `retries` full-jitter retries on 429/500/503/504/timeouts -> atomic
cache write.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import random
import struct
from collections.abc import Mapping
from datetime import UTC
from pathlib import Path
from typing import Any

import httpx

from sevi.adapters.deps import (
    CAL_PURPOSES,
    BillingLedger,
    BreakerLike,
    PermitSemaphore,
    Purpose,
    secret,
)
from sevi.contracts.adapters import Clock, SttRestRequest, SttRestResult, Telemetry
from sevi.contracts.enums import KeyId, Lang, ModelVersion
from sevi.contracts.errors import BreakerOpenError, ErrorCode, SeviError, SttError
from sevi.contracts.telemetry import EventName

REST_URL = "https://api.vachana.ai/stt/v3"
RETRY_STATUS = frozenset({429, 500, 503, 504})
CONNECT_TIMEOUT_S = 5.0
PERMIT_TIMEOUT_S = 30.0


def canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def prisma_model(mv: ModelVersion) -> str:
    return "prisma-v3.0" if mv is ModelVersion.V3 else "prisma-v2.5"


def prisma_language_code(lang: Lang, mv: ModelVersion) -> str:
    """C§4.13: ta v3 "ta-IN,en-IN", ta v2.5 "ta-IN", en (both) "en-IN"."""
    if lang is Lang.EN:
        return "en-IN"
    return f"{lang.value}-IN,en-IN" if mv is ModelVersion.V3 else f"{lang.value}-IN"


def bias_words(keyterms: tuple[str, ...]) -> list[str]:
    """v2.5 word boost: Latin single words, letters only, <= 100 (C§8.2.1 L2 decision)."""
    return [k for k in keyterms if k.isascii() and k.isalpha()][:100]


def pcm16_wav(pcm: bytes, sample_rate_hz: int) -> bytes:
    """44-byte RIFF header + PCM16 mono."""
    n = len(pcm)
    hdr = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate_hz, sample_rate_hz * 2, 2, 16)
    return hdr + b"data" + struct.pack("<I", n) + pcm


def _wav_of(req: SttRestRequest) -> bytes:
    data = req.wav_8k_pcm16
    return data if data[:4] == b"RIFF" else pcm16_wav(data, 8000)


def cache_fields(req: SttRestRequest, audio_sha256: str) -> dict[str, Any]:
    v3 = req.model_version is ModelVersion.V3
    return {
        "endpoint": "stt/v3",
        "audio_sha256": audio_sha256,
        "model": prisma_model(req.model_version),
        "language_code": prisma_language_code(req.lang, req.model_version),
        "format": req.format,
        "keyterms": list(req.keyterms) if v3 and req.keyterms else None,
        "bias": bias_words(req.keyterms) if not v3 and req.keyterms else None,
        "repeat_index": req.repeat_index,
    }


def cache_key(req: SttRestRequest, audio_sha256: str) -> str:
    """C§8.3: sha256(canonical_json({endpoint, audio_sha256, model, language_code, format,
    keyterms|null, bias|null, repeat_index}))."""
    return hashlib.sha256(canonical_json(cache_fields(req, audio_sha256))).hexdigest()


def form_fields(req: SttRestRequest) -> dict[str, str]:
    """Exact §5.4 multipart text fields (the audio part is added separately)."""
    f = {
        "model": prisma_model(req.model_version),
        "language_code": prisma_language_code(req.lang, req.model_version),
        "format": req.format,
    }
    if req.keyterms:
        if req.model_version is ModelVersion.V3:
            f["keyterms"] = json.dumps(
                list(req.keyterms), ensure_ascii=False, separators=(",", ":")
            )
        else:
            f["bias_list"] = ",".join(bias_words(req.keyterms))
            f["bias_score"] = "1.5"
    return f


class PrismaRestClient:
    """Implements contracts §5.2 `STTRestClient`."""

    def __init__(
        self,
        keys: Mapping[KeyId, object],
        sem: PermitSemaphore,
        ledger: BillingLedger,
        http: httpx.AsyncClient,
        clock: Clock,
        telemetry: Telemetry,
        *,
        cache_dir: Path | None,
        url: str = REST_URL,
        purpose: Purpose = "calibration",
        timeout_s: float = 15.0,
        rng: random.Random | None = None,
    ) -> None:
        self.keys = keys
        self.sem = sem
        self.ledger = ledger
        self.http = http
        self.clock = clock
        self.telemetry = telemetry
        self.cache_dir = cache_dir
        self.url = url
        self.purpose: Purpose = purpose
        self.timeout_s = timeout_s
        self.rng = rng or random.Random()  # noqa: S311 (backoff jitter)

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / "prisma" / f"{key}.json"

    async def transcribe(
        self,
        req: SttRestRequest,
        *,
        timeout_s: float | None = None,
        retries: int = 5,
        request_id: str | None = None,
        breaker: BreakerLike | None = None,
    ) -> SttRestResult:
        wav = _wav_of(req)
        key = cache_key(req, hashlib.sha256(wav).hexdigest())
        path = self._cache_path(key)
        if path is not None and path.exists():
            resp = json.loads(path.read_text())["response"]
            self.telemetry.emit(EventName.GNANI_REST_CALL, "DEBUG", dep="prisma", cached=True)
            return self._result(resp, key, cached=True)
        if breaker is not None and breaker.is_open():
            raise BreakerOpenError(ErrorCode.STT_BREAKER_OPEN, "prisma breaker open")
        if self.purpose in CAL_PURPOSES and await self.ledger.cal_stop():
            raise SeviError(ErrorCode.CREDIT_STOP, "Key A CAL bucket stop reached")
        audio_s = max(0, len(wav) - 44) / 2 / 8000.0
        total = timeout_s if timeout_s is not None else self.timeout_s
        async with self.ledger.billed(
            KeyId.A, self.purpose, stt_s=audio_s, stt_rest_calls=1
        ) as bill:
            async with self.sem.acquire(KeyId.A, "rest", PERMIT_TIMEOUT_S):
                try:
                    resp = await self._post(req, wav, total, retries, request_id, bill)
                except SttError:
                    if breaker is not None:
                        breaker.record_failure()
                    raise
            bill.actual(stt_s=audio_s, stt_rest_calls=1)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "request": cache_fields(req, hashlib.sha256(wav).hexdigest()),
                "response": resp,
                "scored_at": self.clock.now_utc().astimezone(UTC).isoformat(),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False))
            os.replace(tmp, path)
        return self._result(resp, key, cached=False)

    @staticmethod
    def _result(resp: Mapping[str, Any], key: str, *, cached: bool) -> SttRestResult:
        pt = resp.get("processing_time")
        return SttRestResult(
            transcript=str(resp["transcript"]),
            request_id=resp.get("request_id"),
            model=resp.get("model"),
            processing_time_s=float(pt) if isinstance(pt, int | float) else None,
            cached=cached,
            cache_key=key,
        )

    async def _post(
        self,
        req: SttRestRequest,
        wav: bytes,
        total: float,
        retries: int,
        request_id: str | None,
        bill: Any,
    ) -> dict[str, Any]:
        headers = {"X-API-Key-ID": secret(self.keys[KeyId.A])}
        if request_id:
            headers["X-API-Request-ID"] = request_id
        fields = form_fields(req)
        last = SttError(ErrorCode.STT_REST_FAILED, "no attempt made")
        for attempt in range(retries + 1):
            if attempt:
                delay = self.rng.uniform(0, min(8.0, 0.5 * 2 ** (attempt - 1)))
                await self.clock.sleep_until(self.clock.mono_ms() + delay * 1000.0)
            t0 = self.clock.mono_ms()
            try:
                async with asyncio.timeout(total):
                    r = await self.http.post(
                        self.url,
                        headers=headers,
                        data=fields,
                        files={"audio_file": ("clip.wav", io.BytesIO(wav), "audio/wav")},
                        timeout=httpx.Timeout(total, connect=min(CONNECT_TIMEOUT_S, total)),
                    )
            except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as e:
                last = SttError(ErrorCode.STT_REST_FAILED, f"prisma rest: {type(e).__name__}")
                continue
            st = r.status_code
            self.telemetry.emit(
                EventName.GNANI_REST_CALL,
                "DEBUG",
                dep="prisma",
                cached=False,
                status=st,
                attempt=attempt + 1,
                dur_ms=self.clock.mono_ms() - t0,
            )
            if st == 200:
                try:
                    body = r.json()
                except ValueError:
                    body = None
                if (
                    not isinstance(body, dict)
                    or body.get("success") is False
                    or "transcript" not in body
                ):
                    raise SttError(ErrorCode.STT_REST_FAILED, "success:false or no transcript")
                return body
            if st in (401, 403):
                bill.not_billed()
                raise SttError(ErrorCode.STT_AUTH, "prisma auth", details={"status": st})
            if st == 402:
                bill.not_billed()
                self.ledger.mark_exhausted(KeyId.A)
                raise SttError(ErrorCode.STT_CREDITS, "credits exhausted", details={"status": st})
            last = SttError(ErrorCode.STT_REST_FAILED, f"HTTP {st}", details={"status": st})
            if st not in RETRY_STATUS:
                bill.not_billed()
                raise last
        raise last
