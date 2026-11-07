"""Timbre TTS client (B spec §5.5, contracts §8.4, §8.8).

`TimbreClient` implements contracts §5.2 `TTSClient`. Every billed request is
`ledger.billed(key, purpose, tts_chars=len(text))` (refused with `BudgetExceeded` before any
I/O) -> `sem.acquire(key, "tts")` -> POST with up to 5 retries (full jitter) on 5xx/timeouts.
A cache hit (`cache_dir/timbre/<sha256(canonical request)>.bin`) makes no request and bills
nothing. Text over `max_chars` (300) raises `TTS_TEXT_TOO_LONG` before anything else.

`offline_client()` is the factory `python -m sevi.calibration.render` uses (A's PR #32): a
ledger-guarded client bound to Key A, configured from `Settings` (env / Infisical at run time).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from sevi.adapters.deps import BillingLedger, PermitSemaphore, Purpose, SerialSemaphore, secret
from sevi.contracts.adapters import Clock, Telemetry
from sevi.contracts.enums import KeyId, Lang
from sevi.contracts.errors import ErrorCode, TtsError
from sevi.contracts.telemetry import EventName
from sevi.core import dsp

TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
MAX_CHARS = 300
CONNECT_TIMEOUT_S = 5.0
TOTAL_TIMEOUT_S = 20.0
RETRIES = 5
PERMIT_TIMEOUT_S = 30.0
WAV_AUDIO_CONFIG: dict[str, Any] = {
    "sample_rate": 48000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}
ULAW_FORMS: dict[str, dict[str, Any]] = {
    "mulaw": {"container": "mulaw"},
    "raw_pcm_mulaw": {"sample_rate": 8000, "encoding": "pcm_mulaw", "container": "raw"},
}


def canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def request_body(
    text: str, lang: Lang, voice: str, model: str, audio_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Exact §5.5 body; `language` = f"{lang}-IN"."""
    return {
        "text": text,
        "voice": voice,
        "model": model,
        "language": f"{lang.value}-IN",
        "speed": 1.0,
        "audio_config": dict(audio_config),
    }


def wav_to_ulaw(data: bytes) -> bytes:
    """RIFF WAV (8/16/48 kHz) -> 8 kHz mu-law through core.dsp."""
    x, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    if sr == 48000:
        x = dsp.resample_48k_to_16k(x)
        sr = 16000
    if sr == 16000:
        x = dsp.resample_16k_to_8k(x)
    elif sr != 8000:
        raise TtsError(ErrorCode.TTS_FAILED, f"unexpected WAV rate {sr}")
    return dsp.ulaw_encode(dsp.float_to_pcm16(x))


class TimbreClient:
    def __init__(
        self,
        keys: Mapping[KeyId, object],
        sem: PermitSemaphore,
        ledger: BillingLedger,
        http: httpx.AsyncClient,
        clock: Clock,
        telemetry: Telemetry,
        *,
        cache_dir: Path | None = None,
        url: str = TTS_URL,
        purpose: Purpose = "prompt_render",
        ulaw_form: Literal["mulaw", "raw_pcm_mulaw"] = "mulaw",
        max_chars: int = MAX_CHARS,
        retries: int = RETRIES,
        timeout_s: float = TOTAL_TIMEOUT_S,
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
        self.ulaw_form = ulaw_form
        self.max_chars = max_chars
        self.retries = retries
        self.timeout_s = timeout_s
        self.rng = rng or random.Random()  # noqa: S311 (backoff jitter, not crypto)

    # ---------- public API (contracts §5.2 TTSClient) ----------
    async def synthesize_ulaw(
        self, text: str, lang: Lang, voice: str, model: str, key_id: KeyId
    ) -> bytes:
        self._check_len(text)
        body = request_body(text, lang, voice, model, ULAW_FORMS[self.ulaw_form])
        raw = await self._request(body, key_id)
        ulaw = wav_to_ulaw(raw) if raw[:4] == b"RIFF" else raw
        return dsp.normalize_ulaw(ulaw, -20.0, -3.0)

    async def synthesize_wav(
        self, text: str, lang: Lang, voice: str, model: str, sample_rate_hz: int = 48000
    ) -> bytes:
        self._check_len(text)
        cfg = {**WAV_AUDIO_CONFIG, "sample_rate": sample_rate_hz}
        raw = await self._request(request_body(text, lang, voice, model, cfg), KeyId.A)
        if raw[:4] != b"RIFF":
            raise TtsError(ErrorCode.TTS_FAILED, "expected a RIFF WAV body")
        return raw

    # ---------- internals ----------
    def _check_len(self, text: str) -> None:
        if len(text) > self.max_chars:
            raise TtsError(
                ErrorCode.TTS_TEXT_TOO_LONG,
                f"text has {len(text)} chars > {self.max_chars}",
                details={"chars": len(text)},
            )

    def _cache_path(self, body: Mapping[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(canonical_json(body)).hexdigest()
        return self.cache_dir / "timbre" / f"{key}.bin"

    async def _request(self, body: dict[str, Any], key_id: KeyId) -> bytes:
        path = self._cache_path(body)
        if path is not None and path.exists():
            return path.read_bytes()
        chars = len(body["text"])
        # reserve BEFORE any I/O: BudgetExceeded propagates and nothing is sent
        async with self.ledger.billed(key_id, self.purpose, tts_chars=chars) as bill:
            async with self.sem.acquire(key_id, "tts", PERMIT_TIMEOUT_S):
                data = await self._post_with_retries(body, key_id, bill)
            bill.actual(tts_chars=chars)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return data

    async def _post_with_retries(self, body: dict[str, Any], key_id: KeyId, bill: Any) -> bytes:
        headers = {"X-API-Key-ID": secret(self.keys[key_id]), "Content-Type": "application/json"}
        timeout = httpx.Timeout(self.timeout_s, connect=CONNECT_TIMEOUT_S)
        last: TtsError | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                delay = self.rng.uniform(0, min(8.0, 0.5 * 2 ** (attempt - 1)))
                await self.clock.sleep_until(self.clock.mono_ms() + delay * 1000.0)
            t0 = self.clock.mono_ms()
            try:
                async with asyncio.timeout(self.timeout_s):
                    r = await self.http.post(
                        self.url, content=canonical_json(body), headers=headers, timeout=timeout
                    )
            except (TimeoutError, httpx.TimeoutException):
                last = TtsError(ErrorCode.TTS_TIMEOUT, "timbre timed out")
                continue
            except httpx.TransportError as e:
                last = TtsError(ErrorCode.TTS_FAILED, f"timbre transport: {type(e).__name__}")
                continue
            self.telemetry.emit(
                EventName.GNANI_REST_CALL,
                "DEBUG",
                dep="timbre",
                status=r.status_code,
                attempt=attempt + 1,
                chars=len(body["text"]),
                dur_ms=self.clock.mono_ms() - t0,
            )
            st = r.status_code
            if st == 200:
                if not r.content:
                    last = TtsError(ErrorCode.TTS_FAILED, "empty body", details={"status": st})
                    bill.not_billed()  # nothing usable came back
                    raise last
                return r.content
            if st == 400:
                bill.not_billed()
                raise TtsError(ErrorCode.TTS_BAD_REQUEST, _msg(r), details={"status": st})
            if st == 402:
                bill.not_billed()
                self.ledger.mark_exhausted(key_id)
                raise TtsError(ErrorCode.TTS_CREDITS, "credits exhausted", details={"status": st})
            if st in (401, 403):
                bill.not_billed()
                raise TtsError(ErrorCode.TTS_FAILED, "timbre auth", details={"status": st})
            last = TtsError(ErrorCode.TTS_FAILED, _msg(r), details={"status": st})
            if st < 500:
                bill.not_billed()
                raise last
        assert last is not None
        raise last


def _msg(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return f"HTTP {r.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("message", ""))[:200]
        return str(body.get("message", f"HTTP {r.status_code}"))[:200]
    return f"HTTP {r.status_code}"


# ---------- offline factory (A's calibration render, PR #32) ----------
class _LazyLedgerTimbre:
    """Opens the persistent ledger on first use (it needs an event loop), then delegates."""

    def __init__(self, build: Any) -> None:
        self._build = build
        self._client: TimbreClient | None = None

    async def _get(self) -> TimbreClient:
        if self._client is None:
            self._client = await self._build()
        return self._client

    async def synthesize_wav(
        self, text: str, lang: Lang, voice: str, model: str, sample_rate_hz: int = 48000
    ) -> bytes:
        return await (await self._get()).synthesize_wav(text, lang, voice, model, sample_rate_hz)

    async def synthesize_ulaw(
        self, text: str, lang: Lang, voice: str, model: str, key_id: KeyId
    ) -> bytes:
        return await (await self._get()).synthesize_ulaw(text, lang, voice, model, key_id)


def offline_client(
    *,
    settings: Any = None,
    ledger: BillingLedger | None = None,
    sem: PermitSemaphore | None = None,
    http: httpx.AsyncClient | None = None,
    clock: Clock | None = None,
    telemetry: Telemetry | None = None,
    purpose: Purpose = "calibration",
    cache_dir: Path | None = None,
) -> _LazyLedgerTimbre:
    """Ledger-guarded Timbre client on Key A for offline renders (`synthesize_wav`).

    Defaults: `Settings()` (reads `SEVI_GNANI__KEY_A` and `SEVI_LEDGER__*` from the environment,
    i.e. `infisical run`), the persistent SQLite ledger at `settings.sqlite_path`, a 1-permit
    semaphore. Every argument can be injected (tests pass a MemoryLedger and a fake transport).
    Raises `TtsError(TTS_FAILED)` at first use if Key A is not configured.
    """

    async def build() -> TimbreClient:
        from sevi.fakes.null_telemetry import NullTelemetry

        s = settings
        if s is None:
            from sevi.app.settings import Settings  # type: ignore[import-not-found,unused-ignore]

            s = Settings()
        key_a = getattr(getattr(s, "gnani", None), "key_a", None)
        if key_a is None:
            raise TtsError(ErrorCode.TTS_FAILED, "SEVI_GNANI__KEY_A is not set")
        clk: Clock
        if clock is None:
            from sevi.app.clock import SystemClock  # type: ignore[import-not-found,unused-ignore]

            clk = SystemClock()
        else:
            clk = clock
        tel = telemetry or NullTelemetry()
        led = ledger
        if led is None:
            from sevi.adapters.ledger import Ledger  # type: ignore[import-not-found,unused-ignore]

            led = await Ledger(
                s.sqlite_path, s.ledger, alias_b=True, clock=clk, telemetry=tel
            ).open()
        if sem is None:
            sm: PermitSemaphore = SerialSemaphore()  # single-process CLI: one call per key
        else:
            sm = sem
        rest_base = str(getattr(s.gnani, "rest_base", "https://api.vachana.ai")).rstrip("/")
        return TimbreClient(
            {KeyId.A: key_a},
            sm,
            led,
            http or httpx.AsyncClient(follow_redirects=False, trust_env=False),
            clk,
            tel,
            cache_dir=cache_dir,
            url=f"{rest_base}/api/v1/tts/inference",
            purpose=purpose,
        )

    return _LazyLedgerTimbre(build)
