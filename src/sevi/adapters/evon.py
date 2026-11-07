"""Evon client: llama-server's OpenAI-compatible API (B spec §5.6.2, contracts §8.5.1-§8.5.2).

Self-hosted, so no Gnani credits; protected by the Evon breaker instead. Never retries (the one
regenerate lives in `app/explain.py`). Total timeout `timeout_s` (6 s) under `asyncio.timeout`;
`complete_json(..., timeout_s=...)` lets the caller pass the remaining browser deadline (10 s).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from sevi.adapters.deps import BreakerLike, secret
from sevi.contracts.adapters import Clock, LlmRequest, LlmResponse, Telemetry
from sevi.contracts.errors import BreakerOpenError, ErrorCode, LlmError
from sevi.contracts.telemetry import EventName

NO_DIGITS_PATTERN = "^[^0-9०-९௦-௲೦-೯]*$"  # noqa: RUF001
MAX_LEN = {"EXPLAIN_SPOKEN": 300, "REPORT_PROSE": 600, "TRANSLATION_DRAFT": 1000}
SCHEMA_NAME = {
    "EXPLAIN_SPOKEN": "kj_explain_spoken",
    "REPORT_PROSE": "kj_report_prose",
    "TRANSLATION_DRAFT": "kj_translation_draft",
}
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class EvonOptions:
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: object | None = None
    model: str = "evon"
    timeout_s: float = 6.0
    connect_timeout_s: float = 1.0
    top_p: float = 0.95
    repeat_penalty: float = 1.0
    response_format_style: Literal["llamacpp", "openai"] = "llamacpp"
    schema_pattern: bool = True

    @classmethod
    def from_settings(cls, s: Any) -> EvonOptions:
        """`s` = `Settings.evon` (EvonSettings)."""
        return cls(
            base_url=str(s.base_url).rstrip("/"),
            api_key=s.api_key,
            model=s.model,
            timeout_s=s.timeout_s,
            connect_timeout_s=s.connect_timeout_s,
            top_p=s.top_p,
            repeat_penalty=s.repeat_penalty,
            response_format_style=s.response_format_style,
            schema_pattern=s.schema_pattern,
        )


def schema_for(purpose: str, pattern: bool) -> dict[str, Any]:
    text: dict[str, Any] = {"type": "string", "minLength": 20, "maxLength": MAX_LEN[purpose]}
    if pattern and purpose != "TRANSLATION_DRAFT":
        text["pattern"] = NO_DIGITS_PATTERN
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": text},
    }


def check_schema(obj: object, schema: dict[str, Any]) -> str | None:
    """Hand check (no jsonschema dependency): returns the failure or None."""
    if not isinstance(obj, dict):
        return "not an object"
    props: dict[str, Any] = schema.get("properties", {})
    for k in schema.get("required", []):
        if k not in obj:
            return f"missing {k}"
    if schema.get("additionalProperties") is False and set(obj) - set(props):
        return "extra keys"
    for k, rule in props.items():
        if k not in obj:
            continue
        v = obj[k]
        if rule.get("type") == "string":
            if not isinstance(v, str):
                return f"{k} not a string"
            if len(v) < rule.get("minLength", 0) or len(v) > rule.get("maxLength", 1 << 30):
                return f"{k} length {len(v)}"
            pat = rule.get("pattern")
            if pat is not None and re.fullmatch(pat, v, flags=re.DOTALL) is None:
                return f"{k} pattern"
    return None


class EvonClient:
    """Implements contracts §5.2 `LLMClient`."""

    def __init__(
        self,
        opts: EvonOptions,
        http: httpx.AsyncClient,
        breaker: BreakerLike,
        clock: Clock,
        telemetry: Telemetry,
    ) -> None:
        self.opts = opts
        self.http = http
        self.breaker = breaker
        self.clock = clock
        self.telemetry = telemetry

    def build_body(self, req: LlmRequest) -> dict[str, Any]:
        schema = req.json_schema or schema_for(req.purpose, self.opts.schema_pattern)
        rf: dict[str, Any]
        if self.opts.response_format_style == "openai":
            rf = {
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME[req.purpose], "strict": True, "schema": schema},
            }
        else:
            rf = {"type": "json_schema", "schema": schema}
        return {
            "model": self.opts.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "temperature": req.temperature,
            "top_p": self.opts.top_p,
            "repeat_penalty": self.opts.repeat_penalty,
            "max_tokens": req.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": rf,
        }

    def _headers(self) -> dict[str, str]:
        if self.opts.api_key is None:
            return {}
        return {"Authorization": f"Bearer {secret(self.opts.api_key)}"}

    async def complete_json(
        self, req: LlmRequest, *, timeout_s: float | None = None
    ) -> LlmResponse:
        if self.breaker.is_open():
            raise BreakerOpenError(ErrorCode.LLM_BREAKER_OPEN, "evon breaker open")
        total = (
            min(self.opts.timeout_s, timeout_s) if timeout_s is not None else self.opts.timeout_s
        )
        body = self.build_body(req)
        schema = (
            body["response_format"].get("schema")
            or body["response_format"]["json_schema"]["schema"]
        )
        t0 = self.clock.mono_ms()
        self.telemetry.emit(EventName.EVON_REQUEST, "DEBUG", purpose=req.purpose, lang=req.lang)
        try:
            async with asyncio.timeout(total):
                r = await self.http.post(
                    f"{self.opts.base_url.rstrip('/')}/chat/completions",
                    json=body,
                    headers=self._headers(),
                    timeout=httpx.Timeout(total, connect=self.opts.connect_timeout_s),
                )
        except (TimeoutError, httpx.TimeoutException):
            self.breaker.record_failure()
            raise LlmError(ErrorCode.LLM_TIMEOUT, f"evon timed out after {total:g} s") from None
        except httpx.TransportError as e:
            self.breaker.record_failure()
            raise LlmError(ErrorCode.LLM_UNAVAILABLE, f"evon: {type(e).__name__}") from None
        latency = self.clock.mono_ms() - t0
        if r.status_code != 200:
            if r.status_code >= 500:
                self.breaker.record_failure()
            raise LlmError(
                ErrorCode.LLM_UNAVAILABLE,
                f"evon HTTP {r.status_code}",
                details={"status": r.status_code},
            )
        try:
            payload = r.json()
            choice = payload["choices"][0]
            content = str(choice["message"]["content"] or "")
            finish = choice.get("finish_reason")
            usage = payload.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            raise LlmError(ErrorCode.LLM_SCHEMA_INVALID, "malformed completion") from None
        raw = _THINK.sub("", content, count=1)
        if finish == "length":
            raise LlmError(ErrorCode.LLM_SCHEMA_INVALID, "finish_reason=length")
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise LlmError(ErrorCode.LLM_SCHEMA_INVALID, "content is not JSON") from None
        why = check_schema(parsed, schema)
        if why is not None:
            raise LlmError(ErrorCode.LLM_SCHEMA_INVALID, f"schema: {why}")
        tokens = usage.get("completion_tokens")
        self.telemetry.emit(
            EventName.EVON_RESPONSE,
            "DEBUG",
            purpose=req.purpose,
            latency_ms=latency,
            tokens_out=tokens,
        )
        return LlmResponse(
            raw_text=raw,
            parsed=parsed,
            tokens_out=tokens if isinstance(tokens, int) else None,
            latency_ms=latency,
            finish_reason=finish,
        )

    async def health(self) -> bool:
        base = self.opts.base_url.rstrip("/")
        base = base[: -len("/v1")] if base.endswith("/v1") else base
        try:
            async with asyncio.timeout(2.0):
                r = await self.http.get(f"{base}/health", headers=self._headers(), timeout=2.0)
            ok = r.status_code == 200 and (r.json() or {}).get("status") == "ok"
        except (TimeoutError, httpx.HTTPError, ValueError, AttributeError):
            ok = False
        self.telemetry.emit(EventName.EVON_HEALTH, "DEBUG", ok=ok)
        return ok
