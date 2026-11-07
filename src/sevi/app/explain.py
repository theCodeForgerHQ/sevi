"""Evon explanation prose (contracts §8.5.3; B spec §11.2, M10).

Code decides the result; Evon only rephrases the code-selected wording. The Evon input is the
band and the pre-written wording only (no transcripts, digits, SRT, names: U-EVN-06). Every
output passes ``core.guard.check``; one regenerate with the rejection suffix, then TEMPLATE.
Hard deadline 10 s from the call. ``SEVI_EVON__ENABLED=false`` -> TEMPLATE without a request.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import yaml

from sevi.contracts.adapters import Clock, LLMClient, LlmRequest
from sevi.contracts.copy import CopyFile
from sevi.contracts.enums import Band, Channel, EvonOutcome, Lang, Surface
from sevi.contracts.errors import BreakerOpenError, ErrorCode, LlmError
from sevi.contracts.result import ResultDecision
from sevi.core import guard

ELIGIBLE_BANDS: Final = frozenset(
    {Band.PASS, Band.PASS_WITH_NOTE, Band.BORDERLINE, Band.REFER, Band.NO_BAND, Band.INCONCLUSIVE}
)
SEMAPHORE_MARGIN_MS: Final = 500
REGENERATE_MIN_MS: Final = 3_000
CALL_MARGIN_S: Final = 0.2
SYSTEM_KEY: Final = "evon.system.explain"
TEMPLATE_KEY: Final = "evon.template.explain"  # optional; absent -> no extra text
# Exact copy sentences the guard lets through MEDICAL/REASSURANCE/BANNED (our own disclaimer).
ALLOWED_KEYS: Final = ("disclaimer.page", "prompt.disclaimer")

Purpose = Literal["EXPLAIN_SPOKEN"]
Cause = Literal[
    "disabled",
    "breaker_open",
    "health_down",
    "semaphore",
    "timeout",
    "unavailable",
    "schema_invalid",
    "guard_rejected",
    "deadline",
]


def explain_schema(max_length: int = 300, pattern: bool = True) -> dict[str, Any]:
    """Contracts §8.5.1 schema; ``pattern`` excludes ASCII, Devanagari, Tamil, Kannada digits."""
    text: dict[str, Any] = {"type": "string", "minLength": 20, "maxLength": max_length}
    if pattern:
        text["pattern"] = "^[^0-9\u0966-\u096f\u0be6-\u0bf2\u0ce6-\u0cef]*$"
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["text"],
        "properties": {"text": text},
    }


# ---------- ports consumed by the explainer ----------
class CopySource(Protocol):
    """Read-only view of A's copy table (``src/sevi/copy/copy.yaml``, contracts §4.11)."""

    @property
    def banned(self) -> tuple[str, ...]: ...

    def text(self, key: str, lang: Lang) -> str | None: ...


class WordingSource(Protocol):
    """A's wording classifier (``core/wording.py``): the code-selected text for a surface."""

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> str: ...


class Breaker(Protocol):
    def is_open(self) -> bool: ...


class Emitter(Protocol):
    def emit(self, event: str, level: str = "INFO", **fields: Any) -> None: ...


class CopyTable:
    """``CopySource`` over a ``CopyFile``: first variant whose ``when`` matches the outcomes."""

    def __init__(self, copy: CopyFile, outcomes: Mapping[str, str] | None = None) -> None:
        self._copy = copy
        self._outcomes = dict(outcomes or {})
        self._by_key = {e.key: e for e in copy.entries}

    @classmethod
    def load(cls, path: Path, outcomes: Mapping[str, str] | None = None) -> CopyTable:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(CopyFile.model_validate(data), outcomes)

    @property
    def banned(self) -> tuple[str, ...]:
        return self._copy.banned

    def text(self, key: str, lang: Lang) -> str | None:
        entry = self._by_key.get(key)
        if entry is None:
            return None
        for v in entry.variants:
            if all(self._outcomes.get(k.value) == val for k, val in v.when.items()):
                return v.text.get(lang.value)
        return None


@dataclass(frozen=True)
class ExplainDone:
    evon: EvonOutcome
    text: str | None  # LIVE: guarded Evon text; TEMPLATE: copy template or None (wording only)
    cause: Cause | None
    calls: int


@dataclass(frozen=True)
class ExplainConfig:
    enabled: bool = True
    deadline_ms: float = 10_000.0
    timeout_s: float = 6.0
    max_tokens: int = 256
    temperature: float = 0.6
    max_concurrency: int = 1
    schema_pattern: bool = True


class _NullEmitter:
    def emit(self, event: str, level: str = "INFO", **fields: Any) -> None:
        return None


class Explainer:
    def __init__(
        self,
        llm: LLMClient,
        copy: CopySource,
        wording: WordingSource,
        clock: Clock,
        cfg: ExplainConfig | None = None,
        breaker: Breaker | None = None,
        telemetry: Emitter | None = None,
    ) -> None:
        self._llm = llm
        self._copy = copy
        self._wording = wording
        self._clock = clock
        self._cfg = cfg or ExplainConfig()
        self._breaker = breaker
        self._tm: Emitter = telemetry or _NullEmitter()
        self._sem = asyncio.Semaphore(self._cfg.max_concurrency)
        self.healthy = True  # last evon.health result (updated by the health loop)

    def eligible(self, channel: Channel, decision: ResultDecision, evon_switch: bool) -> bool:
        """C§6.6: switch on, browser, breaker closed, explainable band. Else evon = SKIPPED."""
        return (
            evon_switch
            and self._cfg.enabled
            and channel == Channel.BROWSER
            and not (self._breaker is not None and self._breaker.is_open())
            and decision.band in ELIGIBLE_BANDS
        )

    async def health_tick(self) -> bool:
        try:
            self.healthy = await self._llm.health()
        except Exception:
            self.healthy = False
        self._tm.emit("evon.health", level="DEBUG", ok=self.healthy)
        return self.healthy

    def _template(self, cause: Cause, lang: Lang, calls: int) -> ExplainDone:
        self._tm.emit("evon.template", cause=cause)
        text = self._copy.text(TEMPLATE_KEY, lang)
        return ExplainDone(EvonOutcome.TEMPLATE, text, cause, calls)

    def _remaining_ms(self, deadline: float) -> float:
        return deadline - self._clock.mono_ms()

    async def explain(self, decision: ResultDecision, lang: Lang) -> ExplainDone:
        cfg = self._cfg
        if not cfg.enabled:
            return self._template("disabled", lang, 0)
        if self._breaker is not None and self._breaker.is_open():
            return self._template("breaker_open", lang, 0)
        if not self.healthy:
            return self._template("health_down", lang, 0)
        deadline = self._clock.mono_ms() + cfg.deadline_ms
        wait_s = (self._remaining_ms(deadline) - SEMAPHORE_MARGIN_MS) / 1000.0
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=max(wait_s, 0.0))
        except TimeoutError:
            return self._template("semaphore", lang, 0)
        try:
            return await self._run(decision, lang, deadline)
        finally:
            self._sem.release()

    def _user_text(self, decision: ResultDecision, lang: Lang) -> str:
        surface = Surface.SPOKEN if Surface.SPOKEN in decision.surfaces else Surface.PAGE
        wording = self._wording.wording(decision, lang, surface)
        # Only band + pre-written wording ever enter the prompt (U-EVN-06).
        return json.dumps({"band": decision.band.value, "wording": wording}, ensure_ascii=False)

    async def _run(self, decision: ResultDecision, lang: Lang, deadline: float) -> ExplainDone:
        cfg = self._cfg
        system = self._copy.text(SYSTEM_KEY, lang)
        if not system:
            return self._template("disabled", lang, 0)
        user = self._user_text(decision, lang)
        schema = explain_schema(300, cfg.schema_pattern)
        purpose: Purpose = "EXPLAIN_SPOKEN"
        calls = 0
        last_cause: Cause = "deadline"
        for attempt in (1, 2):
            remaining = self._remaining_ms(deadline)
            if attempt == 2 and remaining < REGENERATE_MIN_MS:
                return self._template(last_cause, lang, calls)
            call_timeout = min(cfg.timeout_s, remaining / 1000.0 - CALL_MARGIN_S)
            if call_timeout <= 0:
                return self._template("deadline", lang, calls)
            req = LlmRequest(
                purpose=purpose,
                lang=lang,
                system=system,
                user=user,
                json_schema=schema,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            calls += 1
            self._tm.emit("evon.request", purpose=purpose, attempt=attempt)
            try:
                resp = await asyncio.wait_for(self._llm.complete_json(req), timeout=call_timeout)
            except TimeoutError:
                return self._template("timeout", lang, calls)
            except BreakerOpenError:
                return self._template("breaker_open", lang, calls)
            except LlmError as exc:
                if exc.code == ErrorCode.LLM_TIMEOUT:
                    return self._template("timeout", lang, calls)
                if exc.code != ErrorCode.LLM_SCHEMA_INVALID:
                    return self._template("unavailable", lang, calls)
                last_cause = "schema_invalid"
                system = _with_suffix(system, ["SCHEMA_INVALID"])
                continue
            self._tm.emit(
                "evon.response",
                latency_ms=resp.latency_ms,
                tokens_out=resp.tokens_out,
                finish_reason=resp.finish_reason,
            )
            text = resp.parsed.get("text")
            if resp.finish_reason == "length" or not isinstance(text, str):
                # adapters already map these to LLM_SCHEMA_INVALID; defensive (R-21)
                last_cause = "schema_invalid"
                system = _with_suffix(system, ["SCHEMA_INVALID"])
                continue
            allowed = [a for k in ALLOWED_KEYS if (a := self._copy.text(k, lang))]
            verdict = guard.check(text, lang, purpose, banned=self._copy.banned, allowed=allowed)
            if verdict.ok:
                return ExplainDone(EvonOutcome.LIVE, text, None, calls)
            self._tm.emit("evon.guard_rejected", reasons=[r.value for r in verdict.reasons])
            last_cause = "guard_rejected"
            system = _with_suffix(system, [r.value for r in verdict.reasons])
        return self._template(last_cause, lang, calls)


def _with_suffix(system: str, reasons: list[str]) -> str:
    """C§8.5.3 rejection suffix appended to the original system text (only once)."""
    base = system.split("\nYour previous answer was rejected because:")[0]
    return f"{base}\nYour previous answer was rejected because: {', '.join(reasons)}."
