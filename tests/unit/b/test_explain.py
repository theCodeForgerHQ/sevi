"""Explainer tests (B spec §11.2, §13.5 U-EVN-06/08/09, F-EVN-01..05; C§8.5.3). No real calls."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sevi.app.explain import CopyTable, ExplainConfig, Explainer
from sevi.contracts.adapters import LlmRequest, LlmResponse
from sevi.contracts.enums import Band, Channel, EvonOutcome, Lang, Surface, TruthRule, WordingClass
from sevi.contracts.errors import BreakerOpenError, ErrorCode, LlmError
from sevi.contracts.result import ResultDecision
from sevi.fakes.clock import FakeClock

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "evon"
GOOD_EN = "Your hearing in background noise looks typical today, keep listening well."
WORDING = "Your result looks typical for your age group today."


def decision(band: Band = Band.PASS, surfaces=(Surface.SPOKEN, Surface.PAGE)) -> ResultDecision:
    return ResultDecision.model_construct(
        rule=TruthRule.T1,
        band=band,
        refer_reason=None,
        no_band_reason=None,
        wording_class=next(iter(WordingClass)),
        surfaces=tuple(surfaces),
        norm_cell=None,
        cutoff_version=None,
        nh95_db=None,
        n_norm=None,
        srt_db=-8.5,
        srt_display="\u22128.5",
        ages="18-25",
    )


class Wording:
    def __init__(self) -> None:
        self.calls: list[Surface] = []

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> str:
        self.calls.append(surface)
        return WORDING


Step = Callable[[LlmRequest], Any]


class FakeLLM:
    """Scripted fake: each step returns text, raises, or sleeps (fake_evon modes)."""

    def __init__(self, *steps: Any, clock: FakeClock | None = None, healthy: bool = True) -> None:
        self.steps = list(steps)
        self.requests: list[LlmRequest] = []
        self.clock = clock
        self.healthy = healthy

    async def complete_json(self, req: LlmRequest) -> LlmResponse:
        self.requests.append(req)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, tuple) and step[0] == "advance":  # simulated elapsed time
            assert self.clock is not None
            self.clock.advance(step[1])
            step = step[2]
        if step == "hang":
            await asyncio.sleep(3600)
        finish = "length" if step == "LENGTH" else "stop"
        text = GOOD_EN if step in ("valid", "LENGTH") else step
        return LlmResponse(
            raw_text=json.dumps({"text": text}),
            parsed={"text": text},
            tokens_out=10,
            latency_ms=5.0,
            finish_reason=finish,
        )

    async def health(self) -> bool:
        return self.healthy


class Tm:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, level: str = "INFO", **fields: Any) -> None:
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [e for e, _ in self.events]

    def cause(self) -> str:
        return next(f["cause"] for e, f in self.events if e == "evon.template")


class Breaker:
    def __init__(self, open_: bool) -> None:
        self.open_ = open_

    def is_open(self) -> bool:
        return self.open_


@pytest.fixture
def copy() -> CopyTable:
    return CopyTable.load(FIX / "copy_fixture.yaml")


def make(llm: FakeLLM, copy: CopyTable, clock: FakeClock | None = None, **cfg: Any):
    tm = Tm()
    ex = Explainer(llm, copy, Wording(), clock or FakeClock(), ExplainConfig(**cfg), telemetry=tm)
    return ex, tm


@pytest.mark.asyncio
async def test_explain_live_first_try(copy: CopyTable) -> None:
    llm = FakeLLM("valid")
    ex, _ = make(llm, copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.text, done.calls) == (EvonOutcome.LIVE, GOOD_EN, 1)
    req = llm.requests[0]
    assert req.purpose == "EXPLAIN_SPOKEN"
    assert req.json_schema["properties"]["text"]["maxLength"] == 300
    assert "pattern" in req.json_schema["properties"]["text"]


@pytest.mark.asyncio
async def test_U_EVN_06_prompt_contains_only_band_and_wording(copy: CopyTable) -> None:
    llm = FakeLLM("valid")
    ex, _ = make(llm, copy)
    await ex.explain(decision(), Lang.EN)
    req = llm.requests[0]
    assert json.loads(req.user) == {"band": "PASS", "wording": WORDING}
    body = req.model_dump_json()
    assert not re.search(r"[0-9]", req.user + req.system)
    assert "8.5" not in body
    assert "18-25" not in body


@pytest.mark.asyncio
async def test_explain_uses_page_wording_without_spoken_surface(copy: CopyTable) -> None:
    w = Wording()
    ex = Explainer(FakeLLM("valid"), copy, w, FakeClock())
    await ex.explain(decision(surfaces=(Surface.PAGE,)), Lang.EN)
    assert w.calls == [Surface.PAGE]


@pytest.mark.asyncio
async def test_F_EVN_01_timeout_falls_back_to_template(copy: CopyTable) -> None:
    ex, tm = make(FakeLLM("hang"), copy, timeout_s=0.05)
    done = await ex.explain(decision(), Lang.EN)
    assert done.evon == EvonOutcome.TEMPLATE
    assert tm.cause() == "timeout"
    assert done.text == "This is a screening result, not a medical opinion."


@pytest.mark.asyncio
async def test_explain_adapter_timeout_error_no_retry(copy: CopyTable) -> None:
    llm = FakeLLM(LlmError(ErrorCode.LLM_TIMEOUT), "valid")
    ex, tm = make(llm, copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls, tm.cause()) == (EvonOutcome.TEMPLATE, 1, "timeout")


@pytest.mark.parametrize(
    ("exc", "cause"),
    [
        (LlmError(ErrorCode.LLM_UNAVAILABLE), "unavailable"),
        (BreakerOpenError(ErrorCode.LLM_BREAKER_OPEN), "breaker_open"),
    ],
)
@pytest.mark.asyncio
async def test_explain_unavailable_is_immediate_template(copy, exc, cause) -> None:
    llm = FakeLLM(exc, "valid")
    ex, tm = make(llm, copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls, tm.cause()) == (EvonOutcome.TEMPLATE, 1, cause)


@pytest.mark.asyncio
async def test_F_EVN_02_invalid_json_then_valid_regenerates_with_suffix(copy: CopyTable) -> None:
    llm = FakeLLM(LlmError(ErrorCode.LLM_SCHEMA_INVALID), "valid")
    ex, _ = make(llm, copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls) == (EvonOutcome.LIVE, 2)
    assert "rejected because" not in llm.requests[0].system
    assert "Your previous answer was rejected because: SCHEMA_INVALID." in llm.requests[1].system


@pytest.mark.asyncio
async def test_explain_invalid_json_twice_template(copy: CopyTable) -> None:
    err = LlmError(ErrorCode.LLM_SCHEMA_INVALID)
    ex, tm = make(FakeLLM(err, LlmError(ErrorCode.LLM_SCHEMA_INVALID)), copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls, tm.cause()) == (EvonOutcome.TEMPLATE, 2, "schema_invalid")


@pytest.mark.asyncio
async def test_F_EVN_03_guard_rejects_twice_template(copy: CopyTable) -> None:
    bad = "Your hearing is fine, you got seven out of ten right today."
    llm = FakeLLM(bad, bad)
    ex, tm = make(llm, copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.text, done.calls) == (
        EvonOutcome.TEMPLATE,
        copy.text("evon.template.explain", Lang.EN),
        2,
    )
    assert tm.names().count("evon.guard_rejected") == 2
    assert tm.cause() == "guard_rejected"
    assert "NUMBER_WORD" in llm.requests[1].system


@pytest.mark.asyncio
async def test_explain_guard_reject_then_ok_is_live(copy: CopyTable) -> None:
    llm = FakeLLM("Get a hearing aid soon, your result looks typical today.", "valid")
    ex, _ = make(llm, copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.text) == (EvonOutcome.LIVE, GOOD_EN)
    assert "MEDICAL" in llm.requests[1].system


@pytest.mark.asyncio
async def test_U_EVN_09_finish_reason_length_regenerates_then_template(copy: CopyTable) -> None:
    ex, tm = make(FakeLLM("LENGTH", "LENGTH"), copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls, tm.cause()) == (EvonOutcome.TEMPLATE, 2, "schema_invalid")


@pytest.mark.asyncio
async def test_U_EVN_08_deadline_blocks_second_attempt(copy: CopyTable) -> None:
    clock = FakeClock()
    llm = FakeLLM(
        ("advance", 7.5, "Seven is your score today, the result looks typical."),
        "valid",
        clock=clock,
    )
    ex, tm = make(llm, copy, clock=clock)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls) == (EvonOutcome.TEMPLATE, 1)
    assert len(llm.requests) == 1


@pytest.mark.asyncio
async def test_explain_disabled_kill_switch_no_request(copy: CopyTable) -> None:
    llm = FakeLLM("valid")
    ex, tm = make(llm, copy, enabled=False)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.calls, tm.cause()) == (EvonOutcome.TEMPLATE, 0, "disabled")
    assert llm.requests == []
    assert not ex.eligible(Channel.BROWSER, decision(), evon_switch=True)


def test_explain_kill_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from sevi.app.settings import Switches, load_settings

    monkeypatch.setenv("SEVI_EVON__ENABLED", "false")
    assert Switches.from_settings(load_settings()).evon is False


@pytest.mark.asyncio
async def test_F_EVN_04_health_down_template_zero_requests(copy: CopyTable) -> None:
    llm = FakeLLM("valid", healthy=False)
    ex, tm = make(llm, copy)
    assert await ex.health_tick() is False
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, tm.cause(), llm.requests) == (EvonOutcome.TEMPLATE, "health_down", [])


def test_F_EVN_05_eligibility(copy: CopyTable) -> None:
    ex = Explainer(FakeLLM(), copy, Wording(), FakeClock(), breaker=Breaker(True))
    assert not ex.eligible(Channel.BROWSER, decision(), evon_switch=True)
    ex = Explainer(FakeLLM(), copy, Wording(), FakeClock(), breaker=Breaker(False))
    assert ex.eligible(Channel.BROWSER, decision(), evon_switch=True)
    assert not ex.eligible(Channel.PHONE, decision(), evon_switch=True)
    assert not ex.eligible(Channel.BROWSER, decision(Band.FAILED), evon_switch=True)
    assert not ex.eligible(Channel.BROWSER, decision(), evon_switch=False)


@pytest.mark.asyncio
async def test_explain_semaphore_timeout(copy: CopyTable) -> None:
    llm = FakeLLM("hang", "valid")
    ex, tm = make(llm, copy, deadline_ms=700, timeout_s=5.0)
    first = asyncio.create_task(ex.explain(decision(), Lang.EN))
    await asyncio.sleep(0)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, tm.cause()) == (EvonOutcome.TEMPLATE, "semaphore")
    first.cancel()


@pytest.mark.asyncio
async def test_explain_disclaimer_sentence_from_copy_is_live(copy: CopyTable) -> None:
    text = "Your result looks typical today. This is a screening, not a diagnosis."
    ex, _ = make(FakeLLM(text), copy)
    done = await ex.explain(decision(), Lang.EN)
    assert (done.evon, done.text) == (EvonOutcome.LIVE, text)


# en W_REF_SRT spoken wording (src/sevi/copy/copy.yaml): no number word, <= 300 chars.
REF_SRT_EN = (
    "Your result is outside the range of our testers with self-reported normal hearing, "
    "in a quiet room. This is a screening, not a diagnosis. Please get a hearing check with "
    "an audiologist or ENT doctor."
)


class FixedWording(Wording):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> str:
        self.calls.append(surface)
        return self.text


def _make_with_wording(llm: FakeLLM, copy: CopyTable, text: str):
    tm = Tm()
    ex = Explainer(llm, copy, FixedWording(text), FakeClock(), ExplainConfig(), telemetry=tm)
    return ex, tm


@pytest.mark.asyncio
async def test_explain_approved_wording_verbatim_is_echo_template(copy: CopyTable) -> None:
    """#77: restating the wording is not rejected as MEDICAL/REASSURANCE; but a near-verbatim
    echo adds nothing next to the wording on the page, so the approved template is shown."""
    ex, tm = _make_with_wording(FakeLLM(REF_SRT_EN), copy, REF_SRT_EN)
    done = await ex.explain(decision(Band.REFER), Lang.EN)
    assert (done.evon, tm.cause(), done.calls) == (EvonOutcome.TEMPLATE, "echo", 1)


@pytest.mark.asyncio
async def test_explain_rephrased_wording_is_live(copy: CopyTable) -> None:
    said = (
        "Your score sits outside the range of the people we tested. "
        "This is a screening, not a diagnosis. Please get a hearing check with an audiologist."
    )
    ex, _ = _make_with_wording(FakeLLM(said), copy, REF_SRT_EN)
    done = await ex.explain(decision(Band.REFER), Lang.EN)
    assert (done.evon, done.text) == (EvonOutcome.LIVE, said)


@pytest.mark.asyncio
async def test_explain_approved_wording_plus_reassurance_rejected(copy: CopyTable) -> None:
    bad = f"{REF_SRT_EN} Don't worry."
    ex, tm = _make_with_wording(FakeLLM(bad, bad), copy, REF_SRT_EN)
    done = await ex.explain(decision(Band.REFER), Lang.EN)
    assert (done.evon, tm.cause()) == (EvonOutcome.TEMPLATE, "guard_rejected")
