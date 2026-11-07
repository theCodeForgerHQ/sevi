"""The real session runtime behind `SessionPort` (B spec §2.4, §7.1; contracts §6).

`OrchestratorPort` adapts `Orchestrator`/`BrowserSession` to the small surface admission, the WS
route and the operator API use. `build_orchestrator` assembles `OrchestratorDeps` from the
`AppState` built at startup: A's core (procedure, scorer, classifier, T8 decision, banks, copy),
the prompt sets, the Prisma stream client (fake server in demo mode), the Evon explainer (fake
Evon in demo mode, template fallback always), the capture tracker and the uplink.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sevi.app.orchestrator import (
    BrowserSession,
    CaptureLike,
    Orchestrator,
    OrchestratorDeps,
    UplinkLike,
)
from sevi.app.prompts import PromptSet, load_prompts, placeholder_prompts
from sevi.app.session_port import PermitLike, token_matches
from sevi.contracts.adapters import CallTransport, FrameTag
from sevi.contracts.enums import EndReason, KeyId, Lang, Mode, SessionState, Surface
from sevi.contracts.errors import ErrorCode, SeviError
from sevi.contracts.result import ResultDecision
from sevi.contracts.telemetry import EventName
from sevi.contracts.wire import CreateSessionRequest

if TYPE_CHECKING:
    from collections.abc import Callable

    from sevi.app.state import AppState
    from sevi.core.wording import WordingTable

PROMPTS_DIR: Final = Path("stimuli/prompts")


class OrchestratorHandle:
    """`SessionHandle` view of one `BrowserSession`."""

    def __init__(self, sess: BrowserSession) -> None:
        self.sess = sess
        self._transport: CallTransport | None = None

    @property
    def session_id(self) -> str:
        return self.sess.session_id

    @property
    def state(self) -> SessionState:
        return self.sess.state

    @property
    def mode_requested(self) -> Mode:
        return self.sess.rec.mode_requested

    @property
    def lang(self) -> Lang:
        return self.sess.rec.lang

    @property
    def connected(self) -> bool:
        return self.sess.connected and self.sess.state is not SessionState.ENDED

    @property
    def created_at(self) -> datetime:
        return self.sess.rec.created_at

    @property
    def ended_at(self) -> datetime | None:
        return self.sess.rec.ended_at

    @property
    def end_reason(self) -> EndReason | None:
        return self.sess.rec.end_reason

    def token_ok(self, token: str) -> bool:
        return token_matches(token, self.sess.ws_token_sha256)

    async def accept(self, transport: CallTransport) -> None:
        self._transport = transport
        await self.sess.accept(transport)  # marks the socket taken before any await (4409)
        await self.sess.deps.store.put_session(self.sess.rec)

    async def run(self) -> None:
        """session_main plus the transport reader that feeds its queue (B spec §7.2)."""
        t = self._transport
        reader = None if t is None else asyncio.create_task(self._pump(t), name="ws_events")
        try:
            await self.sess.run()
        finally:
            if reader is not None:
                reader.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reader

    async def _pump(self, t: CallTransport) -> None:
        async for ev in t.events():
            await self.sess.put(ev)
            if self.sess.state is SessionState.ENDED:
                return


class OrchestratorPort:
    """`SessionPort` over the real `Orchestrator`."""

    def __init__(self, orch: Orchestrator) -> None:
        self.orch = orch
        self._handles: dict[str, OrchestratorHandle] = {}

    def create(
        self,
        req: CreateSessionRequest,
        *,
        session_id: str,
        ws_token: str,
        permit: PermitLike | None,
        key_id: KeyId,
    ) -> OrchestratorHandle:
        sess, _ = self.orch.create_browser_session(
            req, permit=permit, key_id=key_id, session_id=session_id, ws_token=ws_token
        )
        h = OrchestratorHandle(sess)
        self._handles[session_id] = h
        return h

    def get(self, session_id: str) -> OrchestratorHandle | None:
        return self._handles.get(session_id)

    def all(self) -> Iterable[OrchestratorHandle]:
        return list(self._handles.values())

    def remove(self, session_id: str) -> None:
        self._handles.pop(session_id, None)
        self.orch.sessions.pop(session_id, None)

    async def end(self, session_id: str, reason: EndReason) -> bool:
        """Admission's S14 (never connected) and drain. A running session gets an abort event;
        one whose loop never started is ended here."""
        h = self._handles.get(session_id)
        if h is None or h.state is SessionState.ENDED:
            return False
        sess = h.sess
        if sess.main_task is not None and not sess.main_task.done():
            sess.end(reason)
            return await self.orch.abort(session_id, reason.value)
        sess.end(reason)
        sess.goto(SessionState.ENDED, "S14")
        await sess.ended()
        return True

    async def abort(self, session_id: str, reason: str) -> bool:
        h = self._handles.get(session_id)
        if h is None or h.state is SessionState.ENDED:
            return False
        if h.sess.main_task is None or h.sess.main_task.done():
            return await self.end(session_id, EndReason.OPERATOR_ABORT)
        return await self.orch.abort(session_id, reason)


# ---------- assembly ----------


class _WordingText:
    """`explain.WordingSource` over A's `WordingTable` (the explainer needs the text only)."""

    def __init__(self, table: WordingTable) -> None:
        self._t = table

    def wording(self, decision: ResultDecision, lang: Lang, surface: Surface) -> str:
        return self._t.wording(decision, lang, surface).text


def load_prompt_sets(state: AppState, base: Path = PROMPTS_DIR) -> dict[Lang, PromptSet]:
    """Committed prompt assets per loaded bank; the silent placeholder set outside prod.
    Prod refuses to start without real prompts (BANK_INTEGRITY / BANK_HASH_REJECTED)."""
    prod = state.settings.env == "prod"
    out: dict[Lang, PromptSet] = {}
    for lang, bank in state.banks.items():
        try:
            out[lang] = load_prompts(lang, base, bank.manifest, prod=prod)
        except SeviError:
            if prod:
                raise
            out[lang] = placeholder_prompts(lang, bank.manifest.digit_set)
    return out


def build_orchestrator(state: AppState) -> Orchestrator:
    from sevi.app.explain import CopyTable, ExplainConfig, Explainer
    from sevi.core.capture import CaptureParams, CaptureTracker
    from sevi.core.dsp import ulaw_decode
    from sevi.core.result import t8_decision
    from sevi.core.staircase import PROCEDURE
    from sevi.core.wording import DEFAULT_COPY_PATH
    from sevi.transport.uplink import Uplink

    s = state.settings
    prompts = load_prompt_sets(state)
    state.prompts_hashes = {lang.value: p.prompts_hash for lang, p in prompts.items()}
    for lang, p in prompts.items():
        if p.placeholder:
            state.telemetry.emit(
                EventName.STARTUP_STUB_FACTORY, "WARNING", factory=f"prompts.{lang.value}"
            )

    cap = s.capture
    params = CaptureParams(cap.speech_db_above_floor, cap.speech_min_ms, cap.echo_high_db)

    def capture_factory() -> CaptureLike:
        return CaptureTracker(params)

    def uplink_factory(
        gated_at: Callable[[float], bool], tag_at: Callable[[float], FrameTag | None]
    ) -> UplinkLike:
        return Uplink(ulaw_decode, state.telemetry, gated_at, tag_at)

    explainer = None
    llm = state.adapters.get("llm")
    if llm is not None:
        ev = s.evon
        explainer = Explainer(
            llm,
            CopyTable.load(Path(DEFAULT_COPY_PATH), state.wording.outcomes),
            _WordingText(state.wording),
            state.clock,
            ExplainConfig(
                enabled=ev.enabled,
                deadline_ms=ev.browser_deadline_s * 1000.0,
                timeout_s=ev.timeout_s,
                max_tokens=ev.max_tokens_explain,
                temperature=ev.temperature,
                max_concurrency=ev.max_concurrency,
                schema_pattern=ev.schema_pattern,
            ),
            breaker=state.breakers.get("evon", None),
            telemetry=state.telemetry,  # type: ignore[arg-type]
        )

    def breaker_open(k: KeyId) -> bool:
        return state.breakers.get("prisma", k).is_open()

    deps = OrchestratorDeps(
        settings=s,
        clock=state.clock,
        telemetry=state.telemetry,
        store=state.store,
        procedure=PROCEDURE,
        scorer=state.scorer,
        classifier=state.classifier,
        t8_decision=t8_decision,
        banks=state.banks,
        prompts=prompts,
        copy_text=state.wording.text,
        stt=state.adapters.get("stt"),
        explainer=explainer,
        capture_factory=capture_factory,
        uplink_factory=uplink_factory,
        switches=state.switches,
        git_sha=state.git_sha,
        is_breaker_open=breaker_open,
    )
    return Orchestrator(deps)


def demo_transcriber(orch: Orchestrator) -> Callable[[int], str]:
    """Demo mode only (`SEVI_DEMO_MODE`): the fake Prisma "hears" the digits of the triplet
    being presented in the newest voice session, so a local demo runs end to end with no keys.
    With several concurrent voice sessions the newest one wins (demo, not a measurement)."""

    def text(_k: int) -> str:
        for sess in reversed(list(orch.sessions.values())):
            pres = sess.pres
            if sess.mode is Mode.VOICE and pres is not None:
                return " ".join(str(d) for d in pres.spec.triplet)
        return ""

    return text


def orchestrator_port(state: AppState) -> OrchestratorPort:
    """The default `PortFactory` for `app.main.startup`."""
    try:
        orch = build_orchestrator(state)
        fake = state.adapters.get("fake_prisma")
        if state.settings.demo_mode and fake is not None and fake.text_source is None:
            fake.text_source = demo_transcriber(orch)
        return OrchestratorPort(orch)
    except SeviError as e:
        from sevi.app.main import StartupError

        raise StartupError(
            e.code if isinstance(e.code, ErrorCode) else ErrorCode.INTERNAL, str(e)
        ) from e
