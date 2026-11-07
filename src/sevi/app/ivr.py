"""Twilio trial phone IVR (contracts §6.7, §8.6; B spec §8). Demo only.

Off by default (``SEVI_PHONE__ENABLED=false``): every ``/twilio/*`` and ``/audio/*`` request is
then 404 and the operator dial is 409 ``PHONE_DISABLED``.

The phone test is the *same* test as the browser test: one ``AdaptiveProcedure`` run, the same
bank renders (``/audio`` serves exactly the mu-law bytes the browser transport sends, IT-WEB-01),
the same ``AnswerScorer`` and the same ``ResultClassifier``. Only the transport differs: Twilio
drives a request/response step machine with webhooks (rows I01-I15):

* ``<Play>`` of pre-rendered prompt and stimulus audio served by us (``/audio/{t}/{name}.wav``),
* ``<Gather input="dtmf">`` for keypad answers (no gate, no replay, no ``*``/``0``),
* optional ``<Record>`` + Prisma REST only when ``SEVI_TWILIO__RECORD_ANSWERS=true``; the
  recording is deleted right after transcription, whatever happens.

Trial safety: outbound only, only to the verified caller IDs in
``SEVI_TWILIO__VERIFIED_TO_NUMBERS``, one call at a time, ``time_limit`` <= 600 s, and the
trial-minute ledger refuses a dial whose worst case would pass ``TRIAL_MINUTES_STOP``.
Phone results are NO_BAND (T11 ``NO_CELL``) unless a referral rule matches; the classifier owns
that rule. Evon is never called on the phone.

Every webhook: signature (rebuilt from the stored ``base_url``) -> call token (constant-time
sha256 compare) -> cached TwiML for ``(session, step)`` (a Twilio retry gets identical TwiML and
changes nothing, IT-IVR-01) -> compute under the call lock -> cache.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import io
import math
import re
import secrets
import struct
import wave
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Final, Literal, Protocol

from twilio.twiml.voice_response import VoiceResponse  # type: ignore[import-untyped]

from sevi import __version__
from sevi.app.orchestrator import new_session_id, sha256_hex
from sevi.app.prompts import PromptSet, digit_key, silence, wording_key
from sevi.app.settings import Settings, Switches
from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.adapters import Clock, Telemetry
from sevi.contracts.enums import (
    AnswerStatus,
    Channel,
    Condition,
    ConsentMethod,
    ConsentOutcome,
    ConsentScript,
    EndReason,
    FriendsVariant,
    KeyId,
    Lang,
    Mode,
    ParticipantType,
    PresentationKind,
    PromptKey,
    RetentionClass,
    ScoreAction,
    SessionState,
    Surface,
    SystemEvent,
    Transducer,
    TruthRule,
    WindowCloseReason,
)
from sevi.contracts.errors import ErrorCode, SeviError, TwilioError
from sevi.contracts.languages import EXPERIMENTAL_LANGS
from sevi.contracts.ports import AdaptiveProcedure, AnswerScorer, ResultClassifier, StimulusProvider
from sevi.contracts.procedure import ProcedureConfig, ProcedureState, SrtResult
from sevi.contracts.result import ResultDecision, ResultInput
from sevi.contracts.session import ConsentRecord, SessionRecord
from sevi.contracts.stimulus import RenderedTrial
from sevi.contracts.telemetry import EventName
from sevi.contracts.trial import (
    AnswerEvidence,
    DtmfPress,
    PresentationSpec,
    ScoringResult,
    SttSegment,
    TrialRecord,
)
from sevi.contracts.wire import DialRequest, DialResponse

# ---------- constants (contracts §6.7 / B spec §8) ----------
SIGNATURE_HEADER: Final = "X-Twilio-Signature"
T_DIAL_S: Final = 60.0
TIMEOUT_MARGIN_S: Final = 30  # I15: end at TIME_LIMIT_S - 30 s
STEP_DEADLINE_S: Final = 1.8
RECORD_STEP_DEADLINE_S: Final = 11.5
IDEMPOTENCY_TTL_S: Final = 86_400
GATHER_TIMEOUT_S: Final = 8
CONSENT_TIMEOUT_S: Final = 10
RECORD_MAX_S: Final = 8
SPEECH_PRESENT_S: Final = 0.3
FEEDBACK_GAP_MS: Final = 200.0
TERMINAL_CALL_STATUSES: Final = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})
STEP_RE: Final = re.compile(r"^(?:c[12]|p\d{1,3}|r\d{1,3}(?:w2|x)?)$")
AUDIO_NAME_RE: Final = re.compile(r"^(?:[A-Z0-9_]+|s\d{3})$")
TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
WAVE_FORMAT_MULAW: Final = 7


# ---------- ports ----------
class PhoneStore(Protocol):
    async def put_session(self, rec: SessionRecord) -> None: ...
    async def append_trial(self, trial: TrialRecord) -> None: ...
    async def idempotency_get(self, scope: str, key: str) -> str | None: ...
    async def idempotency_put(self, scope: str, key: str, value: str, ttl_s: int) -> None: ...
    async def twilio_status_upsert(self, call_sid: str, status: str, seq: int) -> bool: ...
    async def trial_minutes_add(self, call_sid: str, minutes: int) -> float: ...
    async def trial_minutes_total(self) -> float: ...


class TwilioPort(Protocol):
    """``adapters.twilio_rest.TwilioRest`` (or ``fakes.fake_twilio_rest.FakeTwilioRest``)."""

    async def create_call(self, *, to_e164: str, session_id: str, base: str, t: str) -> str: ...
    async def update_call(
        self, call_sid: str, status: Literal["canceled", "completed"]
    ) -> None: ...
    async def fetch_recording_wav(self, recording_url: str) -> bytes: ...
    async def delete_recording(self, recording_sid: str) -> None: ...
    def validate(self, url: str, form: Mapping[str, str], signature: str) -> bool: ...


# (wav_8k_pcm16, lang) -> transcript. Wiring builds it from ``PrismaRest.transcribe`` (Key A,
# through the ledger). Only used when ``SEVI_TWILIO__RECORD_ANSWERS=true``.
Transcriber = Callable[[bytes, Lang], Awaitable[str]]


@dataclass
class PhoneDeps:
    settings: Settings
    clock: Clock
    telemetry: Telemetry
    store: PhoneStore
    procedure: AdaptiveProcedure
    scorer: AnswerScorer
    classifier: ResultClassifier
    t8_decision: Callable[[SrtResult | None], ResultDecision]
    banks: Mapping[Lang, StimulusProvider]
    prompts: Mapping[Lang, PromptSet]
    twilio: TwilioPort
    copy_text: Callable[[str, Lang], str] | None = None
    transcribe: Transcriber | None = None
    switches: Switches | None = None
    git_sha: str = "0" * 40


# ---------- audio (contracts §8.6.3) ----------
def wav_ulaw(data: bytes) -> bytes:
    """44-byte WAVE_FORMAT_MULAW header (8 kHz, mono, 8 bit) + the mu-law bytes unchanged."""
    n = len(data)
    fmt = struct.pack("<HHIIHH", WAVE_FORMAT_MULAW, 1, 8000, 8000, 1, 8)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + n)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + (b"data" + struct.pack("<I", n) + data)
    )


def wav_data_chunk(wav: bytes) -> bytes:
    """The data chunk of a ``wav_ulaw`` body (tests and the simulator use it)."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE" or wav[36:40] != b"data":
        raise ValueError("not a 44-byte-header WAV")
    (n,) = struct.unpack("<I", wav[40:44])
    return wav[44 : 44 + n]


def pcm16_8k(wav: bytes) -> bytes:
    """Twilio recordings are 8 kHz PCM16 WAV; return the raw frames (mono)."""
    with wave.open(io.BytesIO(wav), "rb") as w:
        if w.getsampwidth() != 2 or w.getframerate() != 8000:
            raise ValueError("recording is not 8 kHz PCM16")
        frames = w.readframes(w.getnframes())
        if w.getnchannels() == 2:  # keep the left channel
            frames = b"".join(frames[i : i + 2] for i in range(0, len(frames), 4))
    return frames


def digits_of(form: Mapping[str, str]) -> str:
    """``Digits`` filtered to 1-9, first three kept (B spec §8.1)."""
    return "".join(c for c in form.get("Digits", "") if c in "123456789")[:3]


def build_playlist(d: ResultDecision, prompts: PromptSet) -> list[PromptKey]:
    """Contracts §6.5: the same playlist as the browser."""
    out: list[PromptKey] = []
    wk = wording_key(d.wording_class.value)
    if Surface.SPOKEN in d.surfaces and wk is not None and prompts.has(wk):
        out.append(wk)
    if d.rule is not TruthRule.T10:
        out.append(PromptKey.DISCLAIMER)
    out.append(PromptKey.GOODBYE_EARLY if d.rule is TruthRule.T7 else PromptKey.GOODBYE)
    return out


# ---------- per-call runtime (B spec §8.1) ----------
@dataclass
class PhoneCall:
    rec: SessionRecord
    token_sha256: str
    base: str
    mode: Mode
    proc: ProcedureState | None = None
    rendered: dict[int, RenderedTrial] = field(default_factory=dict)
    clips: dict[str, bytes] = field(default_factory=dict)  # composite clips (feedback)
    consent_repeats: int = 0
    answered_mono: float | None = None
    ended_mono: float | None = None
    last_twiml: str = ""
    reask_used: bool = False
    replay_used: bool = False
    voice_scored: int = 0
    keypad_scored: int = 0
    study_form_id: str | None = None
    rules: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dial_task: asyncio.Task[None] | None = None

    @property
    def sid(self) -> str:
        return self.rec.session_id

    @property
    def state(self) -> SessionState:
        return self.rec.state

    @property
    def pending(self) -> PresentationSpec | None:
        return None if self.proc is None else self.proc.pending

    def update(self, **kw: Any) -> None:
        self.rec = self.rec.model_copy(update=kw)


class PhoneService:
    """Dial, webhooks, audio and status for the trial IVR. One instance per process."""

    def __init__(
        self,
        deps: PhoneDeps,
        *,
        arm_dial_timer: bool = True,
        step_deadline_s: float = STEP_DEADLINE_S,
        record_deadline_s: float = RECORD_STEP_DEADLINE_S,
    ) -> None:
        self.deps = deps
        self.settings = deps.settings
        self.clock = deps.clock
        self.tm = deps.telemetry
        self.switches = deps.switches or Switches.from_settings(deps.settings)
        self.arm_dial_timer = arm_dial_timer
        self.step_deadline_s = step_deadline_s
        self.record_deadline_s = record_deadline_s
        self.calls: dict[str, PhoneCall] = {}
        self._by_token: dict[str, str] = {}
        self._dial_lock = asyncio.Lock()

    # ----- switches -----
    @property
    def enabled(self) -> bool:
        return bool(self.settings.phone.enabled and self.switches.phone)

    @property
    def record_mode(self) -> bool:
        return bool(self.settings.twilio.record_answers and self.deps.transcribe is not None)

    def _emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        self.tm.emit(event, level, **fields)

    def _rule(self, rt: PhoneCall, rule_id: str) -> None:
        rt.rules.append(rule_id)
        self._emit(
            EventName.FSM_TRANSITION,
            "DEBUG",
            session_id=rt.sid,
            rule_id=rule_id,
            state=rt.state.value,
        )

    def active_call(self) -> PhoneCall | None:
        for rt in self.calls.values():
            if rt.state is not SessionState.ENDED:
                return rt
        return None

    # ----- I01/I02: operator dial -----
    async def dial(self, req: DialRequest, *, idempotency_key: str | None) -> DialResponse:
        if not self.enabled:
            raise SeviError(ErrorCode.PHONE_DISABLED, "phone IVR is disabled")
        if not idempotency_key:
            raise SeviError(ErrorCode.BAD_REQUEST, "Idempotency-Key header required")
        cached = await self.deps.store.idempotency_get("dial", idempotency_key)
        if cached is not None:
            return DialResponse.model_validate_json(cached)
        to = self.settings.twilio.verified_numbers().get(req.participant_id)
        if to is None:
            self._emit(EventName.ERROR, "WARNING", code=ErrorCode.ALLOWLIST_DENIED.value)
            raise SeviError(
                ErrorCode.ALLOWLIST_DENIED,
                "participant is not a verified trial caller ID",
                details={"participant_id": req.participant_id},
            )
        lang = req.lang or Lang.TA
        if lang not in self.deps.banks or lang not in self.deps.prompts:
            raise SeviError(ErrorCode.LANG_UNAVAILABLE, f"language {lang.value} not loaded")
        async with self._dial_lock:
            if self.active_call() is not None:
                raise SeviError(ErrorCode.SESSION_STATE_CONFLICT, "a phone call is in progress")
            await self._check_minutes(req.interview)
            rt, t = self._new_call(req, lang)
            self.calls[rt.sid] = rt
            self._by_token[rt.token_sha256] = rt.sid
            await self.deps.store.put_session(rt.rec)
            self._rule(rt, "I01")
            try:
                call_sid = await self.deps.twilio.create_call(
                    to_e164=to, session_id=rt.sid, base=rt.base, t=t
                )
            except TwilioError as e:
                if e.details.get("maybe_created"):
                    self._rule(rt, "I02")  # stay DIALING; T_DIAL or the status callback ends it
                    self._arm_dial_timer(rt)
                else:
                    await self._end(rt, EndReason.DIAL_FAILED, classify=False)
                raise
            rt.update(twilio_call_sid=call_sid)
            await self.deps.store.put_session(rt.rec)
            self._arm_dial_timer(rt)
        resp = DialResponse(session_id=rt.sid, call_sid=call_sid)
        await self.deps.store.idempotency_put(
            "dial", idempotency_key, resp.model_dump_json(), IDEMPOTENCY_TTL_S
        )
        return resp

    async def _check_minutes(self, interview: bool) -> None:
        tw = self.settings.twilio
        used = await self.deps.store.trial_minutes_total()
        worst = math.ceil(tw.time_limit_s / 60.0)
        limit = tw.trial_minutes_total if interview else tw.trial_minutes_stop
        if used + worst > limit:
            raise SeviError(
                ErrorCode.TWILIO_TRIAL_MINUTES_STOP,
                "Twilio trial minutes stop reached",
                details={"used": used, "stop": limit},
            )

    def _new_call(self, req: DialRequest, lang: Lang) -> tuple[PhoneCall, str]:
        s = self.settings
        now = self.clock.now_utc()
        bank = self.deps.banks[lang]
        prompts = self.deps.prompts[lang]
        study = req.participant_type is ParticipantType.STUDY
        mode = Mode.VOICE if self.record_mode else Mode.KEYPAD
        t = secrets.token_urlsafe(16)
        rec = SessionRecord(
            session_id=new_session_id(),
            contracts_version=CONTRACTS_VERSION,
            app_version=__version__,
            git_sha=self.deps.git_sha,
            channel=Channel.PHONE,
            mode_requested=mode,
            mode_effective=mode,
            transducer=Transducer.HANDSET,
            lang=lang,
            participant_type=req.participant_type,
            participant_id=req.participant_id,
            pseudonym=req.pseudonym,
            age_band=req.age_band,
            key_id=KeyId.A if mode is Mode.VOICE else KeyId.NONE,
            state=SessionState.DIALING,
            created_at=now,
            started_at=None,
            ended_at=None,
            end_reason=None,
            consent=None,
            d_est_ms=None,
            t_guard_ms=0,  # the phone IVR has no uplink gate (contracts §6.7)
            echo_coupling_db=None,
            bank_hash=bank.manifest.bank_hash,
            bank_version=bank.manifest.bank_version,
            prompts_hash=prompts.prompts_hash,
            retention=RetentionClass.STUDY_KEEP if study else RetentionClass.DELETE_72H,
            delete_after=None if study else now + timedelta(hours=s.retention.hours),
            media_token_sha256=sha256_hex(t),
        )
        rt = PhoneCall(
            rec=rec,
            token_sha256=sha256_hex(t),
            base=str(s.public_base_url).rstrip("/"),
            mode=mode,
            study_form_id=req.study_form_id,
        )
        return rt, t

    def _arm_dial_timer(self, rt: PhoneCall) -> None:
        if not self.arm_dial_timer:
            return

        async def timer() -> None:
            await asyncio.sleep(T_DIAL_S)
            await self.on_dial_timeout(rt.sid)

        rt.dial_task = asyncio.create_task(timer())

    # ----- I11/I12/I13 -----
    async def on_dial_timeout(self, sid: str) -> None:
        rt = self.calls.get(sid)
        if rt is None:
            return
        async with rt.lock:
            if rt.state is SessionState.DIALING:
                self._rule(rt, "I11")
                await self._cancel(rt, "canceled")
                await self._end(rt, EndReason.DIAL_FAILED, classify=False)

    async def abort(self, sid: str) -> bool:
        rt = self.calls.get(sid)
        if rt is None:
            return False
        async with rt.lock:
            if rt.state is SessionState.ENDED:
                return True
            if rt.state is SessionState.DIALING:
                self._rule(rt, "I12")
                await self._cancel(rt, "canceled")
                await self._end(rt, EndReason.OPERATOR_ABORT, classify=False)
            else:
                self._rule(rt, "I13")
                await self._cancel(rt, "completed")
                await self._end(rt, EndReason.OPERATOR_ABORT)
        return True

    async def _cancel(self, rt: PhoneCall, status: Literal["canceled", "completed"]) -> None:
        call_sid = rt.rec.twilio_call_sid
        if call_sid is None:
            return
        try:
            await self.deps.twilio.update_call(call_sid, status)
        except TwilioError as e:
            st = int(e.details.get("status") or 0)
            if status == "canceled" and 400 <= st < 500:  # R-24: already answered
                with contextlib.suppress(TwilioError):
                    await self.deps.twilio.update_call(call_sid, "completed")
            else:
                self._emit(EventName.ERROR, "WARNING", code=ErrorCode.TWILIO_REST_FAILED.value)

    # ----- webhook authentication (contracts §6.7) -----
    def authenticate(
        self, sid: str, t: str | None, path_qs: str, form: Mapping[str, str], signature: str | None
    ) -> PhoneCall:
        """Signature first (403 TWILIO_SIGNATURE_INVALID), then the call token (403
        TWILIO_TOKEN_INVALID). An unknown session is 404 (a restart ends live calls)."""
        rt = self.calls.get(sid)
        if rt is None:
            raise SeviError(ErrorCode.NOT_FOUND, "unknown call")
        if self.settings.twilio.validate_signature and not (
            signature and self.deps.twilio.validate(rt.base + path_qs, form, signature)
        ):
            self._emit(EventName.ERROR, "WARNING", code=ErrorCode.TWILIO_SIGNATURE_INVALID.value)
            raise SeviError(ErrorCode.TWILIO_SIGNATURE_INVALID, "bad X-Twilio-Signature")
        if not t or not hmac.compare_digest(sha256_hex(t), rt.token_sha256):
            self._emit(EventName.ERROR, "WARNING", code=ErrorCode.TWILIO_TOKEN_INVALID.value)
            raise SeviError(ErrorCode.TWILIO_TOKEN_INVALID, "bad call token")
        return rt

    # ----- I03-I09, I14, I15 -----
    async def webhook(self, rt: PhoneCall, step: str, form: Mapping[str, str], t: str) -> str:
        """``step`` = ``voice`` (I03) or a ``/twilio/step`` name. Returns TwiML."""
        if step != "voice" and not STEP_RE.match(step):
            raise SeviError(ErrorCode.NOT_FOUND, "unknown step")
        scope = f"twiml:{rt.sid}"
        hit = await self.deps.store.idempotency_get(scope, step)
        if hit is not None:
            return hit  # IT-IVR-01: a Twilio retry changes nothing
        record = step.startswith("r")
        deadline = self.record_deadline_s if record else self.step_deadline_s
        async with rt.lock:
            hit = await self.deps.store.idempotency_get(scope, step)
            if hit is not None:
                return hit
            try:
                async with asyncio.timeout(deadline):
                    twiml = await self._step(rt, step, form, t)
            except Exception as e:  # I14
                twiml = await self._crash(rt, t, e)
            rt.last_twiml = twiml
            await self.deps.store.idempotency_put(scope, step, twiml, IDEMPOTENCY_TTL_S)
        self._emit(EventName.TWILIO_STEP, session_id=rt.sid, step=step, state=rt.state.value)
        return twiml

    async def _step(self, rt: PhoneCall, step: str, form: Mapping[str, str], t: str) -> str:
        if rt.state is SessionState.ENDED:
            return _hangup()
        if step == "voice":
            if rt.state is not SessionState.DIALING:
                return rt.last_twiml or _hangup()
            self._rule(rt, "I03")
            if rt.dial_task is not None:
                rt.dial_task.cancel()
            rt.answered_mono = self.clock.mono_ms()
            rt.update(state=SessionState.CONSENT)
            await self.deps.store.put_session(rt.rec)
            return self._consent_twiml(rt, t, "c1")
        if step in ("c1", "c2"):
            if rt.state is not SessionState.CONSENT:
                return rt.last_twiml or _hangup()
            return await self._consent_step(rt, step, digits_of(form), t)
        m = re.match(r"^([pr])(\d+)(w2|x)?$", step)
        assert m is not None
        pid = int(m.group(2))
        pending = rt.pending
        if rt.state is not SessionState.TEST or pending is None or pending.presentation_id != pid:
            self._rule(rt, "I09")
            return rt.last_twiml or _hangup()
        if self._elapsed_s(rt) > self.settings.twilio.time_limit_s - TIMEOUT_MARGIN_S:
            self._rule(rt, "I15")
            return await self._terminal(rt, t, EndReason.SESSION_TIMEOUT)
        self._rule(rt, "I08")
        if m.group(1) == "p":
            ev = self._keypad_evidence(rt, pending, digits_of(form))
            return await self._answer(rt, pending, ev, t)
        ev = await self._record_evidence(rt, pending, form, window2=m.group(3) == "w2")
        return await self._answer(rt, pending, ev, t)

    def _elapsed_s(self, rt: PhoneCall) -> float:
        if rt.answered_mono is None:
            return 0.0
        return (self.clock.mono_ms() - rt.answered_mono) / 1000.0

    # ----- consent -----
    def _consent_key(self, rt: PhoneCall) -> PromptKey:
        if rt.rec.participant_type is ParticipantType.STUDY:
            return PromptKey.CONSENT_STUDY
        return PromptKey.CONSENT_PUBLIC

    def _consent_twiml(self, rt: PhoneCall, t: str, action: str) -> str:
        r = VoiceResponse()
        g = r.gather(
            input="dtmf",
            num_digits=1,
            timeout=CONSENT_TIMEOUT_S,
            action_on_empty_result=True,
            action=self._step_url(rt, action, t),
            method="POST",
        )
        g.play(self._audio_url(rt, t, self._consent_key(rt).value))
        return str(r)

    def _consent_record(
        self, rt: PhoneCall, outcome: ConsentOutcome, method: ConsentMethod | None
    ) -> None:
        study = rt.rec.participant_type is ParticipantType.STUDY
        key = self._consent_key(rt)
        copy = self.deps.copy_text
        text = ""
        if copy is not None:
            with contextlib.suppress(Exception):
                text = copy(f"prompt.{key.value.lower()}", rt.rec.lang)
        rt.update(
            consent=ConsentRecord(
                session_id=rt.sid,
                script=ConsentScript.STUDY_V1 if study else ConsentScript.PUBLIC_V1,
                script_sha256=sha256_hex(text),
                lang=rt.rec.lang,
                outcome=outcome,
                method=method,
                repeats=min(rt.consent_repeats, 1),
                at=self.clock.now_utc(),
                study_form_id=rt.study_form_id if study else None,
            )
        )

    async def _consent_step(self, rt: PhoneCall, step: str, digits: str, t: str) -> str:
        first = digits[:1]
        if first == "1":
            self._rule(rt, "I04")
            self._consent_record(rt, ConsentOutcome.GIVEN, ConsentMethod.DTMF_1)
            rt.update(started_at=self.clock.now_utc(), state=SessionState.INSTRUCTIONS)
            self._emit(EventName.CONSENT_GIVEN, session_id=rt.sid)
            bank = self.deps.banks[rt.rec.lang]
            m = bank.manifest
            rt.proc = self.deps.procedure.start(
                ProcedureConfig(
                    seed_scope=rt.sid,
                    design=m.design,
                    digit_set=m.digit_set,
                    condition=Condition.NORMAL,
                    skip_volume_and_practice=False,
                )
            )
            rt.update(state=SessionState.TEST)
            await self.deps.store.put_session(rt.rec)
            pending = rt.pending
            if pending is None:  # pragma: no cover - start() always sets pending
                return await self._terminal(rt, t, EndReason.COMPLETED)
            r = VoiceResponse()
            r.play(self._audio_url(rt, t, PromptKey.INSTRUCTIONS.value))
            await self._presentation(rt, r, pending, t)
            return str(r)
        if first == "2":
            self._rule(rt, "I05")
            self._consent_record(rt, ConsentOutcome.DECLINED, ConsentMethod.DTMF_2)
            return await self._terminal(rt, t, EndReason.CONSENT_DECLINED)
        if step == "c1":
            self._rule(rt, "I06")
            rt.consent_repeats = 1
            return self._consent_twiml(rt, t, "c2")
        self._rule(rt, "I07")
        rt.consent_repeats = 1
        self._consent_record(rt, ConsentOutcome.NO_REPLY, None)
        await self._end(rt, EndReason.CONSENT_NO_REPLY)
        return _hangup()

    # ----- presentations -----
    async def _render(self, rt: PhoneCall, spec: PresentationSpec) -> RenderedTrial:
        pid = spec.presentation_id
        r = rt.rendered.get(pid)
        if r is None or r.triplet != spec.triplet:
            bank = self.deps.banks[rt.rec.lang]
            r = await asyncio.to_thread(bank.render_trial, spec)
            rt.rendered[pid] = r
        return r

    async def _presentation(
        self, rt: PhoneCall, resp: Any, spec: PresentationSpec, t: str, *, suffix: str = ""
    ) -> None:
        """Append the TwiML for ``spec`` (contracts §6.7) to ``resp``."""
        await self._render(rt, spec)
        pid = spec.presentation_id
        stim = self._audio_url(rt, t, f"s{pid:03d}")
        if spec.pre_prompt is not None and not suffix:
            resp.play(self._audio_url(rt, t, spec.pre_prompt.value))
        if rt.mode is Mode.VOICE:
            if suffix != "w2":
                resp.play(stim)
            resp.record(
                max_length=RECORD_MAX_S,
                timeout=2,
                play_beep=False,
                trim="trim-silence",
                action=self._step_url(rt, f"r{pid}{suffix}", t),
                method="POST",
            )
            return
        g = resp.gather(
            input="dtmf",
            num_digits=3,
            timeout=GATHER_TIMEOUT_S,
            finish_on_key="#",
            action_on_empty_result=True,
            action=self._step_url(rt, f"p{pid}", t),
            method="POST",
        )
        g.play(stim)

    def _keypad_evidence(
        self, rt: PhoneCall, spec: PresentationSpec, digits: str
    ) -> AnswerEvidence:
        presses = tuple(DtmfPress(key=d, at_ms=0.0) for d in digits)
        return AnswerEvidence(
            presentation_id=spec.presentation_id,
            lang=rt.rec.lang,
            mode=Mode.KEYPAD,
            window_index=1,
            window_open_mono_ms=0.0,
            window_close_mono_ms=0.0,
            close_reason=(
                WindowCloseReason.DTMF_COMPLETE if len(presses) == 3 else WindowCloseReason.TIMEOUT
            ),
            dtmf=presses,
            speech_present=False,
            speech_ms=0,
            early_speech=False,
            echo_mode=False,
            reask_used=False,
            replay_used=False,
        )

    async def _record_evidence(
        self, rt: PhoneCall, spec: PresentationSpec, form: Mapping[str, str], *, window2: bool
    ) -> AnswerEvidence:
        """B spec §8.4: fetch -> Prisma REST -> delete the recording (always) -> evidence."""
        url = form.get("RecordingUrl", "")
        rsid = form.get("RecordingSid", "")
        try:
            dur_s = float(form.get("RecordingDuration", "0") or 0)
        except ValueError:
            dur_s = 0.0
        text = ""
        events: tuple[SystemEvent, ...] = ()
        try:
            if not url or self.deps.transcribe is None:
                raise ValueError("no recording")
            wav = await self.deps.twilio.fetch_recording_wav(url)
            text = (await self.deps.transcribe(pcm16_8k(wav), rt.rec.lang))[:500]
        except Exception:
            events = (SystemEvent.STT_ERROR,)
            self._emit(EventName.ERROR, "WARNING", session_id=rt.sid, code="STT_REST_FAILED")
        finally:
            if rsid:
                try:
                    await self.deps.twilio.delete_recording(rsid)
                except TwilioError:  # retention retries the backstop delete
                    self._emit(EventName.ERROR, "WARNING", code=ErrorCode.TWILIO_REST_FAILED.value)
        segs = (SttSegment(text=text, received_ms=0.0, after_commit=False),) if text else ()
        return AnswerEvidence(
            presentation_id=spec.presentation_id,
            lang=rt.rec.lang,
            mode=Mode.VOICE,
            window_index=2 if window2 else 1,
            window_open_mono_ms=0.0,
            window_close_mono_ms=dur_s * 1000.0,
            close_reason=WindowCloseReason.SETTLE,
            segments=segs,
            speech_present=dur_s >= SPEECH_PRESENT_S,
            speech_ms=int(dur_s * 1000),
            early_speech=False,
            echo_mode=False,
            system_events=events,
            reask_used=rt.reask_used,
            replay_used=rt.replay_used,
        )

    async def _answer(
        self, rt: PhoneCall, spec: PresentationSpec, ev: AnswerEvidence, t: str
    ) -> str:
        result = self.deps.scorer.score(spec, ev)
        self._emit(
            EventName.SCORE_RESULT,
            session_id=rt.sid,
            status=result.status.value,
            action=result.action.value,
            source=result.source.value,
        )
        r = VoiceResponse()
        if result.action is ScoreAction.REASK and result.reask_prompt is not None:
            rt.reask_used = True
            r.play(self._audio_url(rt, t, result.reask_prompt.value))
            await self._presentation(rt, r, spec, t, suffix="w2")
            return str(r)
        if result.action is ScoreAction.REPLAY:
            rt.replay_used = True
            await self._presentation(rt, r, spec, t, suffix="x")
            return str(r)
        await self._trial(rt, spec, result, ev)
        if spec.feedback:
            r.play(self._audio_url(rt, t, self._feedback_clip(rt, spec, result)))
        assert rt.proc is not None
        rt.proc = self.deps.procedure.record(rt.proc, result)
        self._emit(
            EventName.PROCEDURE_RECORD, session_id=rt.sid, presentation_id=spec.presentation_id
        )
        nxt = rt.proc.pending
        if nxt is None:
            self._emit(
                EventName.PROCEDURE_TERMINAL, session_id=rt.sid, terminal=str(rt.proc.terminal)
            )
            return await self._terminal(rt, t, EndReason.COMPLETED, resp=r)
        await self._presentation(rt, r, nxt, t)
        return str(r)

    def _feedback_clip(self, rt: PhoneCall, spec: PresentationSpec, res: ScoringResult) -> str:
        """Practice feedback as one clip (one fetch): CORRECT, or WRONG + the three digits."""
        prompts = self.deps.prompts[rt.rec.lang]
        if res.triplet_correct:
            return PromptKey.PRACTICE_CORRECT.value
        name = f"F{spec.presentation_id:03d}"
        audio = prompts.get(PromptKey.PRACTICE_WRONG).ulaw
        for d in spec.triplet:
            audio += silence(FEEDBACK_GAP_MS) + prompts.get(digit_key(d)).ulaw
        rt.clips[name] = audio
        return name

    async def _trial(
        self, rt: PhoneCall, spec: PresentationSpec, result: ScoringResult, ev: AnswerEvidence
    ) -> None:
        r = rt.rendered.get(spec.presentation_id)
        rec = TrialRecord(
            session_id=rt.sid,
            spec=spec,
            result=result,
            system_flag=spec.attempt == 2 and result.status is AnswerStatus.CAPTURE_MISS,
            reasks=int(ev.reask_used),
            replays=int(ev.replay_used),
            early_answer=False,
            echo_mode=False,
            rx_floor_dbfs=None,
            speech_ms=ev.speech_ms,
            t_send_mono_ms=0.0,
            gate_end_mono_ms=0.0,
            d_est_ms=0.0,
            t_guard_ms=0,
            last_digit_offset_ms=r.timing.last_digit_offset_ms if r is not None else 0.0,
            transcript=(" ".join(s.text for s in ev.segments)[:1000] or None),
            stale_segments=0,
            turn_ms=None,
            level_limited=bool(r is not None and r.level_limited),
        )
        await self.deps.store.append_trial(rec)
        if spec.kind is PresentationKind.SCORED and result.status is not AnswerStatus.CAPTURE_MISS:
            if result.mode is Mode.VOICE:
                rt.voice_scored += 1
            else:
                rt.keypad_scored += 1
        rt.reask_used = rt.replay_used = False

    # ----- result / end -----
    def _mode_effective(self, rt: PhoneCall) -> Mode:
        v, k = rt.voice_scored, rt.keypad_scored
        if v and k:
            return Mode.MIXED
        if v:
            return Mode.VOICE
        if k:
            return Mode.KEYPAD
        return rt.mode

    def _classify(self, rt: PhoneCall) -> ResultDecision:
        """Exactly once per call (R-15); later steps reuse the stored decision."""
        if rt.rec.decision is not None:
            return rt.rec.decision
        srt = self.deps.procedure.summarize(rt.proc) if rt.proc is not None else None
        reason = rt.rec.end_reason or EndReason.COMPLETED
        bank = self.deps.banks[rt.rec.lang]
        decision = self.deps.classifier.classify(
            ResultInput(
                session_id=rt.sid,
                lang=rt.rec.lang,
                channel=Channel.PHONE,
                mode=self._mode_effective(rt),
                mode_requested=Mode.VOICE if rt.mode is Mode.VOICE else Mode.KEYPAD,
                transducer=Transducer.HANDSET,
                participant_type=rt.rec.participant_type,
                end_reason=reason,
                srt=srt,
                friends=FriendsVariant.ALLOWED
                if self.switches.friends
                else FriendsVariant.DISALLOWED,
                experimental_language=rt.rec.lang in EXPERIMENTAL_LANGS,
                bank_hash=bank.manifest.bank_hash,
                norms=None,
                hhie_total=None,
            )
        )
        rt.update(srt=srt, decision=decision)
        self._emit(
            EventName.RESULT_DECIDED,
            session_id=rt.sid,
            rule=decision.rule.value,
            band=decision.band.value,
        )
        return decision

    async def _end(self, rt: PhoneCall, reason: EndReason, *, classify: bool = True) -> None:
        if rt.state is SessionState.ENDED:
            return
        if rt.rec.end_reason is None:
            rt.update(end_reason=reason)
        if classify:
            self._classify(rt)
        if rt.dial_task is not None:
            rt.dial_task.cancel()
        rt.ended_mono = self.clock.mono_ms()
        rt.update(
            state=SessionState.ENDED,
            ended_at=self.clock.now_utc(),
            mode_effective=self._mode_effective(rt),
        )
        await self.deps.store.put_session(rt.rec)
        self._emit(EventName.SESSION_ENDED, session_id=rt.sid, end_reason=reason.value)

    async def _terminal(self, rt: PhoneCall, t: str, reason: EndReason, *, resp: Any = None) -> str:
        await self._end(rt, reason)
        decision = rt.rec.decision
        assert decision is not None
        r = resp if resp is not None else VoiceResponse()
        for key in build_playlist(decision, self.deps.prompts[rt.rec.lang]):
            r.play(self._audio_url(rt, t, key.value))
        r.hangup()
        return str(r)

    async def _crash(self, rt: PhoneCall, t: str, e: BaseException) -> str:
        """I14: SERVER_ERROR; T8 without classify(); SYSFAIL clip + hangup."""
        self._emit(
            EventName.ERROR, "ERROR", session_id=rt.sid, code="INTERNAL", exc=type(e).__name__
        )
        if rt.rec.decision is None:
            srt = None
            with contextlib.suppress(Exception):
                srt = self.deps.procedure.summarize(rt.proc) if rt.proc is not None else None
            rt.update(srt=srt, decision=self.deps.t8_decision(srt))
        with contextlib.suppress(Exception):
            await self._end(rt, EndReason.SERVER_ERROR, classify=False)
        r = VoiceResponse()
        r.play(self._audio_url(rt, t, PromptKey.WORDING_W_SYSFAIL.value))
        r.hangup()
        return str(r)

    # ----- I10: status callback -----
    async def status(self, rt: PhoneCall, form: Mapping[str, str]) -> None:
        status = form.get("CallStatus", "")
        call_sid = form.get("CallSid", "") or (rt.rec.twilio_call_sid or "")
        try:
            seq = int(form.get("SequenceNumber", "0") or 0)
        except ValueError:
            seq = 0
        if not call_sid or not await self.deps.store.twilio_status_upsert(call_sid, status, seq):
            return  # duplicate: no effect
        self._emit(EventName.TWILIO_STATUS, session_id=rt.sid, call_status=status)
        if status not in TERMINAL_CALL_STATUSES:
            return
        if status == "completed":
            try:
                dur = float(form.get("CallDuration", "0") or 0)
            except ValueError:
                dur = 0.0
            if dur > 0:
                await self.deps.store.trial_minutes_add(call_sid, math.ceil(dur / 60.0))
        async with rt.lock:
            if rt.state is SessionState.ENDED:
                return
            self._rule(rt, "I10")
            if rt.state is SessionState.DIALING:
                await self._end(rt, EndReason.DIAL_FAILED, classify=False)
            else:
                await self._end(rt, EndReason.CALLER_HANGUP)

    # ----- audio (contracts §8.6.3) -----
    def audio(self, t: str, name: str) -> bytes | None:
        """The WAV body for ``/audio/{t}/{name}.wav``, or None (404)."""
        if not TOKEN_RE.match(t) or not AUDIO_NAME_RE.match(name):
            return None
        sid = self._by_token.get(sha256_hex(t))
        rt = self.calls.get(sid) if sid is not None else None
        if rt is None or not hmac.compare_digest(sha256_hex(t), rt.token_sha256):
            return None
        if rt.ended_mono is not None and (
            self.clock.mono_ms() - rt.ended_mono > self.settings.twilio.audio_ttl_s * 1000.0
        ):
            return None
        data: bytes | None
        if name.startswith("s") and name[1:].isdigit():
            r = rt.rendered.get(int(name[1:]))
            data = None if r is None else r.ulaw
        elif name in rt.clips:
            data = rt.clips[name]
        else:
            try:
                data = self.deps.prompts[rt.rec.lang].get(PromptKey(name)).ulaw
            except (ValueError, KeyError):
                data = None
        if data is None:
            return None
        self._emit(EventName.AUDIO_SERVED, session_id=rt.sid, name=name)
        return wav_ulaw(data)

    def sweep(self) -> int:
        """Forget calls whose audio TTL has passed. Returns how many were dropped."""
        ttl = self.settings.twilio.audio_ttl_s * 1000.0
        now = self.clock.mono_ms()
        gone = [
            sid
            for sid, rt in self.calls.items()
            if rt.ended_mono is not None and now - rt.ended_mono > ttl
        ]
        for sid in gone:
            rt = self.calls.pop(sid)
            self._by_token.pop(rt.token_sha256, None)
        return len(gone)

    async def close(self) -> None:
        for rt in self.calls.values():
            if rt.dial_task is not None:
                rt.dial_task.cancel()

    # ----- URLs -----
    def _step_url(self, rt: PhoneCall, step: str, t: str) -> str:
        return f"{rt.base}/twilio/step/{rt.sid}/{step}?t={t}"

    def _audio_url(self, rt: PhoneCall, t: str, name: str) -> str:
        return f"{rt.base}/audio/{t}/{name}.wav"


def _hangup() -> str:
    r = VoiceResponse()
    r.hangup()
    return str(r)


__all__ = [
    "SIGNATURE_HEADER",
    "PhoneCall",
    "PhoneDeps",
    "PhoneService",
    "PhoneStore",
    "Transcriber",
    "TwilioPort",
    "build_playlist",
    "digits_of",
    "pcm16_8k",
    "wav_data_chunk",
    "wav_ulaw",
]
