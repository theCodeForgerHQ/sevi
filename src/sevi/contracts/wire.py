"""Our HTTP bodies and browser WebSocket UI messages (contracts §9).

Scope (BUILD-BRIEF reduced scope), omitted: `PanelNextResponse`, `PanelAnswerRequest` (human panel
cut), `StationSessionRequest`, `StationSessionResponse` (validation station cut), the
`reload_norms` switch and `VersionResponse.norms_sha256` (norms cut). The SSE timeline has no
model. Twilio IVR bodies (`DialRequest`, `DialResponse`) are kept (demo in scope).
`ParticipantView` and `SessionView` are not specified in §9.2 and are left to the operator API PR.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from sevi.contracts.enums import (
    AdmissionLevel,
    AgeBand,
    Band,
    Channel,
    ConsentScript,
    CutoffVersion,
    FriendsVariant,
    Lang,
    Mode,
    ParticipantType,
    PresentationKind,
    SessionState,
    Transducer,
    TruthRule,
    WordingClass,
)
from sevi.contracts.errors import ErrorCode
from sevi.contracts.types import (
    CallSid,
    GitSha,
    ParticipantId,
    Pseudonym,
    SemVer,
    SessionId,
    SeviModel,
    Sha256Hex,
    StudyFormId,
    UtcDatetime,
)

# ---------- HTTP (§9.2) ----------


class ClientHello(SeviModel):
    browser_family: str = Field(max_length=32)
    os_family: str = Field(max_length=32)
    ua: str = Field(max_length=300)  # hashed server-side, never stored raw
    audio_worklet: bool
    sample_rate_hz: int = Field(ge=8000, le=192000)
    page_version: str = Field(max_length=16)  # git sha 12


class CreateSessionRequest(SeviModel):
    lang: Lang
    mode: Literal[Mode.KEYPAD, Mode.VOICE]
    earphones: bool  # declared; False -> transducer SPEAKER -> NO_BAND (T11)
    client: ClientHello


class CreateSessionResponse(SeviModel):
    session_id: SessionId
    status: Literal["ADMITTED", "QUEUED"]
    ws_url: str  # f"wss://{host}/web/stream?sid={session_id}"
    ws_token: str = Field(pattern=r"^[A-Za-z0-9_-]{22}$")
    queue_position: int | None = None  # 1-based, QUEUED only
    eta_s: int | None = None  # queue_position * VOICE_ETA_S / slots
    expires_in_s: int  # ADMITTED_CONNECT_TIMEOUT_S or QUEUE_POLL_TIMEOUT_S


class SessionStatusResponse(SeviModel):
    session_id: SessionId
    status: Literal["QUEUED", "ADMITTED", "ACTIVE", "ENDED", "EXPIRED"]
    queue_position: int | None
    eta_s: int | None


class LangStatus(SeviModel):
    lang: Lang
    experimental: bool
    available: bool


class PublicStatusResponse(SeviModel):
    admission_level: AdmissionLevel
    voice_offered: bool
    voice_queue_len: int
    voice_eta_s: int | None
    languages: tuple[LangStatus, ...]
    phone_pilot: bool  # PHONE == OK -> landing shows the closed-pilot line
    version: str


class ReportTrial(SeviModel):
    scored_index: int
    snr_db: float
    correct: bool


class ReportResponse(SeviModel):
    lang: Lang
    channel: Channel
    mode: Mode
    transducer: Transducer
    created_at: UtcDatetime
    expires_at: UtcDatetime
    band: Band
    rule: TruthRule
    wording_class: WordingClass
    page_text: str  # ResultClassifier.wording(..., Surface.PAGE)
    srt_display: str | None
    n_norm: int | None
    ages: str | None
    cutoff_version: CutoffVersion | None
    bank_version: str
    evon_text: str | None  # guarded REPORT_PROSE, or None (template only)
    disclaimer: str  # P22 text in the session language
    trials: tuple[ReportTrial, ...]


class DialRequest(SeviModel):
    participant_id: ParticipantId
    participant_type: Literal[ParticipantType.PILOT, ParticipantType.STUDY]
    pseudonym: Pseudonym | None = None  # required iff STUDY
    age_band: AgeBand | None = None  # required iff STUDY
    study_form_id: StudyFormId | None = None  # required iff STUDY
    lang: Lang | None = None  # default: allowlist default_lang
    interview: bool = False  # may use the Key A reserve and the trial-minute reserve

    @model_validator(mode="after")
    def _study(self) -> DialRequest:  # R-12
        study = self.participant_type is ParticipantType.STUDY
        fields = (self.pseudonym, self.age_band, self.study_form_id)
        if study and any(f is None for f in fields):
            raise ValueError("STUDY requires pseudonym, age_band, study_form_id")
        if not study and any(f is not None for f in fields):
            raise ValueError("PILOT must not carry study fields")
        return self


class DialResponse(SeviModel):
    session_id: SessionId
    call_sid: CallSid


class SwitchRequest(SeviModel):
    name: Literal[
        "evon", "prisma", "timbre_runtime", "phone", "browser_voice", "demo_mode", "friends"
    ]
    value: bool | FriendsVariant


class PublicReadyz(SeviModel):  # public /readyz (R-40)
    status: Literal["ok", "degraded", "down"]


class ReadyzResponse(SeviModel):  # operator-only /op/api/readyz
    status: Literal["ok", "degraded", "down"]  # down iff no bank loaded or the DB is not writable
    admission_level: AdmissionLevel
    banks: dict[str, dict[str, str | bool]]  # lang -> {"bank_hash","frozen","ok"}
    prisma: dict[str, Literal["closed", "open", "disabled"]]  # "A","B"
    evon: Literal["up", "down", "disabled"]
    twilio: Literal["ok", "disabled", "trial_minutes_stop"]
    credits: dict[str, dict[str, float | bool]]  # "A"/"B" -> {"used_frac","reserve_ok"} + buckets
    active_sessions: dict[str, int]  # "keypad","voice","phone"
    voice_queue_len: int
    switches: dict[str, bool | str]


class VersionResponse(SeviModel):
    app_version: SemVer
    contracts_version: SemVer
    git_sha: GitSha
    banks: dict[str, str]  # lang -> bank_hash
    prompts: dict[str, str]  # lang -> prompts_hash
    copy_sha256: Sha256Hex


# ---------- browser WebSocket UI messages (§9.4) ----------


class BrowserUiEvent(SeviModel):
    type: Literal[
        "consent_given",
        "consent_declined",
        "cant_hear_confirmed",
        "ping",
        "mic_denied",
        "mic_lost",
        "mic_settings",
    ]
    ec: bool | None = None  # mic_settings only
    ns: bool | None = None
    agc: bool | None = None
    sample_rate_hz: int | None = Field(None, ge=8000, le=192000)  # mic_settings only


class UiConsent(SeviModel):
    type: Literal["consent"] = "consent"
    script: ConsentScript
    text: str


class UiPhase(SeviModel):
    type: Literal["phase"] = "phase"
    state: SessionState
    kind: PresentationKind | None
    scored_index: int | None = Field(None, ge=1, le=24)
    total: Literal[24] = 24


class UiKeypad(SeviModel):
    type: Literal["keypad"] = "keypad"
    enabled: bool


class UiMode(SeviModel):
    type: Literal["mode"] = "mode"
    mode: Literal[Mode.KEYPAD, Mode.VOICE]


class UiFeedback(SeviModel):
    type: Literal["feedback"] = "feedback"
    correct: bool


class UiResult(SeviModel):
    type: Literal["result"] = "result"
    band: Band
    wording_class: WordingClass
    page_text: str
    srt_display: str | None
    report_url: str


class UiExplanation(SeviModel):
    type: Literal["explanation"] = "explanation"
    text: str = Field(max_length=300)


class UiError(SeviModel):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: str = Field(max_length=300)


class UiClosed(SeviModel):
    type: Literal["closed"] = "closed"


class UiMic(SeviModel):
    type: Literal["mic"] = "mic"
    start: bool  # R-33: getUserMedia only after consent


class UiPong(SeviModel):
    type: Literal["pong"] = "pong"


class UiNotice(SeviModel):
    type: Literal["notice"] = "notice"
    key: Literal["speaker_echo"]


ServerUiMessage = Annotated[
    UiConsent
    | UiPhase
    | UiKeypad
    | UiMode
    | UiMic
    | UiFeedback
    | UiResult
    | UiExplanation
    | UiError
    | UiClosed
    | UiPong
    | UiNotice,
    Field(discriminator="type"),
]
