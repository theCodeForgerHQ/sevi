"""Session/call, consent, allowlist and client records (contracts §4.8).

Scope (BUILD-BRIEF reduced scope): `PanelTrial` and `PanelResponse` are omitted (human panel is
cut). STUDY fields stay (dormant) so the retention rule and the phone IVR DialRequest keep the
spec shape.

`SessionRecord` references Person A's models (`TrialRecord`, `SrtResult`, `ResultDecision`) by
module name only. They are imported under TYPE_CHECKING; `sevi.contracts.__init__` rebuilds the
model once every contracts module is importable. Until then `SessionRecord` is incomplete and
is left out of `ALL_MODELS` (see `PENDING_MODELS`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from sevi.contracts.enums import (
    AgeBand,
    AllowlistKind,
    Channel,
    ConsentMethod,
    ConsentOutcome,
    ConsentScript,
    EndReason,
    EvonOutcome,
    KeyId,
    Lang,
    Mode,
    ParticipantType,
    RelationLabel,
    RetentionClass,
    SessionFlag,
    SessionState,
    Transducer,
)
from sevi.contracts.types import (
    CallSid,
    GitSha,
    NonNegMs,
    ParticipantId,
    Pseudonym,
    SemVer,
    SessionId,
    SeviModel,
    Sha256Hex,
    StudyFormId,
    UtcDatetime,
)

if TYPE_CHECKING:  # Person A's modules (contracts §2.1); resolved at runtime by model_rebuild
    from sevi.contracts.procedure import SrtResult
    from sevi.contracts.result import ResultDecision
    from sevi.contracts.trial import TrialRecord


class ConsentRecord(SeviModel):
    session_id: SessionId
    script: ConsentScript  # STUDY_V1 iff participant_type == STUDY
    script_sha256: Sha256Hex  # sha of the copy text actually rendered/shown
    lang: Lang
    outcome: ConsentOutcome
    method: ConsentMethod | None  # None iff NO_REPLY
    repeats: int = Field(ge=0, le=1)
    at: UtcDatetime
    study_form_id: StudyFormId | None = None  # required iff script == STUDY_V1

    @model_validator(mode="after")
    def _study(self) -> ConsentRecord:
        if (self.script is ConsentScript.STUDY_V1) != (self.study_form_id is not None):
            raise ValueError("study_form_id required iff STUDY_V1")
        return self


class AllowlistEntry(SeviModel):
    participant_id: ParticipantId
    kind: AllowlistKind
    number_ciphertext: str = Field(min_length=40)  # Fernet token of the E.164 number
    number_hmac: Sha256Hex  # HMAC-SHA256(SEVI_ALLOWLIST__HMAC_KEY, e164)
    relation_label: RelationLabel = RelationLabel.SEVI_TEAM
    default_lang: Lang = Lang.TA
    created_at: UtcDatetime
    expires_at: UtcDatetime  # teammates: 2026-12-31T18:29:59Z (end of interviews)
    active: bool = True


class ClientInfo(SeviModel):
    browser_family: str = Field(max_length=32)  # "Chrome", "Safari", "Firefox", "Other"
    os_family: str = Field(max_length=32)  # "Android", "iOS", "macOS", "Windows", "Linux", "Other"
    ua_sha256: Sha256Hex
    audio_worklet: bool
    sample_rate_hz: int = Field(ge=8000, le=192000)
    # what the browser actually applied (track.getSettings()), voice only
    mic_ec: bool | None = None
    mic_ns: bool | None = None
    mic_agc: bool | None = None


class SessionRecord(SeviModel):
    session_id: SessionId
    contracts_version: SemVer
    app_version: SemVer
    git_sha: GitSha
    channel: Channel
    mode_requested: Mode  # VOICE | KEYPAD
    mode_effective: Mode  # VOICE | KEYPAD | MIXED
    transducer: Transducer
    lang: Lang
    participant_type: ParticipantType  # browser: always PUBLIC
    participant_id: ParticipantId | None  # phone only
    pseudonym: Pseudonym | None  # STUDY only
    age_band: AgeBand | None  # STUDY only
    key_id: KeyId
    state: SessionState
    created_at: UtcDatetime
    started_at: UtcDatetime | None  # consent GIVEN
    ended_at: UtcDatetime | None
    end_reason: EndReason | None
    consent: ConsentRecord | None
    flags: tuple[SessionFlag, ...] = ()
    d_est_ms: NonNegMs | None
    t_guard_ms: int = Field(ge=0)
    echo_coupling_db: float | None
    bank_hash: Sha256Hex
    bank_version: str
    prompts_hash: Sha256Hex
    trials: tuple[TrialRecord, ...] = ()
    srt: SrtResult | None = None
    decision: ResultDecision | None = None
    evon: EvonOutcome = EvonOutcome.SKIPPED
    report_token_sha256: Sha256Hex | None = None  # sha256 of the 128-bit URL token; never the token
    retention: RetentionClass
    delete_after: UtcDatetime | None  # created_at + 72 h for DELETE_72H; None for STUDY_KEEP
    twilio_call_sid: CallSid | None = None
    twilio_deleted: bool = False  # call and any recordings deleted
    media_token_sha256: Sha256Hex | None = None  # phone IVR: token in /audio and /twilio URLs
    client: ClientInfo | None = None  # browser only
