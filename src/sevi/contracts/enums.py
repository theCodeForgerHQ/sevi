"""Every shared enum (contracts §4.2)."""

from enum import StrEnum


class Lang(StrEnum):
    TA = "ta"
    EN = "en"  # v4 (audit M2): ta banded, en experimental (Scope v3 (b): never norm-banded)


class Channel(StrEnum):
    PHONE = "phone"
    BROWSER = "browser"


class Mode(StrEnum):
    VOICE = "voice"
    KEYPAD = "keypad"
    MIXED = "mixed"  # MIXED only as a session/SRT summary


class Transducer(StrEnum):
    HANDSET = "handset"
    EARPHONES = "earphones"
    SPEAKER = "speaker"
    # phone: HANDSET unless ECHO_MODE was set (then SPEAKER);
    # browser: EARPHONES if declared yes, else SPEAKER


class ParticipantType(StrEnum):
    PUBLIC = "PUBLIC"
    PILOT = "PILOT"
    STUDY = "STUDY"


class ModelVersion(StrEnum):
    V3 = "v3"
    V2_5 = "v2.5"


class Condition(StrEnum):
    NORMAL = "NORMAL"
    N2 = "N2"
    N3 = "N3"
    LP1500 = "LP1500"


class BankVariant(StrEnum):
    SHIPPED = "SHIPPED"
    BALANCED = "BALANCED"
    RMS_ONLY = "RMS_ONLY"  # panel uses BALANCED + RMS_ONLY


class OffsetSource(StrEnum):
    PRISMA = "PRISMA"
    HUMAN = "HUMAN"
    RMS = "RMS"


class PresentationKind(StrEnum):
    VOLUME = "VOLUME"
    PRACTICE = "PRACTICE"
    SCORED = "SCORED"
    LOUD = "LOUD"


class ProcedurePhase(StrEnum):
    VOLUME = "VOLUME"
    PRACTICE = "PRACTICE"
    SCORED = "SCORED"
    LOUD = "LOUD"
    DONE = "DONE"


class ProcedureTerminal(StrEnum):
    VOLUME_FAILED = "VOLUME_FAILED"  # T1
    PRACTICE_FAILED = "PRACTICE_FAILED"  # T2
    LOUD_FAILED = "LOUD_FAILED"  # T3
    CEILING_STOP = "CEILING_STOP"  # T5 early stop (6 consecutive valid-wrong at +10)
    NO_RESPONSE_END = "NO_RESPONSE_END"  # T7 (3 no-responses after the 2nd passed loud check)
    COMPLETE = "COMPLETE"  # 24 scored trials


class AnswerStatus(StrEnum):
    OK = "OK"  # exactly 3 digits parsed (correctness separate)
    PARTIAL = "PARTIAL"  # 1-2 digits parsed -> wrong, valid
    DONT_KNOW = "DONT_KNOW"  # "kuch nahi suna" etc. -> wrong, valid, no re-ask
    NO_DIGITS = (
        "NO_DIGITS"  # speech transcribed, no digit, no intent -> re-ask once, then wrong valid
    )
    AMBIGUOUS = "AMBIGUOUS"  # > 3 digits -> re-ask once, then wrong valid
    REPEAT_REQUEST = "REPEAT_REQUEST"  # "phir se" -> replay once, then as NO_DIGITS
    NO_RESPONSE = "NO_RESPONSE"  # nothing and no speech energy (or no key) -> wrong, valid
    CAPTURE_MISS = (
        "CAPTURE_MISS"  # speech energy but no transcript, or breaker/reconnect -> re-present
    )
    EARLY_ANSWER = "EARLY_ANSWER"  # speech onset inside the gate, echo low -> re-ask, not scored


class ScoreAction(StrEnum):
    ACCEPT = "ACCEPT"
    REASK = "REASK"
    REPLAY = "REPLAY"


class AnswerSource(StrEnum):
    SPEECH = "SPEECH"
    DTMF = "DTMF"
    NONE = "NONE"


class WindowCloseReason(StrEnum):
    THREE_DIGITS = "THREE_DIGITS"
    SETTLE = "SETTLE"
    TIMEOUT = "TIMEOUT"
    DTMF_COMPLETE = "DTMF_COMPLETE"
    BREAKER = "BREAKER"
    SESSION_END = "SESSION_END"
    INTENT = "INTENT"  # v2: a DONT_KNOW intent with no digits closes the window at once (§6.3.1)


class SystemEvent(StrEnum):
    BREAKER_TRIP = "BREAKER_TRIP"
    RECONNECT = "RECONNECT"
    STT_ERROR = "STT_ERROR"
    MIC_LOST = (
        "MIC_LOST"  # v3 (CCR-v3-10): microphone denied/lost/stalled in this window -> CAPTURE_MISS
    )


class Censor(StrEnum):
    NONE = "NONE"
    HIGH = "HIGH"
    LOW = "LOW"


class Band(StrEnum):
    PASS = "PASS"
    PASS_WITH_NOTE = "PASS_WITH_NOTE"
    BORDERLINE = "BORDERLINE"
    REFER = "REFER"
    INCONCLUSIVE = "INCONCLUSIVE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    NOT_STARTED = "NOT_STARTED"
    DECLINED = "DECLINED"
    NO_BAND = "NO_BAND"


class TruthRule(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"
    T8 = "T8"
    T9 = "T9"
    T10 = "T10"
    T11 = "T11"
    T12 = "T12"
    T13 = "T13"
    T14 = "T14"
    T15 = "T15"
    T10B = "T10b"  # v3 (CCR-v3-32): absolute referral, evaluated after T10 and before T11


class ReferReason(StrEnum):
    LOUD_NOT_HEARD = "LOUD_NOT_HEARD"
    PRACTICE_FAILED = "PRACTICE_FAILED"
    NO_RESPONSE_LOUD = "NO_RESPONSE_LOUD"
    CANT_HEAR = "CANT_HEAR"
    CEILING = "CEILING"
    SRT = "SRT"
    SRT_ABSOLUTE = "SRT_ABSOLUTE"  # v3: T10b, srt_db >= ABS_REFER_DB with or without a norm cell


class NoBandReason(StrEnum):
    EXPERIMENTAL_LANGUAGE = "EXPERIMENTAL_LANGUAGE"
    FRIENDS_DISALLOWED = "FRIENDS_DISALLOWED"
    MIXED_MODE = "MIXED_MODE"
    SPEAKER_TRANSDUCER = "SPEAKER_TRANSDUCER"
    NO_CELL = "NO_CELL"
    CELL_LT6 = "CELL_LT6"
    NORMS_BANK_MISMATCH = "NORMS_BANK_MISMATCH"


class WordingClass(StrEnum):
    W_REF_LOUD = "W_REF_LOUD"
    W_REF_PRACTICE = "W_REF_PRACTICE"
    W_REF_CEILING = "W_REF_CEILING"
    W_REF_SRT = "W_REF_SRT"
    W_INCONC = "W_INCONC"
    W_INC = "W_INC"
    W_SYSFAIL = "W_SYSFAIL"
    W_NOREPLY = "W_NOREPLY"
    W_DECLINED = "W_DECLINED"
    W_NOBAND_LT6 = "W_NOBAND_LT6"
    W_NOBAND_RANGE = "W_NOBAND_RANGE"
    W_NOBAND_UNCOMPARABLE = "W_NOBAND_UNCOMPARABLE"
    W_BORDER = "W_BORDER"
    W_PASS_NOTE = "W_PASS_NOTE"
    W_PASS = "W_PASS"
    W_REF_ABS = (
        "W_REF_ABS"  # v3: T10b wording; slot {SRT} only (no {N}/{AGES}: there may be no norm cell)
    )


class Surface(StrEnum):
    SPOKEN = "SPOKEN"
    PAGE = "PAGE"
    REPORT = "REPORT"
    OPERATOR = "OPERATOR"


class CutoffVersion(StrEnum):
    V0 = "v0"
    V1 = "v1"  # v0: 6 <= n < 10; v1: n >= 10


class EndReason(StrEnum):
    COMPLETED = "COMPLETED"  # procedure reached a terminal (any ProcedureTerminal)
    CANT_HEAR = "CANT_HEAR"  # browser button confirmed (T4)
    CALLER_HANGUP = "CALLER_HANGUP"  # Twilio stop / browser WS closed by client after start
    OPERATOR_ABORT = "OPERATOR_ABORT"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"  # SEVI_SESSION__MAX_S
    SERVER_ERROR = "SERVER_ERROR"  # unhandled exception in the session task
    SOCKET_LOST = "SOCKET_LOST"  # WS dropped without stop/close frame
    CONSENT_NO_REPLY = (
        "CONSENT_NO_REPLY"  # phone: no reply after one repeat; browser: consent timeout
    )
    CONSENT_DECLINED = "CONSENT_DECLINED"
    DIAL_FAILED = (
        "DIAL_FAILED"  # busy / no-answer / failed / canceled, or no WS start within T_DIAL
    )
    QUEUE_ABANDONED = "QUEUE_ABANDONED"  # browser voice queue or admitted-but-never-connected


class SessionState(StrEnum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    DIALING = "DIALING"
    AWAIT_START = "AWAIT_START"
    CONSENT = "CONSENT"
    INSTRUCTIONS = "INSTRUCTIONS"
    TEST = "TEST"
    RESULT = "RESULT"
    CLOSING = "CLOSING"
    ENDED = "ENDED"  # v2: LANG/EXPLAIN/DISCLAIMER removed (playlist, §6.5)


class PresentationState(StrEnum):
    PRE_PROMPT = "PRE_PROMPT"
    SEND = "SEND"
    GATED = "GATED"
    LISTEN = "LISTEN"
    COMMIT = "COMMIT"
    SCORE = "SCORE"
    REASK_PROMPT = "REASK_PROMPT"
    FEEDBACK = "FEEDBACK"


class ConsentScript(StrEnum):
    PUBLIC_V1 = "PUBLIC_V1"
    STUDY_V1 = "STUDY_V1"


class ConsentOutcome(StrEnum):
    GIVEN = "GIVEN"
    DECLINED = "DECLINED"
    NO_REPLY = "NO_REPLY"


class ConsentMethod(StrEnum):
    DTMF_1 = "DTMF_1"
    SPEECH_YES = "SPEECH_YES"
    CLICK = "CLICK"
    DTMF_2 = "DTMF_2"
    SPEECH_NO = "SPEECH_NO"
    CLICK_DECLINE = "CLICK_DECLINE"


class AllowlistKind(StrEnum):
    TEAMMATE = "TEAMMATE"
    VOLUNTEER = "VOLUNTEER"  # VOLUNTEER rejected unless SEVI_VOLUNTEER_CALLS=true


class RelationLabel(StrEnum):
    SEVI_TEAM = "SEVI_TEAM"
    SON = "SON"
    DAUGHTER = "DAUGHTER"
    FAMILY_MEMBER = "FAMILY_MEMBER"


class AgeBand(StrEnum):
    A18_25 = "18-25"
    A26_35 = "26-35"
    A36_50 = "36-50"
    A51_65 = "51-65"
    A66_PLUS = "66+"


class KeyId(StrEnum):
    A = "A"
    B = "B"
    NONE = "NONE"  # NONE: keypad sessions (no Gnani credits)


class RetentionClass(StrEnum):
    DELETE_72H = "DELETE_72H"
    STUDY_KEEP = "STUDY_KEEP"


class EvonOutcome(StrEnum):
    LIVE = "LIVE"
    TEMPLATE = "TEMPLATE"
    SKIPPED = "SKIPPED"


class SessionFlag(StrEnum):
    ECHO_MODE = "ECHO_MODE"
    MIXED_MODE = "MIXED_MODE"
    BREAKER_TRIPPED = "BREAKER_TRIPPED"
    D_EST_DEFAULTED = "D_EST_DEFAULTED"
    MARK_LOST = "MARK_LOST"
    LEVEL_LIMITED = "LEVEL_LIMITED"
    MIC_PROCESSING = (
        "MIC_PROCESSING"  # v3 (CCR-v3-35): browser reported ns or agc applied (getSettings)
    )
    MIC_LOST = "MIC_LOST"  # v3 (CCR-v3-10): voice session fell back to keypad after mic denial/loss


class FriendsVariant(StrEnum):
    ALLOWED = "ALLOWED"
    DISALLOWED = "DISALLOWED"


class AdmissionLevel(StrEnum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class Owner(StrEnum):
    A = "A"
    B = "B"
    SHARED = "SHARED"


class PromptKey(StrEnum):
    CONSENT_PUBLIC = "CONSENT_PUBLIC"
    CONSENT_STUDY = (
        "CONSENT_STUDY"  # v2: LANG_CONFIRM removed (language chosen on the page / by the operator)
    )
    INSTRUCTIONS = "INSTRUCTIONS"
    VOLUME_INTRO = "VOLUME_INTRO"
    VOLUME_UP = "VOLUME_UP"
    PRACTICE_INTRO = "PRACTICE_INTRO"
    PRACTICE_CORRECT = "PRACTICE_CORRECT"
    PRACTICE_WRONG = "PRACTICE_WRONG"
    PRACTICE_RETRY = "PRACTICE_RETRY"
    TEST_INTRO = "TEST_INTRO"
    REASK_SAY_AGAIN = "REASK_SAY_AGAIN"
    REASK_THREE = "REASK_THREE"
    LOUD_CHECK = "LOUD_CHECK"
    SWITCH_TO_KEYPAD = "SWITCH_TO_KEYPAD"
    DISCLAIMER = "DISCLAIMER"
    GOODBYE = "GOODBYE"
    GOODBYE_EARLY = "GOODBYE_EARLY"
    DIGIT_1 = "DIGIT_1"
    DIGIT_2 = "DIGIT_2"
    DIGIT_3 = "DIGIT_3"
    DIGIT_4 = "DIGIT_4"
    DIGIT_5 = "DIGIT_5"
    DIGIT_6 = "DIGIT_6"
    DIGIT_7 = "DIGIT_7"
    DIGIT_8 = "DIGIT_8"
    DIGIT_9 = "DIGIT_9"
    # spoken wording clips: one per WordingClass, key f"WORDING_{wording_class}"
    WORDING_W_REF_LOUD = "WORDING_W_REF_LOUD"
    WORDING_W_REF_PRACTICE = "WORDING_W_REF_PRACTICE"
    WORDING_W_REF_CEILING = "WORDING_W_REF_CEILING"
    WORDING_W_REF_SRT = "WORDING_W_REF_SRT"
    WORDING_W_INC = "WORDING_W_INC"
    WORDING_W_INCONC = "WORDING_W_INCONC"
    WORDING_W_SYSFAIL = "WORDING_W_SYSFAIL"
    WORDING_W_DECLINED = "WORDING_W_DECLINED"
    WORDING_W_NOBAND_LT6 = "WORDING_W_NOBAND_LT6"
    WORDING_W_NOBAND_RANGE = "WORDING_W_NOBAND_RANGE"
    WORDING_W_NOBAND_UNCOMPARABLE = "WORDING_W_NOBAND_UNCOMPARABLE"
    WORDING_W_BORDER = "WORDING_W_BORDER"
    WORDING_W_PASS_NOTE = "WORDING_W_PASS_NOTE"
    WORDING_W_PASS = "WORDING_W_PASS"
    WORDING_W_REF_ABS = "WORDING_W_REF_ABS"  # v3 (CCR-v3-32)
    # v2: W_INC has a clip (spoken only if the person is still connected); W_NOREPLY has none
