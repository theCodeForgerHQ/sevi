"""Schemas of committed ``results/*`` files (contracts §4.10), reduced scope.

Every JSON results file has a ``meta`` block; every CSV has a sibling ``<name>.meta.json`` with
the same :class:`ResultFileMeta`.

Omitted in this build (their producers are cut, REVIEW §4): listener_fit / listener_curves
(Prisma calibration grid), panel_outcome, power_sim, fit_recovery, channel_check, validation,
preregistration, prisma8k_bench. ``norms.json`` (:class:`~sevi.contracts.result.NormsFile`)
stays in ``result.py`` because the truth table is defined over it.
"""

from typing import Literal

from pydantic import Field

from .enums import Channel, FriendsVariant, Lang, Mode, Owner
from .types import Fraction, GitSha, SemVer, SeviModel, Sha256Hex, UtcDatetime


class ResultFileMeta(SeviModel):
    schema_id: str = Field(pattern=r"^sevi\.results\.[a-z0-9_]+/\d+$")
    generated_at: UtcDatetime
    git_sha: GitSha
    generator: str  # exact command, e.g. "make results"
    owner: Owner
    contracts_version: SemVer
    bank_hash: Sha256Hex | None = None
    n_rows: int | None = None  # CSV siblings only


class Estimate(SeviModel):
    est: float
    ci_lo: float | None = None  # 95 % CI
    ci_hi: float | None = None


# ---------- A-owned ----------
class StaircaseSimCell(SeviModel):
    slope_per_db: float
    theta_db: float
    n_runs: int
    bias_db: float
    sd_db: float


class StaircaseSimFile(SeviModel):  # results/staircase_sim.json (EV-ST-01)
    meta: ResultFileMeta
    cells: tuple[StaircaseSimCell, ...]
    max_abs_bias_db: float
    max_sd_db: float
    passed: bool  # |bias| <= 0.5 and sd <= 1.2 for slopes 12-25 %/dB


class ScoringLangStats(SeviModel):
    lang: Lang
    n_answers: int
    triplet_exact: Fraction
    per_digit: Fraction
    # "western_concat","western_spaced","native_numerals","native_words","english_words",
    # "tanglish","english_in_tamil_script","compound","grouping","unparsed","none"
    by_transcript_form: dict[str, int]
    split_segment_rate: Fraction
    early_answer_rate: Fraction
    confusion: dict[str, dict[str, int]]  # presented digit -> recognised digit/"miss" -> count


class ScoringAccuracyFile(SeviModel):  # results/scoring_accuracy.json (EXP-4)
    meta: ResultFileMeta
    langs: tuple[ScoringLangStats, ...]


GateVerdict = Literal["PASS", "FAIL", "PENDING"]
DiscordReply = Literal["YES", "NO", "NO_REPLY"]


class GatesFile(SeviModel):  # results/gates.json (hand-entered via `sevi gate set`)
    meta: ResultFileMeta
    g1a: GateVerdict
    g1b: GateVerdict  # FAIL -> copy SHIP = PRISMA_UNUSABLE
    g2: GateVerdict
    g3t: GateVerdict  # trial phone gate (stretch); PHONE = OK iff PASS
    g4: GateVerdict
    g5: GateVerdict
    g6_permits: int | None
    g7: Literal["PASS", "PARTIAL", "FAIL", "PENDING"]
    echo_criterion: GateVerdict
    discord_q1: DiscordReply
    discord_q2: DiscordReply
    discord_q3: Literal["YES_WRITTEN", "NO", "NO_REPLY"]
    discord_q4: DiscordReply
    discord_q5: DiscordReply = "NO_REPLY"  # credits valid after 10 Nov
    discord_q6: DiscordReply = "NO_REPLY"  # permits / concurrency per key
    discord_q7: DiscordReply = "NO_REPLY"  # does Gnani retain answer audio? YES = retained
    friends: FriendsVariant
    phone: Literal["OK", "BROWSER_ONLY"]
    evon: Literal["LIVE", "TEMPLATE"]
    version: Literal["v3", "v2.5"]
    ing: GateVerdict  # VT-ING-01 tunnel ingress
    twilio_record_ok: bool | None  # VT-TWL-03
    t_guard_phone_ms: None = None  # phone has no gate (§3.4); kept for schema stability
    t_commit_max_ms: int | None
    lt05: GateVerdict = "PENDING"  # LT-05 is cut in this build; stays PENDING


# ---------- B-owned ----------
class LatencyRow(SeviModel):  # results/latency.csv (EXP-8)
    span: str
    source: Literal["real_phone", "real_browser", "load"]
    channel: Channel
    mode: Mode
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float


class EvonSchemaFile(SeviModel):  # results/evon_schema.json (EXP-9)
    meta: ResultFileMeta
    n_requests: int
    first_try_valid: Fraction
    after_retry_valid: Fraction
    by_lang: dict[str, Fraction]
    host: Literal["laptop-iq4xs", "kaggle-2xt4-q4km"]
    tokens_per_s_p50: float
    latency_p95_ms: float


class EvonGuardFile(SeviModel):  # results/evon_guard.json (EXP-9)
    meta: ResultFileMeta
    n_outputs: int
    rejected: dict[str, int]  # GuardReason -> count
    rejection_rate: Fraction
    template_fallbacks: int
    numerals_reaching_tts: Literal[0]  # hard invariant; any other value fails validation
    by_lang: dict[str, Fraction]


class UsageFile(SeviModel):  # results/usage.json (EXP-10)
    meta: ResultFileMeta
    by_key: dict[str, dict[str, float]]  # "A"|"B" -> {"stt_s","tts_chars","stt_rest_calls",...}
    twilio_minutes: float
    twilio_usd: float
    gnani_share_by_component: dict[str, Fraction]
    # "A.CAL","A.OPS","A.RESERVE","B.DEV","B.PUBLIC","B.RESERVE" -> {"stt_s","tts_chars",...}
    by_bucket: dict[str, dict[str, float]] = Field(default_factory=dict)


class OpsFile(SeviModel):  # results/ops.json (EXP-11)
    meta: ResultFileMeta
    sessions: dict[str, int]  # "browser/keypad", "browser/voice", "phone/voice", ...
    completion_rate: Fraction
    bands: dict[str, int]
    admission_levels_seen: dict[str, int]


class CaptureFile(SeviModel):  # results/capture.json (EXP-12)
    meta: ResultFileMeta
    n_windows: int
    response_presence_rate: Fraction
    capture_rate: Fraction  # windows with speech_present that yielded a transcript
    capture_miss_count: int
    inconclusive_rate: Fraction
    stale_segments: int
    early_answer_rate: Fraction
    echo_mode_sessions: int


class BrowserMatrixRow(SeviModel):
    platform: Literal["android", "ios", "macos", "windows", "linux"]
    browser: Literal["chrome", "safari", "firefox", "edge"]
    browser_version: str
    transducer: Literal["wired", "bluetooth", "laptop_speakers"]
    mode: Literal["keypad", "voice"]
    # keys: full_session, autoplay_unlock, mic_denied_fallback, speaker_echo_warning, result_page
    checks: dict[str, Literal["PASS", "FAIL", "N/A"]]
    tester: Literal["A", "B"]
    tested_at: UtcDatetime
    notes: str = ""


class BrowserMatrixFile(SeviModel):  # results/browser_matrix.json (G7, EXP-14)
    meta: ResultFileMeta
    rows: tuple[BrowserMatrixRow, ...]
    cta_platforms: tuple[str, ...]  # platforms allowed in the CTA


class LoadRun(SeviModel):  # one row per load test (LT-05 is cut in this build)
    test_id: Literal["LT-01", "LT-05"]  # LT-01 loopback; LT-05 through the tunnel
    path: Literal["loopback", "tunnel"]
    concurrent_sessions: int
    http_pollers: int = 0
    duration_s: int
    stim_render_p95_ms: float | None
    turn_total_p95_ms: float | None
    errors: int
    ws_disconnects: int
    non_2xx: int = 0
    over_cap_behaviour: str = ""  # LT-05: what the tunnel did above its in-flight cap
    # LT-01: n >= MAX_KEYPAD_SESSIONS, render p95 <= 50, turn p95 <= 1500, errors == 0
    passed: bool


class LoadFile(SeviModel):  # results/load.json (LT-01, LT-05)
    meta: ResultFileMeta
    runs: tuple[LoadRun, ...]


class CopySelectionFile(SeviModel):  # results/copy_selection.json (§4.11)
    meta: ResultFileMeta
    outcome_values: dict[str, str]  # OutcomeVar -> value
    selected: dict[str, str]  # copy key -> variant_id
    copy_sha256: Sha256Hex
