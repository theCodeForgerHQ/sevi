"""Application settings (contracts §3, B spec §3).

One `Settings` object, loaded once at startup from the environment (prefix `SEVI_`, nested
groups joined with `__`) and optionally a `.env` file. Secrets arrive at runtime through
`infisical run --env=<dev|prod> -- <cmd>`; plain environment variables work the same way.

Scope (BUILD-BRIEF reduced scope): every §3.2/§3.3 key is kept so the spec's `.env.example`
shape survives, plus `SEVI_LEDGER__BUDGET_CREDITS_PER_KEY` (hard per-key credit budget, enforced
by `adapters/ledger.py`). Kill switch defaults follow the brief: browser voice and phone are off,
Evon is on (it falls back to the template).

`.env.example` is generated from this module (`render_env_example`, `scripts/gen_env_example.py`)
and checked by U-CFG-01.
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    IPvAnyNetwork,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from sevi.contracts.enums import KeyId, Lang

ENV_PREFIX = "SEVI_"
NESTED_DELIMITER = "__"
EXIT_CONFIG_INVALID = 78

SCRYPT_RE = re.compile(r"^scrypt\$16384\$8\$1\$[0-9a-f]+\$[0-9a-f]+$")
ACCOUNT_SID_RE = re.compile(r"^AC[0-9a-f]{32}$")
E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
PARTICIPANT_RE = re.compile(r"^P-\d{2}$")


def _split_csv(v: Any) -> Any:
    """Comma list (contracts §3.1, R-48): no spaces, brackets or quotes; '' = empty list."""
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return v


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


def _secret_set(s: SecretStr | None) -> bool:
    return s is not None and s.get_secret_value() != ""


# ---------- groups (contracts §3.3) ----------
class GnaniSettings(_Group):
    key_a: SecretStr | None = None  # required in prod
    key_b: SecretStr | None = None  # empty => Key B aliases Key A
    rest_base: AnyHttpUrl = AnyHttpUrl("https://api.vachana.ai")
    key_a_permits: int = Field(1, ge=1, le=8)
    key_b_permits: int = Field(1, ge=1, le=8)
    stream_rest_coexist: bool = False

    @field_validator("rest_base")
    @classmethod
    def _https(cls, v: AnyHttpUrl) -> AnyHttpUrl:
        if v.scheme != "https":
            raise ValueError("rest_base must be https")
        return v


KeytermsHeader = Literal["keyterms", "x-keyterms", "x-keywords"]
KEYTERMS_DEFAULT: tuple[KeytermsHeader, ...] = ("keyterms", "x-keyterms", "x-keywords")


class PrismaSettings(_Group):
    enabled: bool = True
    ws_url: str = "wss://api.vachana.ai/stt/v3/stream"
    rest_url: AnyHttpUrl = AnyHttpUrl("https://api.vachana.ai/stt/v3")
    format: Literal["transcribe", "codemix", "verbatim"] = "transcribe"
    header_mode: Literal["both", "doc", "sdk"] = "both"
    keyterms_enabled: bool = True
    keyterms_headers: Annotated[list[KeytermsHeader], NoDecode] = Field(
        default_factory=lambda: list(KEYTERMS_DEFAULT)
    )
    send_enable_itn: bool = True
    vad_threshold: float = Field(0.7, ge=0, le=1)
    min_silence_ms: int = Field(800, ge=100, le=2000)
    min_speech_ms: int = Field(100, ge=50, le=2000)
    min_segment_ms: int = Field(100, ge=100, le=1000)
    idle_timeout_s: int = Field(900, ge=5, le=900)
    sample_rate_hz: int = 8000  # 8000 | 16000; == bank chain.prisma_sample_rate_hz
    connect_timeout_s: float = Field(3.0, ge=0.5, le=10)
    commit_emits_processing: bool = True
    send_while_gated: bool = True
    late_grace_ms: int = Field(300, ge=0, le=1000)
    t_commit_max_ms: int = Field(1200, ge=300, le=2000)
    rest_timeout_s: float = Field(15.0, ge=1, le=60)

    _csv = field_validator("keyterms_headers", mode="before")(_split_csv)

    @field_validator("ws_url")
    @classmethod
    def _wss(cls, v: str) -> str:
        if not v.startswith("wss://"):
            raise ValueError("ws_url must start with wss://")
        return v

    @field_validator("rest_url")
    @classmethod
    def _https(cls, v: AnyHttpUrl) -> AnyHttpUrl:
        if v.scheme != "https":
            raise ValueError("rest_url must be https")
        return v

    @field_validator("sample_rate_hz")
    @classmethod
    def _rate(cls, v: int) -> int:
        if v not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        return v

    @field_validator("keyterms_headers")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("keyterms_headers must be non-empty")
        return v


class EvonSettings(_Group):
    enabled: bool = True  # kill switch; off or failing => template text
    base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8080/v1")
    api_key: SecretStr | None = None  # required in prod
    model: str = "evon"
    timeout_s: float = Field(6.0, ge=1, le=30)
    connect_timeout_s: float = Field(1.0, ge=0.2, le=5)
    max_tokens_explain: int = Field(256, ge=16, le=1024)
    max_tokens_report: int = Field(640, ge=16, le=2048)
    max_tokens_translation: int = Field(1024, ge=16, le=4096)
    temperature: float = Field(0.6, ge=0, le=1.5)
    top_p: float = Field(0.95, ge=0, le=1)
    repeat_penalty: float = Field(1.0, ge=1.0, le=1.5)
    response_format_style: Literal["llamacpp", "openai"] = "llamacpp"
    schema_pattern: bool = True
    health_interval_s: int = Field(30, ge=5, le=300)
    max_concurrency: int = Field(1, ge=1, le=8)
    browser_deadline_s: int = Field(10, ge=2, le=30)
    report_deadline_s: int = Field(6, ge=2, le=30)

    @field_validator("base_url")
    @classmethod
    def _v1(cls, v: AnyHttpUrl) -> AnyHttpUrl:
        if not str(v).rstrip("/").endswith("/v1"):
            raise ValueError("base_url must end with /v1")
        return v


class TwilioSettings(_Group):
    account_sid: str | None = None
    auth_token: SecretStr | None = None
    from_number: str | None = None
    validate_signature: bool = True
    ring_timeout_s: int = Field(30, ge=5, le=60)
    time_limit_s: int = Field(600, ge=60, le=600)
    trial_minutes_total: float = 75
    trial_minutes_stop: float = 65
    record_answers: bool = False
    # Trial: only verified caller IDs can be dialled. Comma list of `P-NN:+E164` pairs
    # (participant id -> number). Numbers are never logged.
    verified_to_numbers: Annotated[list[str], NoDecode] = Field(default_factory=list)
    audio_ttl_s: int = Field(120, ge=10, le=900)  # /audio stays valid this long after the call ends

    @field_validator("verified_to_numbers", mode="before")
    @classmethod
    def _csv(cls, v: Any) -> Any:
        return [] if v is None else _split_csv(v)

    @field_validator("verified_to_numbers")
    @classmethod
    def _pairs(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for item in v:
            pid, _, num = item.partition(":")
            if not PARTICIPANT_RE.match(pid) or not E164_RE.match(num):
                raise ValueError("verified_to_numbers entries must be P-NN:+E164")
            if pid in seen:
                raise ValueError(f"duplicate participant id {pid} in verified_to_numbers")
            seen.add(pid)
        return v

    def verified_numbers(self) -> dict[str, str]:
        """participant_id -> E.164 (the trial's verified caller IDs)."""
        return dict(item.split(":", 1) for item in self.verified_to_numbers)

    @field_validator("account_sid")
    @classmethod
    def _sid(cls, v: str | None) -> str | None:
        if v is not None and not ACCOUNT_SID_RE.match(v):
            raise ValueError("account_sid must match ^AC[0-9a-f]{32}$")
        return v

    @field_validator("from_number")
    @classmethod
    def _e164(cls, v: str | None) -> str | None:
        if v is not None and not E164_RE.match(v):
            raise ValueError("from_number must be E.164")
        return v


class PhoneSettings(_Group):
    enabled: bool = False  # trial IVR demo; off by default (BUILD-BRIEF)


class BrowserVoiceSettings(_Group):
    enabled: bool = False  # BUILD-BRIEF: kill switch defaults off (keypad only)


class AdmissionSettings(_Group):
    max_keypad_sessions: int = Field(120, ge=1, le=1000)
    voice_queue_visible_max: int = Field(3, ge=0, le=20)
    per_ip_keypad_per_hour: int = Field(20, ge=1, le=1000)
    per_ip_voice_per_hour: int = Field(3, ge=1, le=100)
    ip_hash_key: SecretStr | None = None  # >= 32 bytes; required in prod
    voice_eta_s: int = 240
    waiting_room_retry_s: int = Field(60, ge=10, le=600)
    queue_poll_timeout_s: int = 30
    admitted_connect_timeout_s: int = 120
    admitted_connect_timeout_keypad_s: int = Field(20, ge=5, le=120)
    pre_start_sockets_per_ip: int = Field(2, ge=1, le=20)

    @field_validator("ip_hash_key")
    @classmethod
    def _len(cls, v: SecretStr | None) -> SecretStr | None:
        if _secret_set(v) and len(v.get_secret_value().encode()) < 32:  # type: ignore[union-attr]
            raise ValueError("ip_hash_key must be at least 32 bytes")
        return v


class CaptureSettings(_Group):
    t_guard_browser_ms: int = Field(50, ge=0, le=300)
    t_answer_ms: int = Field(8000, ge=3000, le=15000)
    t_settle_ms: int = Field(900, ge=300, le=2000)
    speech_min_ms: int = Field(300, ge=60, le=2000)
    speech_db_above_floor: float = Field(12.0, ge=3, le=30)
    echo_high_db: float = Field(10.0, ge=3, le=30)
    d_est_default_browser_ms: int = 150
    d_est_max_ms: int = 1500
    mark_timeout_extra_ms: int = 3000


class SessionSettings(_Group):
    t_consent_phone_s: int = 10
    t_consent_browser_s: int = 300
    t_start_s: int = Field(5, ge=2, le=30)
    t_rx_idle_s: int = Field(35, ge=20, le=120)
    send_timeout_s: float = Field(5.0, ge=1, le=15)
    t_dial_s: int = 60
    max_s: int = 720
    instruction_repeats_max: int = 2


class ResultSettings(_Group):
    bank_dir: Path = Path("./stimuli")
    frozen_bank_hashes: dict[Lang, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        default_factory=dict
    )
    allow_unfrozen_bank: bool = False
    norms_path: Path = Path("./results/norms.json")
    enabled_langs: Annotated[list[Lang], NoDecode] = Field(
        default_factory=lambda: [Lang.TA, Lang.EN]
    )
    experimental_langs: Annotated[list[Lang], NoDecode] = Field(default_factory=lambda: [Lang.EN])
    hhie_enabled: bool = False

    _csv = field_validator("enabled_langs", "experimental_langs", mode="before")(_split_csv)

    @field_validator("enabled_langs")
    @classmethod
    def _non_empty(cls, v: list[Lang]) -> list[Lang]:
        if not v:
            raise ValueError("enabled_langs must be non-empty")
        return v

    @model_validator(mode="after")
    def _en_experimental(self) -> ResultSettings:
        if Lang.EN in self.enabled_langs and Lang.EN not in self.experimental_langs:
            raise ValueError("experimental_langs must contain en whenever en is enabled")
        if self.hhie_enabled:
            raise ValueError("hhie_enabled: the HHIE questionnaire is not built (cut)")
        return self


class RetentionSettings(_Group):
    hours: int = Field(72, ge=1, le=72)
    job_interval_s: int = Field(600, ge=60, le=3600)
    twilio_delete: bool = True


class OperatorSettings(_Group):
    username: str = Field("sevi-op", min_length=3, max_length=32)
    password_scrypt: SecretStr | None = None  # scrypt$16384$8$1$<salt_hex>$<hash_hex>
    allowed_cidrs: Annotated[list[IPvAnyNetwork], NoDecode] = Field(
        default_factory=lambda: [
            ipaddress.ip_network("100.64.0.0/10"),
            ipaddress.ip_network("127.0.0.1/32"),
        ]
    )

    _csv = field_validator("allowed_cidrs", mode="before")(_split_csv)

    @field_validator("password_scrypt")
    @classmethod
    def _scrypt(cls, v: SecretStr | None) -> SecretStr | None:
        if _secret_set(v) and not SCRYPT_RE.match(v.get_secret_value()):  # type: ignore[union-attr]
            raise ValueError("password_scrypt must be scrypt$16384$8$1$<salt_hex>$<hash_hex>")
        return v


class AllowlistSettings(_Group):
    fernet_key: SecretStr | None = None  # required in prod
    hmac_key: SecretStr | None = None  # >= 32 bytes; required in prod

    @field_validator("hmac_key")
    @classmethod
    def _len(cls, v: SecretStr | None) -> SecretStr | None:
        if _secret_set(v) and len(v.get_secret_value().encode()) < 32:  # type: ignore[union-attr]
            raise ValueError("hmac_key must be at least 32 bytes")
        return v


class LedgerSettings(_Group):
    model_config = ConfigDict(extra="forbid", validate_default=True, allow_inf_nan=False)

    credits_per_stt_s: float | None = Field(None, gt=0)  # required in prod
    credits_per_tts_char: float | None = Field(None, gt=0)  # required in prod
    key_a_total: float = Field(5000, gt=0)
    key_b_total: float = Field(5000, gt=0)
    reserve_frac: float = Field(0.10, ge=0, le=0.5)
    key_a_cal_stop_frac: float = Field(0.55, ge=0, le=0.9)
    key_b_public_stop_sessions: int = Field(20, ge=0)
    voice_session_credits_est: float | None = None  # required in prod
    # BUILD-BRIEF credit safety: hard per-key budget. A Gnani call whose reservation would take
    # a key's spend past this figure is refused (BudgetExceeded). 0 = refuse every billed call
    # (fail closed); prod requires > 0.
    budget_credits_per_key: float = Field(0.0, ge=0)


class CalSettings(_Group):
    cache_dir: Path = Path("./var/cal-cache")
    max_concurrency: int = Field(1, ge=1, le=4)
    repeats: int = Field(1, ge=1, le=5)


# ---------- root (contracts §3.2) ----------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=NESTED_DELIMITER,
        env_file=".env",
        extra="forbid",
        env_parse_none_str="",
        validate_default=True,
    )

    env: Literal["dev", "test", "demo", "prod"] = "dev"
    demo_mode: bool = False
    public_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8000")
    pages_origin: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8000")
    bind_host: IPvAnyAddress = IPv4Address("127.0.0.1")
    bind_port: int = Field(8000, ge=1024, le=65535)
    op_bind_host: IPvAnyAddress = IPv4Address("127.0.0.1")
    op_bind_port: int = Field(8001, ge=1024, le=65535)
    trusted_proxies: Annotated[list[IPvAnyNetwork], NoDecode] = Field(
        default_factory=lambda: [
            ipaddress.ip_network("127.0.0.1/32"),
            ipaddress.ip_network("::1/128"),
        ]
    )
    client_ip_header: str = "X-Forwarded-For"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("./var/logs")
    log_retention_hours: int = Field(72, ge=1, le=72)
    data_dir: Path = Path("./var/data")
    drain_s: int = Field(30, ge=0, le=120)
    friends: Literal["ALLOWED", "DISALLOWED"] = "ALLOWED"
    volunteer_calls: bool = False

    gnani: GnaniSettings = Field(default_factory=GnaniSettings)
    prisma: PrismaSettings = Field(default_factory=PrismaSettings)
    evon: EvonSettings = Field(default_factory=EvonSettings)
    twilio: TwilioSettings = Field(default_factory=TwilioSettings)
    phone: PhoneSettings = Field(default_factory=PhoneSettings)
    browser_voice: BrowserVoiceSettings = Field(default_factory=BrowserVoiceSettings)
    admission: AdmissionSettings = Field(default_factory=AdmissionSettings)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    result: ResultSettings = Field(default_factory=ResultSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    operator: OperatorSettings = Field(default_factory=OperatorSettings)
    allowlist: AllowlistSettings = Field(default_factory=AllowlistSettings)
    ledger: LedgerSettings = Field(default_factory=LedgerSettings)
    cal: CalSettings = Field(default_factory=CalSettings)

    _csv = field_validator("trusted_proxies", mode="before")(_split_csv)

    @field_validator("bind_host")
    @classmethod
    def _loopback(cls, v: IPv4Address | IPv6Address) -> IPv4Address | IPv6Address:
        if not v.is_loopback:
            raise ValueError("bind_host must be loopback (the ingress connects locally)")
        return v

    @property
    def key_b_aliased(self) -> bool:
        kb = self.gnani.key_b
        return kb is None or kb.get_secret_value() == ""

    @property
    def public_key(self) -> KeyId:
        return KeyId.A if self.key_b_aliased else KeyId.B

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "sevi.sqlite"

    @model_validator(mode="after")
    def _invariants(self) -> Settings:
        if self.op_bind_port == self.bind_port:
            raise ValueError("op_bind_port must differ from bind_port")
        if not self.twilio.validate_signature and self.env != "test":
            raise ValueError("twilio.validate_signature may be false only when env=test")
        if self.cal.max_concurrency > self.gnani.key_a_permits:
            raise ValueError("cal.max_concurrency must be <= gnani.key_a_permits")
        if self.result.allow_unfrozen_bank and self.env not in ("dev", "test", "demo"):
            raise ValueError("result.allow_unfrozen_bank is allowed only in dev/test/demo")
        if self.phone.enabled:
            t = self.twilio
            if not (t.account_sid and _secret_set(t.auth_token) and t.from_number):
                raise ValueError(
                    "phone.enabled requires twilio.account_sid, twilio.auth_token, "
                    "twilio.from_number"
                )
        if self.env == "prod":
            self._prod_invariants()
        return self

    def _prod_invariants(self) -> None:
        problems: list[str] = []
        for name in ("public_base_url", "pages_origin"):
            if getattr(self, name).scheme != "https":
                problems.append(f"{name} must be https in prod")
        if self.demo_mode:
            problems.append("demo_mode must be false in prod")
        if self.log_level == "DEBUG":
            problems.append("log_level must not be DEBUG in prod")
        required_secrets = {
            "gnani.key_a": self.gnani.key_a,
            "evon.api_key": self.evon.api_key,
            "admission.ip_hash_key": self.admission.ip_hash_key,
            "operator.password_scrypt": self.operator.password_scrypt,
            "allowlist.fernet_key": self.allowlist.fernet_key,
            "allowlist.hmac_key": self.allowlist.hmac_key,
        }
        problems += [
            f"{k} is required in prod" for k, v in required_secrets.items() if not _secret_set(v)
        ]
        led = self.ledger
        for name in ("credits_per_stt_s", "credits_per_tts_char", "voice_session_credits_est"):
            if getattr(led, name) is None:
                problems.append(f"ledger.{name} is required in prod")
        if led.budget_credits_per_key <= 0:
            problems.append("ledger.budget_credits_per_key must be > 0 in prod")
        if problems:
            raise ValueError("; ".join(problems))


# ---------- runtime switches (B spec §3) ----------
@dataclass
class Switches:
    """In-memory kill switches, initialised from settings; sessions read them once at creation."""

    evon: bool
    prisma: bool
    timbre_runtime: bool
    phone: bool
    browser_voice: bool
    demo_mode: bool
    friends: bool

    @classmethod
    def from_settings(cls, s: Settings) -> Switches:
        return cls(
            evon=s.evon.enabled,
            prisma=s.prisma.enabled,
            timbre_runtime=False,
            phone=s.phone.enabled,
            browser_voice=s.browser_voice.enabled,
            demo_mode=s.demo_mode,
            friends=s.friends == "ALLOWED",
        )


def load_settings(**overrides: Any) -> Settings:
    """Load from the environment (and `.env` if present). Callers exit 78 on ValidationError."""
    return Settings(**overrides)


# ---------- .env.example generation (U-CFG-01) ----------
_HEADER = """\
# Sevi configuration: names and non-secret defaults only. Never put real secrets in this file.
# Generated from src/sevi/app/settings.py by `uv run python scripts/gen_env_example.py`
# (test U-CFG-01 checks they agree). Secrets are injected at runtime by Infisical (project
# `sevi`): `infisical run --env=dev -- uv run sevi ...`. Plain environment variables work too.
"""


def _render_value(v: Any) -> str:
    if v is None or isinstance(v, SecretStr):
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Enum):
        return str(v.value)
    if isinstance(v, AnyHttpUrl):
        s = str(v)
        return s[:-1] if v.path == "/" and s.endswith("/") else s
    if isinstance(v, Path):
        return v.as_posix() if v.is_absolute() else f"./{v.as_posix()}"
    if isinstance(v, list):
        return ",".join(_render_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps({_render_value(k): x for k, x in v.items()}, separators=(",", ":"))
    return str(v)


def env_entries() -> list[tuple[str, str, str]]:
    """(group, env name, default rendering) for every Settings field, in declaration order."""
    defaults = Settings.model_construct()
    out: list[tuple[str, str, str]] = []
    for name, info in Settings.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, _Group):
            group = getattr(defaults, name)
            if not isinstance(group, ann):  # model_construct skips default_factory
                group = ann()
            for sub in ann.model_fields:
                env = f"{ENV_PREFIX}{name}{NESTED_DELIMITER}{sub}".upper()
                out.append((name, env, _render_value(getattr(group, sub))))
        else:
            default = info.get_default(call_default_factory=True)
            out.append(("top", f"{ENV_PREFIX}{name}".upper(), _render_value(default)))
    return out


def env_names() -> list[str]:
    return [e for _, e, _ in env_entries()]


def render_env_example() -> str:
    lines = [_HEADER.rstrip("\n")]
    current = None
    for group, env, value in env_entries():
        if group != current:
            lines.append(f"# ---- {group}")
            current = group
        lines.append(f"{env}={value}")
    return "\n".join(lines) + "\n"
