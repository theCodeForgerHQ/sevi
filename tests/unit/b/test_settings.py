"""Settings (contracts §3, B spec §3): U-CFG-01/02/03/04/08/09."""

from __future__ import annotations

import ipaddress
from pathlib import Path

import pytest
from pydantic import ValidationError

from sevi.app.settings import Settings, Switches, env_names, render_env_example
from sevi.contracts.enums import KeyId, Lang

ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLE = ROOT / ".env.example"

PROD_ENV = {
    "SEVI_ENV": "prod",
    "SEVI_PUBLIC_BASE_URL": "https://sevi.example.ts.net",
    "SEVI_PAGES_ORIGIN": "https://thecodeforgerhq.github.io",
    "SEVI_GNANI__KEY_A": "dummy-key-a",
    "SEVI_EVON__API_KEY": "dummy-evon",
    "SEVI_ADMISSION__IP_HASH_KEY": "x" * 32,
    "SEVI_OPERATOR__PASSWORD_SCRYPT": "scrypt$16384$8$1$00ff$abcd",
    "SEVI_ALLOWLIST__FERNET_KEY": "dummy-fernet",
    "SEVI_ALLOWLIST__HMAC_KEY": "h" * 32,
    "SEVI_LEDGER__CREDITS_PER_STT_S": "0.118",
    "SEVI_LEDGER__CREDITS_PER_TTS_CHAR": "0.01",
    "SEVI_LEDGER__VOICE_SESSION_CREDITS_EST": "40",
    "SEVI_LEDGER__BUDGET_CREDITS_PER_KEY": "4500",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os

    for k in list(os.environ):
        if k.startswith("SEVI_") and k != "SEVI_ALLOW_LIVE":
            monkeypatch.delenv(k)
    monkeypatch.chdir(tmp_path)  # no stray .env


def _example_keys() -> list[str]:
    keys = []
    for line in ENV_EXAMPLE.read_text().splitlines():
        if line and not line.startswith("#"):
            keys.append(line.split("=", 1)[0])
    return keys


def test_U_CFG_01_env_example_matches_settings_both_directions() -> None:
    keys = _example_keys()
    assert len(keys) == len(set(keys))
    assert set(keys) == set(env_names())
    # and the committed file is exactly what the generator produces
    assert ENV_EXAMPLE.read_text() == render_env_example()


def _secret_env_names() -> set[str]:
    from pydantic import BaseModel, SecretStr

    out = set()
    for name, info in Settings.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            for sub, sinfo in ann.model_fields.items():
                if "SecretStr" in str(sinfo.annotation) or sinfo.annotation is SecretStr:
                    out.add(f"SEVI_{name}__{sub}".upper())
    return out


def test_U_CFG_01_env_example_holds_no_secret_values() -> None:
    secrets = _secret_env_names()
    assert "SEVI_GNANI__KEY_A" in secrets
    assert "SEVI_TWILIO__AUTH_TOKEN" in secrets
    for line in ENV_EXAMPLE.read_text().splitlines():
        name, _, value = line.partition("=")
        if name in secrets:
            assert value == "", name


def test_U_CFG_02_env_example_loads_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in PROD_ENV.items():
        monkeypatch.setenv(k, v)
    s = Settings(_env_file=ENV_EXAMPLE)  # type: ignore[call-arg]
    assert s.env == "prod"
    assert s.trusted_proxies == [
        ipaddress.ip_network("127.0.0.1/32"),
        ipaddress.ip_network("::1/128"),
    ]
    assert s.prisma.keyterms_headers == ["keyterms", "x-keyterms", "x-keywords"]
    assert s.result.enabled_langs == [Lang.TA, Lang.EN]
    assert s.result.experimental_langs == [Lang.EN]
    assert s.client_ip_header == "X-Forwarded-For"
    assert s.result.frozen_bank_hashes == {}
    assert s.admission.max_keypad_sessions == 120
    assert s.session.t_start_s == 5
    assert s.session.t_rx_idle_s == 35
    assert s.session.send_timeout_s == 5.0
    assert s.phone.enabled is False
    assert s.result.hhie_enabled is False
    assert s.ledger.budget_credits_per_key == 4500


def test_plain_env_vars_without_any_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEVI_GNANI__KEY_A_PERMITS", "2")
    monkeypatch.setenv("SEVI_RESULT__ENABLED_LANGS", "ta")
    monkeypatch.setenv("SEVI_RESULT__FROZEN_BANK_HASHES", '{"ta":"' + "a" * 64 + '"}')
    monkeypatch.setenv("SEVI_LEDGER__BUDGET_CREDITS_PER_KEY", "100")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.gnani.key_a_permits == 2
    assert s.result.enabled_langs == [Lang.TA]
    assert s.result.frozen_bank_hashes == {Lang.TA: "a" * 64}
    assert s.ledger.budget_credits_per_key == 100


def test_kill_switch_defaults_per_brief() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.browser_voice.enabled is False
    assert s.phone.enabled is False
    assert s.evon.enabled is True
    assert s.ledger.budget_credits_per_key == 0.0  # fail closed: no billed call without a budget
    sw = Switches.from_settings(s)
    assert (sw.browser_voice, sw.phone, sw.evon) == (False, False, True)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"SEVI_PAGES_ORIGIN": "http://x.github.io"}, "pages_origin"),
        ({"SEVI_LOG_LEVEL": "DEBUG"}, "log_level"),
        ({"SEVI_DEMO_MODE": "true"}, "demo_mode"),
        ({"SEVI_BIND_HOST": "0.0.0.0"}, "bind_host"),  # noqa: S104
        ({"SEVI_LEDGER__CREDITS_PER_STT_S": ""}, "credits_per_stt_s"),
        ({"SEVI_LEDGER__BUDGET_CREDITS_PER_KEY": "0"}, "budget_credits_per_key"),
        ({"SEVI_GNANI__KEY_A": ""}, "key_a"),
        ({"SEVI_RESULT__ALLOW_UNFROZEN_BANK": "true"}, "allow_unfrozen_bank"),
        ({"SEVI_TWILIO__VALIDATE_SIGNATURE": "false"}, "validate_signature"),
    ],
)
def test_U_CFG_03_prod_invariants(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, str], field: str
) -> None:
    for k, v in {**PROD_ENV, **override}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError, match=field):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_U_CFG_04_key_b_empty_aliases_key_a(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEVI_GNANI__KEY_B", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.key_b_aliased
    assert s.public_key is KeyId.A
    monkeypatch.setenv("SEVI_GNANI__KEY_B", "dummy-b")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert not s.key_b_aliased
    assert s.public_key is KeyId.B


def test_U_CFG_08_en_must_be_experimental_and_hhie_not_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEVI_RESULT__ENABLED_LANGS", "ta,en")
    monkeypatch.setenv("SEVI_RESULT__EXPERIMENTAL_LANGS", "")
    with pytest.raises(ValidationError, match="experimental_langs"):
        Settings(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.setenv("SEVI_RESULT__EXPERIMENTAL_LANGS", "en")
    monkeypatch.setenv("SEVI_RESULT__HHIE_ENABLED", "true")
    with pytest.raises(ValidationError, match="hhie"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_U_CFG_09_unknown_prefixed_key_in_env_file_rejected(tmp_path: Path) -> None:
    env = tmp_path / "bad.env"
    env.write_text("SEVI_OP_USER=x\n")
    with pytest.raises(ValidationError):
        Settings(_env_file=env)  # type: ignore[call-arg]


def test_U_CFG_09_unprefixed_variables_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEVIOP_USER", "someone")
    monkeypatch.setenv("SEVIOP_PASS", "pw")
    assert Settings(_env_file=None) == Settings(_env_file=None)  # type: ignore[call-arg]


def test_phone_enabled_requires_twilio_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEVI_PHONE__ENABLED", "true")
    with pytest.raises(ValidationError, match="twilio"):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("SEVI_GNANI__KEY_A_PERMITS", "9"),
        ("SEVI_LOG_RETENTION_HOURS", "73"),
        ("SEVI_PRISMA__WS_URL", "ws://insecure"),
        ("SEVI_EVON__BASE_URL", "http://127.0.0.1:8080"),
        ("SEVI_TWILIO__ACCOUNT_SID", "AC123"),
        ("SEVI_OP_BIND_PORT", "8000"),
        ("SEVI_LEDGER__BUDGET_CREDITS_PER_KEY", "-1"),
        ("SEVI_PRISMA__KEYTERMS_HEADERS", ""),
    ],
)
def test_field_validation(monkeypatch: pytest.MonkeyPatch, env: str, value: str) -> None:
    monkeypatch.setenv(env, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
