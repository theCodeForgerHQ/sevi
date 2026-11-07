"""structlog configuration and PII redaction (contracts §10.5, B spec §4.2-§4.3).

`redact` is the single redaction routine. `Telemetry` applies it to every event's `fields`;
`configure_logging` installs it as a structlog processor for ad-hoc `structlog` loggers.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog
from pydantic import SecretStr

REDACTED_NUMBER = "[REDACTED_NUMBER]"
SECRET_MASK = "**********"  # noqa: S105

# Keys are compared in normalised form: lowercase with "-", "_", "." and spaces removed, so
# `X-Api-Key`, `api_key`, `apiKey` and `APIKEY` are the same key.
DROP_KEYS = frozenset(
    {
        "to",
        "from",
        "called",
        "caller",
        "number",
        "phone",
        "e164",
        "authorization",
        "proxyauthorization",
        "cookie",
        "setcookie",
        "xapikeyid",
        "apikey",
        "xapikey",
        "authtoken",
        "password",
        "token",
        "wstoken",
        "reporttoken",
        "paneltoken",
        "secret",
        "t",
        "recordingurl",
        "ua",
        "useragent",
    }
)
# Last word of the key (snake, kebab or camelCase): `foo_secret`, `ipHashKey`, `X-Auth-Token`.
DROP_LAST_WORDS = frozenset({"key", "token", "secret", "password", "apikey"})
# Raw client addresses: only a truncated network is logged (IPv4 /24, IPv6 /48).
IP_KEYS = frozenset(
    {"clientip", "ip", "peer", "peerip", "remoteaddr", "xforwardedfor", "cfconnectingip", "xrealip"}
)
# Twilio form dicts: location fields (C§8.6.2), e.g. CallerCity, FromState, ToZip, CalledCountry.
_TWILIO_GEO_RE = re.compile(r"(City|State|Zip|Country)$")
# Contracts §10.5 pattern, widened to common separators: (987) 654-3210, 987.654.3210,
# +91 (98765) 43210.
NUMBER_RE = re.compile(r"\+?\(?\d[\d\s\-().]{9,}\d")
_WORD_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def _norm(key: str) -> str:
    return re.sub(r"[-_.\s]", "", key).lower()


def _last_word(key: str) -> str:
    words = _WORD_RE.findall(key)
    return words[-1].lower() if words else ""


def truncate_ip(value: Any) -> str:
    """203.0.113.9 -> 203.0.113.0/24; 2001:db8::1 -> 2001:db8::/48; lists keep each entry."""
    out = []
    for part in str(value).split(","):
        part = part.strip()
        try:
            addr = ipaddress.ip_address(part)
        except ValueError:
            out.append("[REDACTED_IP]")
            continue
        prefix = 24 if addr.version == 4 else 48
        out.append(str(ipaddress.ip_network(f"{addr}/{prefix}", strict=False)))
    return ",".join(out)


def token_sha8(token: str) -> str:
    """The only form in which a token may be logged."""
    return hashlib.sha256(token.encode()).hexdigest()[:8]


def _drop(key: str) -> bool:
    return (
        _norm(key) in DROP_KEYS
        or _last_word(key) in DROP_LAST_WORDS
        or bool(_TWILIO_GEO_RE.search(key))
    )


def redact_value(v: Any) -> Any:
    if isinstance(v, SecretStr):
        return SECRET_MASK
    if isinstance(v, str):
        return NUMBER_RE.sub(REDACTED_NUMBER, v)
    if isinstance(v, Mapping):
        return redact(v)
    if isinstance(v, list | tuple):
        return [redact_value(x) for x in v]
    return v


def redact(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive §10.5 redaction: drop sensitive keys, scrub phone numbers, mask secrets."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        key = str(k)
        if _drop(key):
            continue
        out[k] = truncate_ip(v) if _norm(key) in IP_KEYS and v is not None else redact_value(v)
    return out


_BASE_KEYS = frozenset({"event", "level", "timestamp", "ts"})


def redact_processor(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor form of `redact` (the event name and level are kept)."""
    kept = {k: v for k, v in event_dict.items() if k in _BASE_KEYS}
    rest = redact({k: v for k, v in event_dict.items() if k not in _BASE_KEYS})
    return {**kept, **rest}


def configure_logging(level: str = "INFO") -> None:
    """Ad-hoc structlog loggers: JSON to stdout, redacted, at `level` or above."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            redact_processor,
            structlog.processors.JSONRenderer(
                serializer=lambda o, **kw: json.dumps(o, ensure_ascii=False, default=str)
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
