"""structlog configuration and PII redaction (contracts §10.5, B spec §4.2-§4.3).

`redact` is the single redaction routine. `Telemetry` applies it to every event's `fields`;
`configure_logging` installs it as a structlog processor for ad-hoc `structlog` loggers.
"""

from __future__ import annotations

import hashlib
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
        "x-api-key-id",
        "api_key",
        "auth_token",
        "password",
        "token",
        "ws_token",
        "report_token",
        "panel_token",
        "t",
        "recordingurl",
        "ua",
    }
)
_DROP_SUFFIX_RE = re.compile(r"(_key|_token|_secret|_password)$", re.IGNORECASE)
# Twilio form dicts: location fields (C§8.6.2), e.g. CallerCity, FromState, ToZip, CalledCountry.
_TWILIO_GEO_RE = re.compile(r"(City|State|Zip|Country)$")
NUMBER_RE = re.compile(r"\+?\d[\d\s\-]{9,}\d")


def token_sha8(token: str) -> str:
    """The only form in which a token may be logged."""
    return hashlib.sha256(token.encode()).hexdigest()[:8]


def _drop(key: str) -> bool:
    return (
        key.lower() in DROP_KEYS
        or bool(_DROP_SUFFIX_RE.search(key))
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
    return {k: redact_value(v) for k, v in fields.items() if not _drop(str(k))}


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
