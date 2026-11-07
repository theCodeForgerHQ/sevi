"""Shared helpers for B's store/retention tests (JSON-level session records)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any


def sid() -> str:
    """A valid-looking UUIDv7 string."""
    h = uuid.uuid4().hex
    return f"{h[:8]}-{h[8:12]}-7{h[13:16]}-8{h[17:20]}-{h[20:32]}"


def session_json(
    created_at: datetime,
    *,
    state: str = "ENDED",
    retention: str = "DELETE_72H",
    participant_type: str = "PUBLIC",
    delete_after: datetime | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    study = retention == "STUDY_KEEP"
    data: dict[str, Any] = {
        "session_id": session_id or sid(),
        "state": state,
        "retention": retention,
        "participant_type": participant_type,
        "created_at": created_at.isoformat(),
        "started_at": created_at.isoformat(),
        "delete_after": None
        if study
        else (delete_after or created_at + timedelta(hours=72)).isoformat(),
        "participant_id": "P-01" if study else None,
        "pseudonym": "S-001" if study else None,
        "age_band": "A18_29" if study else None,
        "twilio_call_sid": None,
        "client": {"browser_family": "Chrome"},
        "report_token_sha256": uuid.uuid4().hex * 2,
        "decision": {"band": "PASS"},
        "srt": {"srt_db": -9.5},
    }
    data.update(extra)
    return data


def trial_json(presentation_id: int, transcript: str = "one two three") -> str:
    import json

    return json.dumps(
        {
            "presentation_id": presentation_id,
            "digits": [1, 2, 3],
            "transcript": transcript,
            "normalised_tokens": ["1", "2", "3"],
            "noise_seed": 2**64 - 1,
        }
    )
