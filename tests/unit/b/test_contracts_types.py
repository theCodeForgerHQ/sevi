"""Validators in sevi.contracts.types and sevi.contracts.enums (contracts §4.1, §4.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from sevi.contracts import CONTRACTS_VERSION
from sevi.contracts.enums import AgeBand, Band, Lang, TruthRule
from sevi.contracts.types import (
    CallSid,
    SemVer,
    SessionId,
    SeviModel,
    SnrDb,
    Triplet,
    UtcDatetime,
)

UUID7 = "01890a5d-ac96-774b-bcce-b302099a8057"


def test_contracts_version_is_semver() -> None:
    assert TypeAdapter(SemVer).validate_python(CONTRACTS_VERSION) == "0.2.1"


def test_session_id_accepts_uuid7_only() -> None:
    ta = TypeAdapter(SessionId)
    assert ta.validate_python(UUID7) == UUID7
    for bad in (UUID7.upper(), "01890a5d-ac96-474b-bcce-b302099a8057", "nope"):
        with pytest.raises(ValidationError):
            ta.validate_python(bad)


def test_utc_datetime_rejects_non_utc_and_naive() -> None:
    ta = TypeAdapter(UtcDatetime)
    now = datetime(2026, 11, 7, 12, tzinfo=UTC)
    assert ta.validate_python(now) == now
    with pytest.raises(ValidationError):
        ta.validate_python(now.astimezone(timezone(timedelta(hours=5, minutes=30))))
    with pytest.raises(ValidationError):
        ta.validate_python(datetime(2026, 11, 7, 12))


def test_triplet_digits_distinct_and_in_range() -> None:
    ta: TypeAdapter[tuple[int, int, int]] = TypeAdapter(Triplet)
    assert ta.validate_python((1, 2, 3)) == (1, 2, 3)
    for bad in ((1, 1, 2), (1, 2, 10), (-1, 2, 3)):
        with pytest.raises(ValidationError):
            ta.validate_python(bad)


def test_snr_db_bounds_and_finite() -> None:
    ta = TypeAdapter(SnrDb)
    assert ta.validate_python(-20.0) == -20.0
    for bad in (-20.1, 10.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            ta.validate_python(bad)


def test_call_sid_pattern() -> None:
    ta = TypeAdapter(CallSid)
    ta.validate_python("CA" + "0" * 32)
    with pytest.raises(ValidationError):
        ta.validate_python("CA" + "G" * 32)


def test_base_model_is_frozen_and_forbids_extra() -> None:
    class M(SeviModel):
        x: int

    m = M(x=1)
    with pytest.raises(ValidationError):
        m.x = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        M(x=1, y=2)  # type: ignore[call-arg]


def test_enum_spec_values() -> None:
    assert TruthRule.T10B.value == "T10b"
    assert AgeBand.A66_PLUS.value == "66+"
    assert {lang.value for lang in Lang} == {"ta", "en"}
    assert Band("NO_BAND") is Band.NO_BAND
