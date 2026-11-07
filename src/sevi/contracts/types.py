"""Annotated scalar types and the base contract model (contracts §4.1)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

UUID7_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

SessionId = Annotated[str, Field(pattern=UUID7_RE)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
SemVer = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
Digit = Annotated[int, Field(ge=0, le=9)]
SnrDb = Annotated[float, Field(ge=-20.0, le=10.0, allow_inf_nan=False)]
NonNegMs = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
UInt64 = Annotated[int, Field(ge=0, le=2**64 - 1)]
ParticipantId = Annotated[str, Field(pattern=r"^P-\d{2}$")]
Pseudonym = Annotated[str, Field(pattern=r"^S-\d{3}$")]
StudyFormId = Annotated[str, Field(pattern=r"^SF-\d{3}$")]
CallSid = Annotated[str, Field(pattern=r"^CA[0-9a-f]{32}$")]


def _utc(d: datetime) -> datetime:
    if d.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be UTC")
    return d


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_utc)]  # R-07


def _distinct3(v: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(set(v)) != 3:
        raise ValueError("triplet digits must be distinct")
    return v


Triplet = Annotated[tuple[Digit, Digit, Digit], AfterValidator(_distinct3)]


class SeviModel(BaseModel):
    """Base of every contract model."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        ser_json_timedelta="float",
        use_enum_values=False,
        validate_default=True,
    )


# Spec name (contracts §4.1 calls the base KJModel); kept so spec-literal imports work.
KJModel = SeviModel
