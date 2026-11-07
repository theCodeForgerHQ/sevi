"""Operator password hashing (B spec §9.5, U-OP-02)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sevi.app.operator import hash_password, verify_password
from sevi.app.settings import Settings


def test_U_OP_02_scrypt_roundtrip_and_malformed_setting() -> None:
    stored = hash_password("pw-1")
    assert verify_password("pw-1", stored)
    assert not verify_password("pw-2", stored)
    assert not verify_password("pw-1", "plain")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, operator={"password_scrypt": "md5$abc"})  # type: ignore[call-arg]
