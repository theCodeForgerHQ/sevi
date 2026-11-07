"""C-CON-01: the JSON schema of every contract model matches the committed snapshot.

Regenerate after an intentional contract change with `make schema-snapshot`
(then bump CONTRACTS_VERSION and label the PR `contract-change`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sevi.contracts import ALL_MODELS, CONTRACTS_VERSION

SNAPSHOT = Path(__file__).parent / "snapshots" / "schema.json"
UPDATE_ENV = "SEVI_UPDATE_SCHEMA_SNAPSHOT"


def current_schema() -> dict[str, Any]:
    return {
        "contracts_version": CONTRACTS_VERSION,
        "models": {
            f"{m.__module__}.{m.__qualname__}": m.model_json_schema(mode="validation")
            for m in ALL_MODELS
        },
    }


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_C_CON_01_schema_snapshot() -> None:
    text = render(current_schema())
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(text, encoding="utf-8")
    assert SNAPSHOT.exists(), "schema snapshot missing: run `make schema-snapshot`"
    assert SNAPSHOT.read_text(encoding="utf-8") == text, (
        "contract schema drifted from tests/contract/snapshots/schema.json; if intended, run "
        "`make schema-snapshot`, bump CONTRACTS_VERSION and label the PR contract-change"
    )


def test_C_CON_01_registry_is_discovered_and_sorted() -> None:
    names = [f"{m.__module__}.{m.__qualname__}" for m in ALL_MODELS]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert all(n.startswith("sevi.contracts.") for n in names)
    assert "sevi.contracts.types.SeviModel" not in names
