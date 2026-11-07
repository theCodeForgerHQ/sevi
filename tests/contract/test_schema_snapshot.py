"""C-CON-01: the JSON schema of every contract model matches the committed snapshot.

Regenerate after an intentional contract change with `make schema-snapshot`
(then bump CONTRACTS_VERSION and label the PR `contract-change`).
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

import sevi.contracts
from sevi.contracts import ALL_MODELS, CONTRACTS_VERSION, PENDING_MODELS
from sevi.contracts.adapters import SttEvent, TransportEvent
from sevi.contracts.wire import ServerUiMessage

SNAPSHOT = Path(__file__).parent / "snapshots" / "schema.json"
UPDATE_ENV = "SEVI_UPDATE_SCHEMA_SNAPSHOT"


UNIONS: dict[str, Any] = {
    "sevi.contracts.wire.ServerUiMessage": ServerUiMessage,
    "sevi.contracts.adapters.SttEvent": SttEvent,
    "sevi.contracts.adapters.TransportEvent": TransportEvent,
}


def _enums() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for info in pkgutil.iter_modules(sevi.contracts.__path__):
        module = importlib.import_module(f"sevi.contracts.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, StrEnum) and obj.__module__ == module.__name__:
                out[f"{obj.__module__}.{obj.__qualname__}"] = {m.name: m.value for m in obj}
    return out


def current_schema() -> dict[str, Any]:
    return {
        "contracts_version": CONTRACTS_VERSION,
        "models": {
            f"{m.__module__}.{m.__qualname__}": m.model_json_schema(mode="validation")
            for m in ALL_MODELS
        },
        "unions": {
            name: TypeAdapter(tp).json_schema(mode="validation") for name, tp in UNIONS.items()
        },
        "enums": _enums(),
        "pending_models": list(PENDING_MODELS),
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
