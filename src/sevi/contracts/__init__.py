"""Shared contracts between Person A (measurement) and Person B (real-time).

`ALL_MODELS` is built by auto-discovery: every `pydantic.BaseModel` subclass defined in any
`sevi.contracts.*` module is registered, so adding a contracts module needs no edit here.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from pydantic import BaseModel

CONTRACTS_VERSION = "0.1.0"

_EXCLUDED = frozenset({"sevi.contracts.types.SeviModel"})  # abstract base, not a contract


def _discover_models() -> tuple[type[BaseModel], ...]:
    found: dict[str, type[BaseModel]] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj.__module__ == module.__name__:
                name = f"{obj.__module__}.{obj.__qualname__}"
                if name not in _EXCLUDED:
                    found[name] = obj
    return tuple(found[k] for k in sorted(found))


ALL_MODELS: tuple[type[BaseModel], ...] = _discover_models()

__all__ = ["ALL_MODELS", "CONTRACTS_VERSION"]
