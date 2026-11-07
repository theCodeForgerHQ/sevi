"""Shared contracts between Person A (measurement) and Person B (real-time).

`ALL_MODELS` is built by auto-discovery: every `pydantic.BaseModel` subclass defined in any
`sevi.contracts.*` module is registered, so adding a contracts module needs no edit here.

Models that reference another contracts module by forward reference (e.g. `SessionRecord` ->
Person A's `TrialRecord`) are rebuilt against the namespace of every contracts module. A model
that still cannot be completed (the referenced module has not landed yet) is listed in
`PENDING_MODELS` instead of `ALL_MODELS`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any

from pydantic import BaseModel

CONTRACTS_VERSION = "0.1.1"

_EXCLUDED = frozenset({"sevi.contracts.types.SeviModel"})  # abstract base, not a contract


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _modules() -> list[ModuleType]:
    return [
        importlib.import_module(f"{__name__}.{info.name}")
        for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name)
        if not info.name.startswith("_")
    ]


def _discover() -> tuple[tuple[type[BaseModel], ...], tuple[str, ...]]:
    modules = _modules()
    namespace: dict[str, Any] = {}
    found: dict[str, type[BaseModel]] = {}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            namespace[name] = obj
            if issubclass(obj, BaseModel) and _qualname(obj) not in _EXCLUDED:
                found[_qualname(obj)] = obj
    for model in found.values():
        if not model.__pydantic_complete__:
            model.model_rebuild(_types_namespace=namespace, raise_errors=False)
    complete = tuple(found[k] for k in sorted(found) if found[k].__pydantic_complete__)
    pending = tuple(k for k in sorted(found) if not found[k].__pydantic_complete__)
    return complete, pending


ALL_MODELS: tuple[type[BaseModel], ...]
PENDING_MODELS: tuple[str, ...]
ALL_MODELS, PENDING_MODELS = _discover()

__all__ = ["ALL_MODELS", "CONTRACTS_VERSION", "PENDING_MODELS"]
