"""Telemetry sinks for tests (B spec §12.1, CCR-v3-30; contracts §5.2 `Telemetry`)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sevi.contracts.telemetry import EventName, SpanName


class NullTelemetry:
    """Discards everything. Never raises."""

    def emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        return None

    @contextmanager
    def span(self, name: SpanName, **fields: Any) -> Iterator[None]:
        yield

    def mark_span(
        self, name: SpanName, start_mono_ms: float, end_mono_ms: float, **fields: Any
    ) -> None:
        return None


class RecordingTelemetry(NullTelemetry):
    """Keeps every event as `(name, level, fields)` so tests can assert on them."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def emit(self, event: EventName, level: str = "INFO", **fields: Any) -> None:
        self.events.append((str(event), level, dict(fields)))

    @contextmanager
    def span(self, name: SpanName, **fields: Any) -> Iterator[None]:
        self.events.append((f"span.start:{name}", "DEBUG", dict(fields)))
        try:
            yield
        finally:
            self.events.append((f"span.end:{name}", "DEBUG", dict(fields)))

    def mark_span(
        self, name: SpanName, start_mono_ms: float, end_mono_ms: float, **fields: Any
    ) -> None:
        self.events.append(
            (f"span:{name}", "DEBUG", {**fields, "dur_ms": end_mono_ms - start_mono_ms})
        )

    def names(self) -> list[str]:
        return [e[0] for e in self.events]

    def of(self, event: EventName) -> list[dict[str, Any]]:
        return [f for n, _, f in self.events if n == str(event)]
