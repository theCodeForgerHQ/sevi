"""HTTP/WS routes (B spec §9; contracts §9.1). Twilio routes are #19."""

from __future__ import annotations

from typing import cast

from starlette.requests import HTTPConnection

from sevi.app.state import AppState


def state_of(conn: HTTPConnection) -> AppState:
    return cast(AppState, conn.app.state.sevi)


def bearer(conn: HTTPConnection) -> str | None:
    h = conn.headers.get("authorization")
    if not h or not h.lower().startswith("bearer "):
        return None
    tok = h[7:].strip()
    return tok or None
