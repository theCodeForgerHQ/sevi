"""Operator API routes (`/op/api/*`, operator listener only; contracts §9.1 `L=O`).

Every route depends on `require_operator` (CIDR, no forwarding headers, CSRF, Basic + scrypt).
Not built (scope): `/op/` console HTML, `/op/api/calls` (#19, phone), session timeline (SSE).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request

from sevi.app.operator import (
    OperatorService,
    ParticipantView,
    ReconcileResponse,
    SessionView,
    StationRequest,
    StationResponse,
)
from sevi.app.routes import state_of
from sevi.contracts.adapters import LedgerEvent
from sevi.contracts.wire import ReadyzResponse, SwitchRequest


async def require_operator(request: Request) -> None:
    auth = request.app.state.operator_auth
    peer = request.client.host if request.client else None
    auth.check(peer, {k.lower(): v for k, v in request.headers.items()}, request.method)


router = APIRouter(prefix="/op/api", dependencies=[Depends(require_operator)])


def svc(request: Request) -> OperatorService:
    return OperatorService(state_of(request))


@router.get("/participants", response_model=list[ParticipantView])
async def participants(request: Request) -> list[ParticipantView]:
    return await svc(request).participants()


@router.get("/sessions", response_model=list[SessionView])
async def sessions(request: Request, limit: int = Query(50, ge=1, le=500)) -> list[SessionView]:
    return svc(request).sessions(limit)


@router.post("/sessions/{session_id}/abort", response_model=SessionView)
async def abort(session_id: str, request: Request) -> SessionView:
    return await svc(request).abort(session_id)


@router.post("/switches", response_model=ReadyzResponse)
async def switches(req: SwitchRequest, request: Request) -> ReadyzResponse:
    return await svc(request).set_switch(req)


@router.post("/stations/sessions", response_model=StationResponse, status_code=201)
async def stations(
    req: StationRequest,
    request: Request,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> StationResponse:
    return await svc(request).create_station(req, idempotency_key)


@router.get("/readyz", response_model=ReadyzResponse)
async def readyz(request: Request) -> ReadyzResponse:
    return await state_of(request).readiness.full()


@router.get("/ledger")
async def ledger(request: Request) -> dict[str, Any]:
    return await svc(request).ledger_show()


@router.post("/ledger/reconcile", response_model=ReconcileResponse)
async def reconcile(ev: LedgerEvent, request: Request) -> ReconcileResponse:
    return await svc(request).reconcile(ev)
