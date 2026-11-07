"""Public JSON API: status, session create, session poll (contracts §9.1; B spec §9.2, §9.4)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from sevi.app.admission import ClientContext, client_ip, ip_hmac, peer_trusted
from sevi.app.routes import bearer, state_of
from sevi.app.state import AppState
from sevi.contracts.wire import (
    CreateSessionRequest,
    PublicStatusResponse,
    SessionStatusResponse,
)

router = APIRouter(prefix="/api/v1")


def client_context(
    app: AppState, request_peer: str | None, headers: dict[str, str], host: str | None
) -> ClientContext:
    s = app.settings
    trusted = peer_trusted(request_peer, s.trusted_proxies)
    ip = client_ip(request_peer, headers, s.trusted_proxies, s.client_ip_header)
    key = s.admission.ip_hash_key
    if ip is None:
        app.admission.warn_no_client_ip(s.client_ip_header)
        bucket = None
    else:
        secret = key.get_secret_value().encode() if key is not None else b"sevi-dev-ip-hash-key"
        bucket = ip_hmac(secret, ip)
    return ClientContext(ip_hmac=bucket, ws_base_host=host, trusted_peer=trusted)


def context_of(request: Request) -> ClientContext:
    app = state_of(request)
    peer = request.client.host if request.client else None
    headers = {k.lower(): v for k, v in request.headers.items()}
    return client_context(app, peer, headers, request.headers.get("host"))


@router.get("/status", response_model=PublicStatusResponse)
async def status(request: Request) -> PublicStatusResponse:
    return await state_of(request).admission.public_status()


@router.post("/sessions", status_code=201)
async def create_session(req: CreateSessionRequest, request: Request) -> JSONResponse:
    resp = await state_of(request).admission.create(req, context_of(request))
    return JSONResponse(resp.model_dump(mode="json"), status_code=201)


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def session_status(session_id: str, request: Request) -> SessionStatusResponse:
    return await state_of(request).admission.status(session_id, bearer(request))
