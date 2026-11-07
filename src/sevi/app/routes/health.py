"""`/healthz`, `/readyz`, `/version` (public listener; contracts §9.1)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from sevi.app.routes import state_of
from sevi.contracts.wire import VersionResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    r = await state_of(request).readiness.public()
    return JSONResponse(r.model_dump(mode="json"), status_code=503 if r.status == "down" else 200)


@router.get("/version", response_model=VersionResponse)
async def version(request: Request) -> VersionResponse:
    return state_of(request).readiness.version()
