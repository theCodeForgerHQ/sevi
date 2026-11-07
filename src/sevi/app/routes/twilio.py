"""Twilio trial webhooks and the operator dial (contracts §6.7, §8.6.2; B spec §8).

Mount with ``install_phone(app, service)`` on the public app (``/twilio/*``, ``/audio/*``) and
``operator_router(dependencies=[...])`` on the operator app (``/op/api/calls``), passing the
operator auth dependency. While the phone is disabled every ``/twilio/*`` request is 404 and
the dial is 409 ``PHONE_DISABLED``.

Every webhook is form-encoded, signature-checked against the URL rebuilt from the call's stored
``base_url`` (403 ``TWILIO_SIGNATURE_INVALID``), then token-checked (403
``TWILIO_TOKEN_INVALID``). Caller fields (``To``, ``From``, ``Caller``, ``*City`` ...) are never
logged: the form is handed to the service and only ``Digits``/``Recording*``/``Call*`` are read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

from sevi.app.ivr import SIGNATURE_HEADER, PhoneService
from sevi.contracts.errors import HTTP_STATUS, ErrorCode, ErrorEnvelope, SeviError
from sevi.contracts.wire import DialRequest, DialResponse

XML = "application/xml"
STATE_ATTR = "sevi_phone"

router = APIRouter(prefix="/twilio", include_in_schema=False)


def phone_of(request: Request) -> PhoneService | None:
    svc = getattr(request.app.state, STATE_ATTR, None)
    if svc is None:  # app.main: the service lives in AppState.adapters["phone"]
        st = getattr(request.app.state, "sevi", None)
        svc = getattr(st, "adapters", {}).get("phone")
    return svc if isinstance(svc, PhoneService) else None


def error_response(e: SeviError) -> JSONResponse:
    return JSONResponse(
        ErrorEnvelope(error=e.to_body()).model_dump(mode="json"),
        status_code=HTTP_STATUS.get(e.code, 500),
    )


def _not_found() -> Response:
    return error_response(SeviError(ErrorCode.NOT_FOUND, "not found"))


async def _form(request: Request) -> dict[str, str]:
    form = await request.form()
    return {k: v for k, v in form.items() if isinstance(v, str)}


def _path_qs(request: Request) -> str:
    q = request.url.query
    return request.url.path + (f"?{q}" if q else "")


async def _webhook(request: Request, sid: str, step: str | None) -> Response:
    svc = phone_of(request)
    if svc is None or not svc.enabled:
        return _not_found()
    form = await _form(request)
    t = request.query_params.get("t")
    try:
        rt = svc.authenticate(
            sid, t, _path_qs(request), form, request.headers.get(SIGNATURE_HEADER)
        )
        assert t is not None
        if step is None:
            await svc.status(rt, form)
            return Response(status_code=204)
        twiml = await svc.webhook(rt, step, form, t)
    except SeviError as e:
        return error_response(e)
    return Response(twiml, media_type=XML)


@router.post("/voice/{sid}")
async def voice(sid: str, request: Request) -> Response:
    """I03: Twilio fetches the call's first TwiML after the trial preamble."""
    return await _webhook(request, sid, "voice")


@router.post("/step/{sid}/{step}")
async def step(sid: str, step: str, request: Request) -> Response:
    """I04-I09, I14, I15."""
    return await _webhook(request, sid, step)


@router.post("/status/{sid}")
async def status(sid: str, request: Request) -> Response:
    """I10: status callback (204)."""
    return await _webhook(request, sid, None)


def operator_router(dependencies: Sequence[Any] = ()) -> APIRouter:
    """``POST /op/api/calls`` (I01) and ``POST /op/api/calls/{sid}/abort`` (I12/I13).

    Pass the operator auth dependency (``Depends(require_operator)``) from the app wiring."""
    r = APIRouter(prefix="/op/api", dependencies=list(dependencies))

    @r.post("/calls", response_model=DialResponse, status_code=201)
    async def dial(
        req: DialRequest,
        request: Request,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Response:
        svc = phone_of(request)
        try:
            if svc is None:
                raise SeviError(ErrorCode.PHONE_DISABLED, "phone IVR is not installed")
            resp = await svc.dial(req, idempotency_key=idempotency_key)
        except SeviError as e:
            return error_response(e)
        return JSONResponse(resp.model_dump(mode="json"), status_code=201)

    @r.post("/calls/{sid}/abort", status_code=204)
    async def abort(sid: str, request: Request) -> Response:
        svc = phone_of(request)
        if svc is None or not await svc.abort(sid):
            return _not_found()
        return Response(status_code=204)

    return r


def install_phone(app: FastAPI, service: PhoneService | None) -> None:
    """Startup hook for the public app: store the service and mount ``/twilio`` + ``/audio``.

    Safe with ``service=None`` or a disabled service (every route then answers 404)."""
    from sevi.app.routes import audio

    if service is not None:
        setattr(app.state, STATE_ATTR, service)
    app.include_router(router)
    app.include_router(audio.router)


__all__ = ["install_phone", "operator_router", "phone_of", "router"]
