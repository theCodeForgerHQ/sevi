"""``GET /audio/{t}/{name}.wav`` (contracts §8.6.3, B spec §8.3).

``t`` is the call's unguessable 128-bit token (compared by sha256, constant time); audio is
served only while the call runs and for ``SEVI_TWILIO__AUDIO_TTL_S`` after it ends. Anything
else, including a disabled phone, is 404. ``Cache-Control: no-store``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from sevi.app.routes.twilio import error_response, phone_of
from sevi.contracts.errors import ErrorCode, SeviError

router = APIRouter(include_in_schema=False)


@router.get("/audio/{t}/{name}.wav")
async def audio(t: str, name: str, request: Request) -> Response:
    svc = phone_of(request)
    body = svc.audio(t, name) if svc is not None and svc.enabled else None
    if body is None:
        return error_response(SeviError(ErrorCode.NOT_FOUND, "not found"))
    return Response(body, media_type="audio/wav", headers={"Cache-Control": "no-store"})
