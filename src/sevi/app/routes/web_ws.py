"""`WS /web/stream?sid=` (contracts §9.4; B spec §6.2, §9.1, S13).

Accept checks (`transport.web_ws.check_accept`): Origin == `SEVI_PAGES_ORIGIN`, `sid` names an
admitted AWAIT_START session, one socket per session (4409), pre-`start` sockets capped per
client network (or globally with no trusted client IP; 4429). Then the `WebTransport` is handed
to the session runtime through the `SessionPort`; the `start` token is checked by the runtime.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from sevi.app.routes import state_of
from sevi.app.routes.public import client_context
from sevi.transport.web_ws import AcceptView, WebTransport, check_accept, reject

router = APIRouter()


@router.websocket("/web/stream")
async def web_stream(ws: WebSocket) -> None:
    app = state_of(ws)
    peer = ws.client.host if ws.client else None
    headers = {k.lower(): v for k, v in ws.headers.items()}
    ctx = client_context(app, peer, headers, ws.headers.get("host"))
    sid = ws.query_params.get("sid")

    def lookup(s: str) -> AcceptView | None:
        h = app.port.get(s)
        return None if h is None else AcceptView(state=h.state, has_socket=h.connected)

    adm = app.admission
    rejection = check_accept(
        sid,
        ws.headers.get("origin"),
        str(app.settings.pages_origin),
        lookup,
        adm.pre_start_count(ctx.ip_hmac),
        adm.pre_start_cap(ctx.ip_hmac),
    )
    handle = app.port.get(sid) if sid else None
    if rejection is not None or handle is None or sid is None:
        if rejection is not None:
            await reject(ws, rejection, app.telemetry)
        return
    await ws.accept()
    adm.socket_opened(sid, ctx.ip_hmac)
    transport = WebTransport(
        ws, sid, app.clock, app.telemetry, send_timeout_s=app.settings.session.send_timeout_s
    )
    await handle.accept(transport)
    transport.start()
    await handle.run()
