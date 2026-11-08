"""App factory, startup/shutdown and the two listeners (B spec §2.1-§2.3, §9.1; contracts §5.1.1).

Startup order (any failure -> `StartupError`, `serve()` exits 78):
1. settings (loaded by the caller) and the one-worker assertion;
2. logging + telemetry (writer task);
3. store open + migrations + `PRAGMA quick_check`;
4. banks: `load_bank(lang, BANK_DIR, FROZEN_BANK_HASHES, ALLOW_UNFROZEN, prod=(env == "prod"))`
   per enabled language; a `BankError` makes that language unavailable, none loaded -> exit 78;
   the Prisma sample rate must equal the bank's chain rate (`BANK_CONFIG_MISMATCH`);
5. scorer, classifier, copy (stub_core fallback only in `SEVI_ENV=demo`; never in prod);
6. ledger, breakers, key semaphore, adapters (fakes when `SEVI_DEMO_MODE=true`);
7. the session port: `runtime.OrchestratorPort` over the real orchestrator (tests may inject
   another `PortFactory`, e.g. `StubSessionPort`);
8. the startup retention pass; `/readyz` is `down` until it finishes;
9. background tasks: retention loop, admission sweeper (1 s), readiness refresh (30 s).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import re
import signal
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sevi.adapters.breaker import BreakerRegistry
from sevi.adapters.key_semaphore import ProcessKeySemaphore
from sevi.adapters.ledger import Ledger, TwilioLimits
from sevi.app.admission import Admission, LedgerCreditGate
from sevi.app.clock import SystemClock
from sevi.app.logging import configure_logging
from sevi.app.operator import REALM, OperatorAuth
from sevi.app.readiness import Readiness
from sevi.app.retention import retention_loop, run_retention_once
from sevi.app.session_port import SessionPort, StubSessionPort
from sevi.app.settings import EXIT_CONFIG_INVALID, Settings, Switches, load_settings
from sevi.app.state import AppState
from sevi.app.store import SqliteStore
from sevi.app.telemetry import Telemetry
from sevi.contracts.adapters import Clock
from sevi.contracts.enums import KeyId, Lang
from sevi.contracts.errors import HTTP_STATUS, BankError, ErrorBody, ErrorCode, SeviError
from sevi.contracts.ports import AnswerScorer, ResultClassifier, StimulusProvider
from sevi.contracts.telemetry import EventName
from sevi.core.wording import WordingTable

MAX_BODY: Final = 16 * 1024
SWEEP_EVERY_S: Final = 1.0
READY_EVERY_S: Final = 30.0
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
WEB_STATIC: Final = REPO_ROOT / "web" / "static"
STATIC_MOUNT: Final = "/sevi"
_RID_RE: Final = re.compile(r"^[A-Za-z0-9-]{1,64}$")

PortFactory = Callable[[AppState], SessionPort]


class StartupError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------- errors -> envelope ----------


def error_response(err: SeviError, request_id: str | None) -> JSONResponse:
    details = dict(err.details)
    retry = details.pop("retry_after_s", None)
    retry_s = int(retry) if isinstance(retry, int | float) else None
    body = ErrorBody(
        code=err.code,
        message=(err.message or err.code.value)[:300],
        retryable=err.retryable,
        retry_after_s=retry_s,
        request_id=request_id,
        details={k: v for k, v in details.items() if isinstance(v, str | int | float | bool)},
    )
    headers = {"Retry-After": str(retry_s)} if retry_s is not None else None
    return JSONResponse(
        {"error": body.model_dump(mode="json")},
        status_code=HTTP_STATUS.get(err.code, 500),
        headers=headers,
    )


def _rid(scope: Scope) -> str | None:
    v = scope.get("sevi_request_id")
    return v if isinstance(v, str) else None


def install_handlers(app: FastAPI, *, operator: bool = False) -> None:
    @app.exception_handler(SeviError)
    async def _sevi(request: Request, exc: SeviError) -> JSONResponse:
        resp = error_response(exc, _rid(request.scope))
        if operator and exc.code is ErrorCode.UNAUTHORIZED:
            resp.headers["WWW-Authenticate"] = REALM
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        err = SeviError(ErrorCode.VALIDATION_FAILED, "request validation failed")
        return error_response(err, _rid(request.scope))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.BAD_REQUEST
        resp = error_response(SeviError(code, str(exc.detail)[:300]), _rid(request.scope))
        resp.status_code = exc.status_code
        return resp


# ---------- ASGI middleware ----------


class _TooLarge(Exception):
    pass


async def _send_envelope(send: Send, status: int, code: ErrorCode, rid: str | None) -> None:
    resp = error_response(SeviError(code, "request body too large" if status == 413 else ""), rid)
    resp.status_code = status
    await send({"type": "http.response.start", "status": status, "headers": resp.raw_headers})
    await send({"type": "http.response.body", "body": resp.body})


class GuardMiddleware:
    """Inside CORS: body limit (Content-Length and a streaming counter -> 413 `BAD_REQUEST`) and
    any unhandled exception -> 500 `INTERNAL`, so the page can always read the envelope."""

    def __init__(self, app: ASGIApp, telemetry: Any = None, max_body: int = MAX_BODY) -> None:
        self.app = app
        self.max_body = max_body
        self.telemetry = telemetry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = _rid(scope)
        cl = dict(scope.get("headers") or []).get(b"content-length")
        if cl is not None and (not cl.isdigit() or int(cl) > self.max_body):
            await _send_envelope(send, 413, ErrorCode.BAD_REQUEST, rid)
            return
        started = False
        too_large = False
        seen = 0

        async def recv() -> Message:
            nonlocal seen, too_large
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > self.max_body:
                    too_large = True
                    raise _TooLarge
            return msg

        async def snd(msg: Message) -> None:
            nonlocal started
            if too_large:
                if msg["type"] == "http.response.start" and not started:
                    started = True
                    await _send_envelope(send, 413, ErrorCode.BAD_REQUEST, rid)
                return
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, recv, snd)
        except Exception as exc:
            if started:
                raise
            if not too_large and self.telemetry is not None:
                self.telemetry.emit(
                    EventName.ERROR, "ERROR", error_code="INTERNAL", exc=type(exc).__name__
                )
            status = 413 if too_large else 500
            code = ErrorCode.BAD_REQUEST if too_large else ErrorCode.INTERNAL
            await _send_envelope(send, status, code, rid)


class HeadersMiddleware:
    """Outermost: request id, `Referrer-Policy: no-referrer`, `nosniff`, `/api/*` no-store."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        raw = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = raw if _RID_RE.match(raw) else uuid.uuid4().hex
        scope["sevi_request_id"] = rid
        api = str(scope.get("path", "")).startswith(("/api/", "/op/api/"))

        async def snd(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                headers = [
                    (k, v)
                    for k, v in msg.get("headers", [])
                    if k.lower() not in (b"referrer-policy", b"x-content-type-options")
                ]
                headers += [
                    (b"x-request-id", rid.encode()),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-content-type-options", b"nosniff"),
                ]
                if api:
                    headers = [(k, v) for k, v in headers if k.lower() != b"cache-control"]
                    headers.append((b"cache-control", b"no-store"))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, snd)


# ---------- startup ----------


@functools.cache
def git_sha() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out if re.fullmatch(r"[0-9a-f]{40}", out) else "0" * 40


def _assert_one_worker() -> None:
    if os.environ.get("WEB_CONCURRENCY") not in (None, "1"):
        raise StartupError(ErrorCode.CONFIG_INVALID, "exactly one worker is supported")


def load_banks(s: Settings, telemetry: Any) -> dict[Lang, StimulusProvider]:
    from sevi.core.bank import load_bank

    banks: dict[Lang, StimulusProvider] = {}
    for lang in s.result.enabled_langs:
        try:
            p = load_bank(
                lang,
                str(s.result.bank_dir),
                dict(s.result.frozen_bank_hashes),
                s.result.allow_unfrozen_bank,
                prod=(s.env == "prod"),
            )
        except BankError as e:
            telemetry.emit(
                EventName.ERROR, "ERROR", error_code=e.code.value, lang=lang.value, msg=str(e)
            )
            continue
        if p.manifest.chain.prisma_sample_rate_hz != s.prisma.sample_rate_hz:
            raise StartupError(
                ErrorCode.BANK_CONFIG_MISMATCH,
                f"prisma.sample_rate_hz != bank chain rate ({lang.value})",
            )
        banks[lang] = p
    if not banks:
        raise StartupError(ErrorCode.BANK_HASH_REJECTED, "no stimulus bank could be loaded")
    return banks


@functools.cache
def _copy_table() -> WordingTable:
    """The copy table is immutable process data: parse the YAML once per process."""
    from sevi.core.wording import load_copy

    return load_copy()


def _factories(
    s: Settings, telemetry: Any, stubs: list[str]
) -> tuple[AnswerScorer, ResultClassifier]:
    """Step 5 with the demo-only stub_core fallback (contracts §3.1, IR2 M-01). The classifier
    is `build_classifier()`'s exact construction, sharing the one parsed copy table."""

    def stub(name: str, err: Exception) -> None:
        if s.env != "demo":
            raise StartupError(ErrorCode.CONFIG_INVALID, f"{name} unavailable: {err}") from err
        stubs.append(name)
        telemetry.emit(EventName.STARTUP_STUB_FACTORY, "WARNING", factory=name)

    scorer: AnswerScorer
    classifier: ResultClassifier
    try:
        from sevi.core.scoring import load_lexicons

        scorer = load_lexicons()
    except (ImportError, NotImplementedError) as e:
        stub("load_lexicons", e)
        from sevi.fakes.stub_core import StubScorer

        scorer = StubScorer()
    try:
        from sevi.core.result import TruthTableClassifier

        classifier = TruthTableClassifier(_copy_table())
    except (ImportError, NotImplementedError) as e:
        stub("build_classifier", e)
        from sevi.fakes.stub_core import StubClassifier

        classifier = StubClassifier()
    return scorer, classifier


async def _adapters(state: AppState) -> None:
    """Build the Gnani/Evon clients (no network at construction). Demo mode: the same clients
    against in-process fakes. Every billed call still goes through the ledger."""
    from sevi.adapters.evon import EvonClient, EvonOptions
    from sevi.adapters.prisma_stream import PrismaOptions, PrismaStreamClient

    s = state.settings
    opts = PrismaOptions.from_settings(s)
    evon_opts = EvonOptions.from_settings(s.evon)
    keys: dict[KeyId, object]
    if s.demo_mode:
        from sevi.fakes.fake_evon import FakeEvon
        from sevi.fakes.fake_prisma import FakePrisma

        fake = FakePrisma(clock=state.clock)  # latency on the session clock
        opts = dataclasses.replace(opts, ws_url=await fake.start())
        state.closers.append(fake.stop)
        state.adapters["fake_prisma"] = fake  # the runtime installs the demo transcriber
        http = httpx.AsyncClient(transport=FakeEvon().transport())
        keys = {KeyId.A: "demo-key-a", KeyId.B: "demo-key-b"}
    else:
        http = httpx.AsyncClient()
        key_a = s.gnani.key_a
        keys = {KeyId.A: key_a or "", KeyId.B: s.gnani.key_b or key_a or ""}
    state.closers.append(http.aclose)
    state.adapters["stt"] = PrismaStreamClient(
        opts, keys, state.breakers, state.ledger, state.clock, state.telemetry
    )
    state.adapters["llm"] = EvonClient(
        evon_opts, http, state.breakers.get("evon", None), state.clock, state.telemetry
    )


async def startup(
    settings: Settings,
    *,
    clock: Clock | None = None,
    telemetry: Telemetry | None = None,
    port_factory: PortFactory | None = None,
    background: bool = True,
) -> AppState:
    _assert_one_worker()
    s = settings
    clock = clock or SystemClock()
    configure_logging(s.log_level)
    tel = telemetry or Telemetry(s.log_dir, clock=clock, level=s.log_level)
    tasks: list[asyncio.Task[Any]] = []
    if background:
        tasks.append(asyncio.create_task(tel.run(), name="telemetry"))

    store = await SqliteStore(s.sqlite_path, clock=clock, telemetry=tel).open()
    try:
        async with store.db.execute("PRAGMA quick_check") as cur:
            row = await cur.fetchone()
        if row is None or row[0] != "ok":
            raise StartupError(ErrorCode.STORE_FAILED, "PRAGMA quick_check failed")

        banks = await asyncio.to_thread(load_banks, s, tel)
        stubs: list[str] = []
        scorer, classifier = _factories(s, tel, stubs)
        from sevi.core.wording import DEFAULT_COPY_PATH

        wording = _copy_table()
        copy_sha = hashlib.sha256(Path(DEFAULT_COPY_PATH).read_bytes()).hexdigest()

        alias_b = s.key_b_aliased
        ledger = Ledger(
            store,
            s.ledger,
            alias_b=alias_b,
            clock=clock,
            telemetry=tel,
            twilio=TwilioLimits(
                trial_minutes_total=s.twilio.trial_minutes_total,
                trial_minutes_stop=s.twilio.trial_minutes_stop,
            ),
        )
        breakers = BreakerRegistry(clock, tel, alias_b=alias_b)
        sem = ProcessKeySemaphore(
            {KeyId.A: s.gnani.key_a_permits, KeyId.B: s.gnani.key_b_permits},
            alias_b=alias_b,
            stream_rest_coexist=s.gnani.stream_rest_coexist,
            clock=clock,
            telemetry=tel,
        )
    except BaseException:
        await store.close()
        for t in tasks:
            t.cancel()
        raise
    state = AppState(
        settings=s,
        switches=Switches.from_settings(s),
        clock=clock,
        telemetry=tel,
        store=store,
        ledger=ledger,
        credit=LedgerCreditGate(ledger, s.public_key, clock.mono_ms),
        breakers=breakers,
        sem=sem,
        port=StubSessionPort(clock, tel),
        banks=banks,
        scorer=scorer,
        classifier=classifier,
        wording=wording,
        git_sha=git_sha(),
        copy_sha256=copy_sha,
        stub_factories=stubs,
        tasks=tasks,
    )
    state.closers.append(store.close)
    state.admission = Admission(state)
    state.readiness = Readiness(state)
    try:
        await _adapters(state)
        if s.phone.enabled:
            from sevi.app.ivr import build_phone_service

            phone = build_phone_service(state)
            if phone is not None:
                state.adapters["phone"] = phone
                state.closers.append(phone.close)
        if port_factory is None:
            from sevi.app.runtime import orchestrator_port

            port_factory = orchestrator_port
        state.port = port_factory(state)
        if not s.client_ip_header:
            state.admission.warn_no_client_ip("")
        await run_retention_once(
            settings=s, store=store, telemetry=tel, now=clock.now_utc(), startup=True
        )
        state.startup_retention_done = True
        await state.readiness.refresh()
    except BaseException:
        await shutdown(state)
        raise
    if background:
        _start_background(state)
    return state


def _start_background(state: AppState) -> None:
    async def sweeper() -> None:
        while True:
            await asyncio.sleep(SWEEP_EVERY_S)
            with contextlib.suppress(Exception):
                await state.admission.sweep()

    async def ready() -> None:
        while True:
            await asyncio.sleep(READY_EVERY_S)
            with contextlib.suppress(Exception):
                await state.readiness.refresh()

    s = state.settings
    state.tasks += [
        asyncio.create_task(sweeper(), name="admission_sweeper"),
        asyncio.create_task(ready(), name="readiness_refresh"),
        asyncio.create_task(
            retention_loop(
                settings=s, store=state.store, telemetry=state.telemetry, clock=state.clock
            ),
            name="retention_loop",
        ),
    ]


async def shutdown(state: AppState) -> None:
    for t in state.tasks:
        t.cancel()
    for t in state.tasks:
        with contextlib.suppress(BaseException):
            await t
    state.tasks.clear()
    tel = state.telemetry
    if isinstance(tel, Telemetry):
        with contextlib.suppress(Exception):
            tel.drain()
            tel.close()
    for close in reversed(state.closers):
        with contextlib.suppress(Exception):
            await close()
    state.closers.clear()


# ---------- apps ----------


def _routes_public(app: FastAPI) -> None:
    from sevi.app.routes import health, public, report, web_ws

    for r in (health.router, public.router, report.router, web_ws.router):
        app.include_router(r)


def _routes_phone_public(app: FastAPI, settings: Settings) -> None:
    """Twilio trial IVR (#19): `/twilio/*` + `/audio/*`, only when SEVI_PHONE__ENABLED."""
    if settings.phone.enabled:
        from sevi.app.routes.twilio import install_phone

        install_phone(app, None)  # the service is read from AppState.adapters["phone"]


def _middleware_public(app: FastAPI, settings: Settings, telemetry: Any = None) -> None:
    app.add_middleware(GuardMiddleware, telemetry=telemetry)  # innermost of ours
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(settings.pages_origin).rstrip("/")],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["Retry-After", "X-Request-ID"],
        allow_credentials=False,
        max_age=600,
    )
    app.add_middleware(HeadersMiddleware)  # outermost


class _RevalidatingStatic(StaticFiles):
    """The page is ES modules imported by URL with no version suffix; without this a browser may
    keep running a cached old script after a deploy. `no-cache` still allows the ETag 304."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def _mount_static(app: FastAPI, settings: Settings) -> None:
    if settings.env != "prod" and WEB_STATIC.is_dir():
        app.mount(STATIC_MOUNT, _RevalidatingStatic(directory=WEB_STATIC, html=True), name="static")


def build_public_app(state: AppState) -> FastAPI:
    """§9.1 rows with L=P; `/op/*` is not mounted (404)."""
    app = FastAPI(title="Sevi", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.sevi = state
    install_handlers(app)
    _routes_public(app)
    _routes_phone_public(app, state.settings)
    _mount_static(app, state.settings)
    _middleware_public(app, state.settings, state.telemetry)
    return app


def build_operator_app(state: AppState) -> FastAPI:
    """Rows with L=O only."""
    from sevi.app.routes import operator

    s = state.settings
    app = FastAPI(title="Sevi operator", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.sevi = state
    app.state.operator_auth = OperatorAuth(
        s.operator, f"http://{s.op_bind_host}:{s.op_bind_port}", state.clock.mono_ms
    )
    install_handlers(app, operator=True)
    app.include_router(operator.router)
    if s.phone.enabled:
        from fastapi import Depends

        from sevi.app.routes.twilio import operator_router

        app.include_router(operator_router([Depends(operator.require_operator)]))
    app.add_middleware(GuardMiddleware, telemetry=state.telemetry)
    app.add_middleware(HeadersMiddleware)
    return app


def create_app(
    settings: Settings | None = None,
    *,
    clock: Clock | None = None,
    telemetry: Telemetry | None = None,
    port_factory: PortFactory | None = None,
) -> FastAPI:
    """Public app whose lifespan runs `startup()` (TestClient, `uvicorn --factory`)."""
    s = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = await startup(s, clock=clock, telemetry=telemetry, port_factory=port_factory)
        app.state.sevi = state
        try:
            yield
        finally:
            state.draining = True
            await shutdown(state)

    app = FastAPI(title="Sevi", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    install_handlers(app)
    _routes_public(app)
    _routes_phone_public(app, s)
    _mount_static(app, s)
    _middleware_public(app, s)
    return app


# ---------- serve (two uvicorn servers, one loop) ----------


def _uvicorn_server(app: FastAPI, host: str, port: int, drain_s: int) -> Any:
    import uvicorn

    class _NoSignalServer(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:  # we own SIGINT/SIGTERM
            yield

    cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        ws="wsproto",
        access_log=False,
        proxy_headers=False,
        ws_max_size=65536,
        timeout_graceful_shutdown=drain_s,
        lifespan="off",
        log_config=None,
    )
    return _NoSignalServer(cfg)


async def _drain(state: AppState) -> None:
    """§2.3: stop admitting, wait for sessions (<= DRAIN_S), then abort the rest."""
    state.draining = True
    deadline = state.clock.mono_ms() + state.settings.drain_s * 1000.0
    while state.clock.mono_ms() < deadline and (
        state.admission.active_keypad() + state.admission.active_voice()
    ):
        await asyncio.sleep(0.25)
    for h in list(state.port.all()):
        with contextlib.suppress(Exception):
            await state.port.abort(h.session_id, "drain")


async def serve(settings: Settings) -> int:
    try:
        state = await startup(settings)
    except StartupError as e:
        Telemetry(settings.log_dir).emit(
            EventName.ERROR, "ERROR", error_code=e.code.value, msg=str(e)
        )
        return EXIT_CONFIG_INVALID
    s = settings
    pub = _uvicorn_server(build_public_app(state), str(s.bind_host), s.bind_port, s.drain_s)
    op = _uvicorn_server(build_operator_app(state), str(s.op_bind_host), s.op_bind_port, s.drain_s)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    hits = 0

    def on_signal() -> None:
        nonlocal hits
        hits += 1
        stop.set()
        if hits > 1:  # a second signal skips the drain
            pub.should_exit = op.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, on_signal)
    servers = [asyncio.create_task(pub.serve()), asyncio.create_task(op.serve())]
    waiter = asyncio.create_task(stop.wait())
    await asyncio.wait([*servers, waiter], return_when=asyncio.FIRST_COMPLETED)
    await _drain(state)
    pub.should_exit = op.should_exit = True
    await asyncio.gather(*servers, return_exceptions=True)
    waiter.cancel()
    await shutdown(state)
    return 0


def main() -> int:
    try:
        settings = load_settings()
    except ValidationError as e:
        msg = {"level": "ERROR", "event": "error", "error_code": "CONFIG_INVALID"}
        print(json.dumps({**msg, "msg": str(e)[:2000]}))
        return EXIT_CONFIG_INVALID
    return asyncio.run(serve(settings))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
