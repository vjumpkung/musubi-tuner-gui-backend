"""FastAPI application entry point.

Run with a single worker process — the queue runner lives in this process and
multiple workers would double-execute jobs:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

In production the built Vite frontend is served from the `web/` directory
(`pnpm build` in musubi-tuner-gui-frontend writes there); API routes always
take precedence over static files.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import load_settings
from .db import Database
from .routes import auth as auth_routes
from .routes import datasets, downloads, system_monitor, training
from .routes.mcp import build_mcp_app
from .services.auth import AuthManager, session_token_from_cookie_header
from .services.downloads import DownloadManager
from .services.queue_runner import QueueRunner, recover_interrupted_jobs
from .services.realtime import RealtimeHub

logger = logging.getLogger(__name__)

CSRF_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class _ManagedBodyTooLarge(OSError):
    pass


class ManagedUploadBodyLimitMiddleware:
    """Reject managed multipart bodies before Starlette can spool them to disk."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        method = scope.get("method")
        path = str(scope.get("path", "")).rstrip("/")
        is_managed_upload = (
            method == "POST"
            and path in {"/api/datasets/managed", "/api/datasets/managed/batch"}
        ) or (
            method == "POST"
            and path.startswith("/api/datasets/managed/upload-sessions/")
            and path.endswith("/files")
        ) or (
            method == "PUT"
            and path.startswith("/api/datasets/")
            and path.endswith("/managed")
        )
        if scope["type"] != "http" or not is_managed_upload:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _ManagedBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _ManagedBodyTooLarge:
            await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(
            status_code=413,
            content={
                "detail": (
                    "Managed dataset request body exceeds the "
                    f"{self.max_bytes}-byte limit"
                )
            },
        )(scope, receive, send)


class AuthMiddleware:
    """Require a session cookie or Bearer credential on protected paths.

    Protected: the API (except ``/api/auth``), the MCP endpoint, and the OpenAPI
    docs. The SPA and its assets stay public so the login page can load.
    """

    def __init__(self, app: ASGIApp, auth: AuthManager):
        self.app = app
        self.auth = auth

    @staticmethod
    def _is_protected(path: str) -> bool:
        if path == "/api/auth" or path.startswith("/api/auth/"):
            return False
        return (
            path.startswith("/api/")
            or path == "/mcp"
            or path.startswith("/mcp/")
            or path == "/openapi.json"
            or path.startswith("/docs")
            or path.startswith("/redoc")
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        if (
            not self.auth.enabled
            or scope["type"] not in {"http", "websocket"}
            or not self._is_protected(path)
        ):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        by_cookie = False
        outcome = "unauthenticated"

        authorization = headers.get("authorization", "")
        scheme, _, credential = authorization.partition(" ")
        credential = credential.strip()
        if scheme.lower() == "bearer" and credential:
            if self.auth.is_internal_token(credential):
                outcome = "ok"
            else:
                result = await self.auth.check_secret(credential)
                outcome = {"ok": "ok", "throttled": "throttled"}.get(
                    result, "unauthenticated"
                )
        else:
            token = session_token_from_cookie_header(headers.get("cookie"))
            if token is not None and self.auth.verify_token(token):
                outcome = "ok"
                by_cookie = True

        if (
            outcome == "ok"
            and by_cookie
            and scope["type"] == "http"
            and scope.get("method") in CSRF_METHODS
            and not path.startswith("/api/socket.io/")
            and not headers.get("x-requested-with")
        ):
            outcome = "csrf"

        if outcome == "ok":
            await self.app(scope, receive, send)
        else:
            await self._reject(scope, receive, send, outcome)

    async def _reject(
        self, scope: Scope, receive: Receive, send: Send, outcome: str
    ) -> None:
        if scope["type"] == "websocket":
            # ASGI: the connect message must be consumed before closing.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        if outcome == "throttled":
            response = JSONResponse(
                status_code=429,
                content={"detail": "Too many attempts, try again shortly"},
                headers={"Retry-After": "1"},
            )
        elif outcome == "csrf":
            response = JSONResponse(
                status_code=403,
                content={"detail": "Missing X-Requested-With header"},
            )
        else:
            response = JSONResponse(
                status_code=401, content={"detail": "Authentication required"}
            )
        await response(scope, receive, send)


class McpPathMiddleware:
    """Serve ``/mcp`` as ``/mcp/``.

    The MCP app is mounted at ``/mcp`` with its endpoint at ``/``. A bare ``/mcp``
    would otherwise be redirected by Starlette, or swallowed by the SPA catch-all
    (405), so rewrite it before routing.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.managed_datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.managed_uploads_dir.mkdir(parents=True, exist_ok=True)
    datasets.cleanup_staged_uploads(settings.managed_uploads_dir)

    db = await Database.open(settings.database_path)
    await app.state.auth.load_secret(db)
    if app.state.auth.enabled:
        logger.info("Password authentication enabled")
    else:
        logger.info("Password authentication disabled (MUSUBI_GUI_PASSWORD not set)")
    await datasets.reconcile_managed_storage(db, settings.managed_datasets_dir)
    await recover_interrupted_jobs(db)

    runner = QueueRunner(db, settings, events=app.state.realtime)
    await runner.restore_state()
    # Start the hub before the runner so no early state event is dropped.
    app.state.realtime.start()
    runner_task = asyncio.create_task(runner.run())

    app.state.settings = settings
    app.state.db = db
    app.state.runner = runner
    app.state.downloads = DownloadManager(settings, events=app.state.realtime)
    app.state.managed_storage_lock = asyncio.Lock()
    app.state.managed_upload_lock = asyncio.Lock()

    mcp_stack = AsyncExitStack()
    try:
        # The MCP session manager must be running before any /mcp request.
        await mcp_stack.enter_async_context(
            app.state.mcp_app.lifespan(app.state.mcp_app)
        )
        yield
    finally:
        runner_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass
        if runner.current_process is not None:
            from .utils.process import terminate_tree

            await terminate_tree(runner.current_process)
        await app.state.downloads.shutdown()
        await app.state.realtime.stop()
        await mcp_stack.aclose()
        await db.close()


class SpaStaticFiles(StaticFiles):
    """Serve the built frontend; unknown paths fall back to index.html."""

    async def get_response(self, path: str, scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="musubi-tuner-gui backend", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> Response:
        if request.url.path == "/api/downloads" or request.url.path.startswith(
            "/api/downloads/"
        ):
            return JSONResponse(
                status_code=422, content={"detail": "Invalid download request"}
            )
        return await request_validation_exception_handler(request, error)

    # Added first so the subsequently-added CORS middleware wraps direct 413
    # responses from the body limiter as well as normal FastAPI responses.
    app.add_middleware(
        ManagedUploadBodyLimitMiddleware,
        max_bytes=settings.managed_max_request_bytes,
    )

    auth = AuthManager(settings.password)
    app.state.auth = auth
    app.add_middleware(AuthMiddleware, auth=auth)

    app.add_middleware(McpPathMiddleware)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

    app.include_router(auth_routes.router)
    app.include_router(downloads.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(system_monitor.router)

    hub = RealtimeHub(settings)
    app.state.realtime = hub
    # Mounted after the routers and before the SPA catch-all. AuthMiddleware
    # already gates /api/socket.io/ because it is under /api/.
    app.mount("/api/socket.io", hub.asgi_app(), name="socketio")

    # Built after every router is included: the tools come from the OpenAPI
    # schema. Mounted before the SPA catch-all; its lifespan runs in `lifespan`.
    mcp_app = build_mcp_app(app, auth)
    app.state.mcp_app = mcp_app
    app.mount("/mcp", mcp_app, name="mcp")

    if (settings.web_dir / "index.html").is_file():
        app.mount(
            "/", SpaStaticFiles(directory=settings.web_dir, html=True), name="web"
        )

    return app


app = create_app()
