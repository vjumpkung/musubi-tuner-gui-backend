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
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import datasets, downloads, training
from .config import load_settings
from .db import Database
from .downloads import DownloadManager
from .queue_runner import QueueRunner, recover_interrupted_jobs


class _ManagedBodyTooLarge(OSError):
    pass


class ManagedUploadBodyLimitMiddleware:
    """Reject managed multipart bodies before Starlette can spool them to disk."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and str(scope.get("path", "")).rstrip("/")
            in {"/api/datasets/managed", "/api/datasets/managed/batch"}
        ):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.managed_datasets_dir.mkdir(parents=True, exist_ok=True)

    db = await Database.open(settings.database_path)
    await datasets.reconcile_managed_storage(db, settings.managed_datasets_dir)
    await recover_interrupted_jobs(db)

    runner = QueueRunner(db, settings)
    await runner.restore_state()
    runner_task = asyncio.create_task(runner.run())

    app.state.settings = settings
    app.state.db = db
    app.state.runner = runner
    app.state.downloads = DownloadManager(settings)
    app.state.managed_storage_lock = asyncio.Lock()

    try:
        yield
    finally:
        runner_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass
        if runner.current_process is not None:
            from .process import terminate_tree

            await terminate_tree(runner.current_process)
        await app.state.downloads.shutdown()
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

    # Added first so the subsequently-added CORS middleware wraps direct 413
    # responses from the body limiter as well as normal FastAPI responses.
    app.add_middleware(
        ManagedUploadBodyLimitMiddleware,
        max_bytes=settings.managed_max_request_bytes,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(downloads.router)
    app.include_router(datasets.router)
    app.include_router(training.router)

    if (settings.web_dir / "index.html").is_file():
        app.mount(
            "/", SpaStaticFiles(directory=settings.web_dir, html=True), name="web"
        )

    return app


app = create_app()
