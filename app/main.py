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
from starlette.responses import Response

from . import datasets, downloads, training
from .config import load_settings
from .db import Database
from .downloads import DownloadManager
from .queue_runner import QueueRunner, recover_interrupted_jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    db = await Database.open(settings.database_path)
    await recover_interrupted_jobs(db)

    runner = QueueRunner(db, settings)
    await runner.restore_state()
    runner_task = asyncio.create_task(runner.run())

    app.state.settings = settings
    app.state.db = db
    app.state.runner = runner
    app.state.downloads = DownloadManager(settings)

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
        app.mount("/", SpaStaticFiles(directory=settings.web_dir, html=True), name="web")

    return app


app = create_app()
