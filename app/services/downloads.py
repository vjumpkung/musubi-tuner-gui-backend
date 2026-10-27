"""Download job execution service."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import HTTPException

from ..config import Settings
from ..download_catalog import DOWNLOAD_MODELS
from ..schemas.downloads import DownloadModel


@dataclass
class DownloadJob:
    id: str
    script_id: str
    destination: Path
    status: str = "queued"
    progress: float | None = 0
    current_file: str | None = None
    message: str | None = "Waiting to start"
    error: str | None = None
    task: asyncio.Task[None] | None = None
    targets: tuple[Path, ...] = ()
    started: bool = False
    cancel_requested: bool = False

    def to_response(self) -> dict:
        payload: dict = {
            "id": self.id,
            "script_id": self.script_id,
            "status": self.status,
        }
        if self.progress is not None:
            payload["progress"] = round(min(100.0, max(0.0, self.progress)), 1)
        if self.current_file:
            payload["current_file"] = self.current_file
        if self.message:
            payload["message"] = self.message
        if self.error:
            payload["error"] = self.error
        return payload


class DownloadManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.transport: httpx.AsyncBaseTransport | None = None
        self.jobs: dict[str, DownloadJob] = {}
        self.reservations: dict[Path, str] = {}

    def get(self, job_id: str) -> DownloadJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown download job")
        return job

    def list_jobs(self) -> list[DownloadJob]:
        return list(reversed(self.jobs.values()))

    def start(
        self, script_id: str, destination_raw: str, hf_token: str | None = None
    ) -> DownloadJob:
        model = DOWNLOAD_MODELS.get(script_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Unknown download model")
        try:
            destination = self.settings.resolve_inside_workspace(destination_raw or ".")
            targets = tuple((destination / file.path).resolve() for file in model.files)
            if any(not target.is_relative_to(destination) for target in targets):
                raise ValueError
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Destination is outside the allowed root"
            )
        if any(target in self.reservations for target in targets):
            raise HTTPException(
                status_code=409,
                detail="A download target already has an active job",
            )

        destination.mkdir(parents=True, exist_ok=True)
        job = DownloadJob(
            id=str(uuid.uuid4()),
            script_id=script_id,
            destination=destination,
            targets=targets,
        )
        self.jobs[job.id] = job
        for target in targets:
            self.reservations[target] = job.id
        token = hf_token or os.environ.get("HF_TOKEN")
        job.task = asyncio.create_task(self._run(job, model, token))
        return job

    def _release(self, job: DownloadJob) -> None:
        for target in job.targets:
            if self.reservations.get(target) == job.id:
                del self.reservations[target]

    async def _run(
        self, job: DownloadJob, model: DownloadModel, token: str | None
    ) -> None:
        job.started = True
        try:
            job.status = "running"
            async with httpx.AsyncClient(
                transport=self.transport,
                follow_redirects=True,
                timeout=httpx.Timeout(60, connect=30),
                headers={"Accept-Encoding": "identity"},
            ) as client:
                count = len(model.files)
                for index, file in enumerate(model.files):
                    job.current_file = file.path
                    job.message = f"Downloading {file.path} ({index + 1}/{count})"
                    target = job.destination / file.path
                    temporary = target.parent / f".{target.name}.{job.id}.part"
                    created = False
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.parent.resolve().is_relative_to(job.destination):
                            job.error = (
                                f"Download path is outside destination: {file.path}"
                            )
                            return
                        headers = {"Authorization": f"Bearer {token}"} if token else {}
                        async with client.stream(
                            "GET", file.url, headers=headers
                        ) as response:
                            if response.status_code != 200:
                                if response.status_code in {401, 403}:
                                    job.error = (
                                        f"Access denied for {file.path}: accept the model "
                                        "license on Hugging Face and provide a token"
                                    )
                                elif response.status_code == 404:
                                    job.error = (
                                        f"File not found on Hugging Face: {file.path}"
                                    )
                                else:
                                    job.error = (
                                        f"Download failed for {file.path} "
                                        f"(HTTP {response.status_code})"
                                    )
                                return
                            if (
                                response.headers.get(
                                    "Content-Encoding", "identity"
                                ).lower()
                                != "identity"
                            ):
                                job.error = (
                                    f"Unexpected content encoding for {file.path}"
                                )
                                return
                            length = response.headers.get("Content-Length")
                            expected = int(length) if length is not None else None
                            written = 0
                            with temporary.open("xb") as handle:
                                created = True
                                async for chunk in response.aiter_bytes(1024 * 1024):
                                    handle.write(chunk)
                                    written += len(chunk)
                                    fraction = (
                                        min(written / expected, 1) if expected else 0
                                    )
                                    job.progress = min(
                                        (index + fraction) / count * 100, 99.9
                                    )
                            if expected is not None and written != expected:
                                job.error = f"Incomplete download for {file.path}"
                                return
                            if any(
                                not path.resolve().is_relative_to(job.destination)
                                for path in (temporary, target)
                            ):
                                job.error = (
                                    f"Download path is outside destination: {file.path}"
                                )
                                return
                            os.replace(temporary, target)
                            job.progress = min((index + 1) / count * 100, 99.9)
                    finally:
                        if created:
                            temporary.unlink(missing_ok=True)
            job.status = "completed"
            job.progress = 100
            job.message = "Download completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.message = "Cancelled"
            raise
        except httpx.HTTPError:
            job.error = f"Download failed for {job.current_file}: network error"
        except (OSError, ValueError):
            job.error = f"Download failed for {job.current_file}"
        finally:
            if job.error and job.status != "cancelled":
                job.status = "failed"
                job.message = None
            self._release(job)

    async def cancel(self, job_id: str) -> DownloadJob:
        job = self.get(job_id)
        if job.status in {"completed", "failed"}:
            raise HTTPException(
                status_code=409, detail=f"A {job.status} job can no longer be cancelled"
            )
        job.status = "cancelled"
        job.message = "Cancelled"
        if not job.cancel_requested:
            job.cancel_requested = True
            if job.task is not None and not job.task.done():
                job.task.cancel()
        if job.task is not None:
            while not job.task.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(job.task)
            if not job.started:
                self._release(job)
        return job

    async def shutdown(self) -> None:
        await asyncio.gather(
            *(
                self.cancel(job.id)
                for job in self.jobs.values()
                if job.task is not None and not job.task.done()
            )
        )
