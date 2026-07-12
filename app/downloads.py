"""Download jobs: run allowlisted model download scripts as async jobs."""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .config import Settings
from .process import spawn, terminate_tree

# script_id -> fixed filename. Never accept a filename or shell command from the request.
SCRIPT_ALLOWLIST: dict[str, str] = {
    "flux2-dev": "download_flux2_dev.sh",
    "flux-kontext-dev": "download_flux_kontext_dev.sh",
    "framepack": "download_framepack.sh",
    "hidream-o1": "download_hidream_o1.sh",
    "hunyuan-video": "download_hunyuan_video.sh",
    "hunyuan-video-1-5": "download_hunyuan_video_1_5.sh",
    "ideogram-4": "download_ideogram4.sh",
    "kandinsky-5-t2v": "download_kandinsky5_t2v.sh",
    "krea-2": "download_krea2.sh",
    "qwen-image-bf16": "download_qwen_image_fp16.sh",
    "qwen-image-fp8": "download_qwen_image_fp8.sh",
    "qwen-image-edit": "download_qwen_image_edit_fp16.sh",
    "qwen-image-edit-2509": "download_qwen_image_edit_2509_fp16.sh",
    "wan-2-1-t2v-14b": "download_wan21_t2v_14B_fp16.sh",
    "wan-2-1-i2v-14b": "download_wan21_i2v_14B_fp16.sh",
    "wan-2-2-t2v-14b": "download_wan22_t2v_14B_fp16.sh",
    "wan-2-2-i2v-14b": "download_wan22_i2v_14B_fp16.sh",
    "z-image-base": "download_z_image_base_bf16.sh",
    "z-image-turbo": "download_z_image_turbo_bf16.sh",
}

ACTIVE_STATUSES = {"queued", "running"}
MODEL_DIR_PATTERN = re.compile(
    r"(?:diffusion_models|vae|text_encoders|image_encoder)/[\w./+-]+"
)
PERCENT_PATTERN = re.compile(r"(?<![\w.])(\d{1,3})%")


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
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None
    last_lines: deque[str] = field(default_factory=lambda: deque(maxlen=20))

    def to_response(self) -> dict:
        payload: dict = {"id": self.id, "script_id": self.script_id, "status": self.status}
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
        self.jobs: dict[str, DownloadJob] = {}

    def get(self, job_id: str) -> DownloadJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown download job")
        return job

    def start(self, script_id: str, destination_raw: str) -> DownloadJob:
        filename = SCRIPT_ALLOWLIST.get(script_id)
        if filename is None:
            raise HTTPException(status_code=404, detail="Unknown download script")
        script_path = self.settings.scripts_dir / filename
        if not script_path.is_file():
            raise HTTPException(status_code=503, detail=f"Download script is not installed: {filename}")
        # Resolve bash through PATH ourselves: on Windows CreateProcess would search
        # system32 first and hit the WSL stub instead of Git Bash.
        bash = shutil.which("bash")
        if bash is None:
            raise HTTPException(status_code=503, detail="bash is not available on the server")
        if shutil.which("hf") is None:
            raise HTTPException(status_code=503, detail="Hugging Face hf CLI is not available on the server")

        try:
            destination = self.settings.resolve_inside_workspace(destination_raw or ".")
        except ValueError:
            raise HTTPException(status_code=400, detail="Destination is outside the allowed root")

        for existing in self.jobs.values():
            if (
                existing.script_id == script_id
                and existing.destination == destination
                and existing.status in ACTIVE_STATUSES
            ):
                raise HTTPException(
                    status_code=409,
                    detail="This script already has an active job for that destination",
                )

        destination.mkdir(parents=True, exist_ok=True)
        job = DownloadJob(id=str(uuid.uuid4()), script_id=script_id, destination=destination)
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job, bash, script_path))
        return job

    async def _run(self, job: DownloadJob, bash: str, script_path: Path) -> None:
        if job.status == "cancelled":
            return
        try:
            job.process = await spawn([bash, str(script_path)], cwd=job.destination)
        except asyncio.CancelledError:
            if job.process is not None:
                await terminate_tree(job.process)
            raise
        except OSError as error:
            if job.status == "cancelled":
                return
            job.status = "failed"
            job.error = f"Could not start the download script: {error}"
            job.message = None
            return

        # Cancellation can arrive while create_subprocess_exec is awaiting the
        # child process. In that case, never resurrect the job as running.
        if job.status == "cancelled":
            await terminate_tree(job.process)
            return
        job.status = "running"
        job.message = "Download started"
        assert job.process.stdout is not None
        async for raw_line in job.process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if not line.strip():
                continue
            job.last_lines.append(line)
            self._apply_line(job, line)

        return_code = await job.process.wait()
        if job.status == "cancelled":
            return
        if return_code == 0:
            job.status = "completed"
            job.progress = 100
            job.message = "Download completed"
        else:
            job.status = "failed"
            job.message = None
            job.error = "\n".join(job.last_lines) or f"Script exited with code {return_code}"

    def _apply_line(self, job: DownloadJob, line: str) -> None:
        """Best-effort progress from the script's echo/hf output."""
        job.message = line[-300:]
        file_match = MODEL_DIR_PATTERN.search(line)
        if file_match:
            job.current_file = file_match.group(0)
        percent_match = None
        for percent_match in PERCENT_PATTERN.finditer(line):
            pass
        if percent_match:
            value = int(percent_match.group(1))
            if 0 <= value <= 100:
                job.progress = value

    async def cancel(self, job_id: str) -> DownloadJob:
        job = self.get(job_id)
        if job.status == "cancelled":
            return job
        if job.status in {"completed", "failed"}:
            raise HTTPException(status_code=409, detail=f"A {job.status} job can no longer be cancelled")
        job.status = "cancelled"
        job.message = "Cancelled"
        if job.process is not None:
            await terminate_tree(job.process)
        return job

    async def shutdown(self) -> None:
        for job in self.jobs.values():
            if job.status in ACTIVE_STATUSES and job.process is not None:
                await terminate_tree(job.process)
            if job.task is not None and not job.task.done():
                try:
                    # A cancelled job may still be returning from process
                    # creation and performing its own terminate_tree call.
                    await asyncio.wait_for(asyncio.shield(job.task), timeout=15)
                except TimeoutError:
                    if job.process is not None:
                        await terminate_tree(job.process)
                    job.task.cancel()
                    try:
                        await job.task
                    except asyncio.CancelledError:
                        pass
                except asyncio.CancelledError:
                    pass


class DownloadRequest(BaseModel):
    script_id: str
    destination: str = "."


router = APIRouter(prefix="/api/downloads", tags=["downloads"])


def _manager(request: Request) -> DownloadManager:
    return request.app.state.downloads


@router.post("", status_code=202)
async def start_download(payload: DownloadRequest, request: Request) -> dict:
    return _manager(request).start(payload.script_id, payload.destination).to_response()


@router.get("/{job_id}")
async def read_download(job_id: str, request: Request) -> dict:
    return _manager(request).get(job_id).to_response()


@router.post("/{job_id}/cancel")
async def cancel_download(job_id: str, request: Request) -> dict:
    job = await _manager(request).cancel(job_id)
    return job.to_response()
