"""Download job API routes."""

from fastapi import APIRouter, Request

from ..schemas.downloads import DownloadRequest
from ..services.downloads import DownloadManager

router = APIRouter(prefix="/api/downloads", tags=["downloads"])


def _manager(request: Request) -> DownloadManager:
    return request.app.state.downloads


@router.post("", status_code=202)
async def start_download(payload: DownloadRequest, request: Request) -> dict:
    hf_token = payload.hf_token.get_secret_value() if payload.hf_token else None
    return _manager(request).start(
        payload.script_id, payload.destination, hf_token
    ).to_response()


@router.get("/{job_id}")
async def read_download(job_id: str, request: Request) -> dict:
    return _manager(request).get(job_id).to_response()


@router.post("/{job_id}/cancel")
async def cancel_download(job_id: str, request: Request) -> dict:
    job = await _manager(request).cancel(job_id)
    return job.to_response()
