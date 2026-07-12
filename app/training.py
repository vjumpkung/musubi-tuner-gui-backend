"""Training job queue API: enqueue jobs, control the queue, read status and logs."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel

from .command_builder import ExtraArgsError, ValuesError, validate_values
from .config import Settings
from .datasets import render_toml
from .db import Database, utc_now
from .profiles import TRAINING_PROFILES
from .queue_runner import QueueRunner, renumber_queue

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
JOB_STATUSES = {"queued", "running", "completed", "failed", "cancelled"}
STAGE_KEYS = ("cache_latents", "cache_text_encoder", "train")
LOG_CHUNK_BYTES = 64 * 1024

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingJobRequest(BaseModel):
    name: str
    profile_id: str
    dataset_config_id: str
    skip_cache_stages: bool = False
    values: dict[str, Any] = {}


class ReorderRequest(BaseModel):
    queue_position: int


def _db(request: Request) -> Database:
    return request.app.state.db


def _runner(request: Request) -> QueueRunner:
    return request.app.state.runner


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _job_response(row) -> dict:
    progress = json.loads(row["progress_json"]) if row["progress_json"] else {}
    return {
        "id": row["id"],
        "name": row["name"],
        "profile_id": row["profile_id"],
        "dataset_config_id": row["dataset_config_id"],
        "status": row["status"],
        "queue_position": row["queue_position"],
        "current_stage": row["current_stage"],
        "stages": json.loads(row["stages_json"]),
        "progress": {
            "epoch": progress.get("epoch"),
            "step": progress.get("step"),
            "total_steps": progress.get("total_steps"),
            "percent": progress.get("percent"),
        },
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


async def _get_job(db: Database, job_id: str):
    row = await db.fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown training job")
    return row


def _normalize_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        normalized[key] = value if isinstance(value, bool) else str(value)
    return normalized


async def _queue_object(db: Database, runner: QueueRunner) -> dict:
    queued_row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM training_jobs WHERE status = 'queued'"
    )
    running_row = await db.fetch_one(
        "SELECT id FROM training_jobs WHERE status = 'running' LIMIT 1"
    )
    return {
        "state": runner.state,
        "queued": queued_row["n"] if queued_row else 0,
        "running_job_id": running_row["id"] if running_row else None,
    }


@router.get("/queue")
async def read_queue(request: Request) -> dict:
    return await _queue_object(_db(request), _runner(request))


@router.post("/queue/start")
async def start_queue(request: Request) -> dict:
    runner = _runner(request)
    await runner.set_state("running")
    return await _queue_object(_db(request), runner)


@router.post("/queue/pause")
async def pause_queue(request: Request) -> dict:
    runner = _runner(request)
    await runner.set_state("paused")
    return await _queue_object(_db(request), runner)


async def _insert_job(
    db: Database,
    settings: Settings,
    runner: QueueRunner,
    *,
    name: str,
    profile_id: str,
    dataset_config_id: str | None,
    dataset_config_toml: str,
    values: dict[str, Any],
    skip_cache_stages: bool,
) -> dict:
    job_id = str(uuid.uuid4())
    stages = [
        {
            "key": key,
            "status": "skipped" if skip_cache_stages and key != "train" else "pending",
        }
        for key in STAGE_KEYS
    ]
    snapshot_path = settings.jobs_dir / job_id / "dataset_config.toml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(dataset_config_toml, encoding="utf-8")

    try:
        async with db.write_lock:
            try:
                if dataset_config_id is not None:
                    dataset_row = await db.fetch_one(
                        "SELECT id FROM dataset_configs WHERE id = ?",
                        (dataset_config_id,),
                    )
                    if dataset_row is None:
                        raise HTTPException(
                            status_code=404, detail="Unknown dataset config"
                        )
                position_row = await db.fetch_one(
                    "SELECT COUNT(*) AS n FROM training_jobs WHERE status = 'queued'"
                )
                await db.connection.execute(
                    "INSERT INTO training_jobs (id, name, profile_id, dataset_config_id, "
                    "dataset_config_toml, values_json, status, queue_position, stages_json, "
                    "created_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                    (
                        job_id,
                        name,
                        profile_id,
                        dataset_config_id,
                        dataset_config_toml,
                        json.dumps(values),
                        position_row["n"] if position_row else 0,
                        json.dumps(stages),
                        utc_now(),
                    ),
                )
                await db.connection.commit()
            except BaseException:
                await db.connection.rollback()
                raise
    except BaseException:
        shutil.rmtree(snapshot_path.parent, ignore_errors=True)
        raise
    runner.wake.set()
    return _job_response(await _get_job(db, job_id))


@router.post("/jobs", status_code=202)
async def create_job(payload: TrainingJobRequest, request: Request) -> dict:
    db, settings, runner = _db(request), _settings(request), _runner(request)

    profile = TRAINING_PROFILES.get(payload.profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Unknown training profile")
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    dataset_row = await db.fetch_one(
        "SELECT * FROM dataset_configs WHERE id = ?", (payload.dataset_config_id,)
    )
    if dataset_row is None:
        raise HTTPException(status_code=404, detail="Unknown dataset config")

    values = _normalize_values(payload.values)
    values["skipCacheStages"] = payload.skip_cache_stages
    try:
        validate_values(profile, values)
    except ExtraArgsError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except ValuesError as error:
        raise HTTPException(status_code=422, detail=str(error))

    resolved_paths: dict[str, Path] = {}
    for key in ("musubiPath", "outputDir", "loggingDir"):
        raw = str(values.get(key) or "").strip()
        if not raw:
            continue
        try:
            resolved_paths[key] = settings.resolve_inside_workspace(raw)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"{key} must resolve inside the workspace root"
            )

    if shutil.which("accelerate") is None:
        raise HTTPException(
            status_code=503, detail="accelerate is not available on the server"
        )
    if shutil.which("python") is None:
        raise HTTPException(
            status_code=503, detail="python is not available on the server"
        )

    musubi_path = resolved_paths["musubiPath"]
    if not musubi_path.is_dir():
        raise HTTPException(
            status_code=503, detail="musubi-tuner directory is not available"
        )
    required_scripts = [
        profile.cache_commands[0].script,
        profile.cache_commands[1].script,
        profile.trainer,
    ]
    missing_scripts = [
        name for name in required_scripts if not (musubi_path / name).is_file()
    ]
    if missing_scripts:
        raise HTTPException(
            status_code=503,
            detail=f"Required musubi-tuner script is not available: {missing_scripts[0]}",
        )

    dataset_config_toml = render_toml(json.loads(dataset_row["config_json"]))
    return await _insert_job(
        db,
        settings,
        runner,
        name=name,
        profile_id=payload.profile_id,
        dataset_config_id=payload.dataset_config_id,
        dataset_config_toml=dataset_config_toml,
        values=values,
        skip_cache_stages=payload.skip_cache_stages,
    )


@router.get("/jobs")
async def list_jobs(
    request: Request, status: str | None = Query(default=None)
) -> list[dict]:
    if status is not None and status not in JOB_STATUSES:
        raise HTTPException(status_code=400, detail="Unknown status filter")
    if status:
        rows = await _db(request).fetch_all(
            "SELECT * FROM training_jobs WHERE status = ? ORDER BY created_at DESC, id",
            (status,),
        )
    else:
        rows = await _db(request).fetch_all(
            "SELECT * FROM training_jobs ORDER BY created_at DESC, id"
        )
    return [_job_response(row) for row in rows]


@router.get("/jobs/{job_id}")
async def read_job(job_id: str, request: Request) -> dict:
    return _job_response(await _get_job(_db(request), job_id))


@router.get("/jobs/{job_id}/logs")
async def read_job_logs(
    job_id: str, request: Request, offset: int = Query(default=0, ge=0)
) -> dict:
    row = await _get_job(_db(request), job_id)
    terminal = row["status"] in TERMINAL_STATUSES
    log_path = Path(row["log_path"]) if row["log_path"] else None
    if log_path is None or not log_path.is_file():
        return {"offset": offset, "next_offset": offset, "content": "", "eof": terminal}

    size = log_path.stat().st_size
    offset = min(offset, size)
    with log_path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(LOG_CHUNK_BYTES)
    next_offset = offset + len(chunk)
    return {
        "offset": offset,
        "next_offset": next_offset,
        "content": chunk.decode("utf-8", errors="replace"),
        "eof": terminal and next_offset >= size,
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> dict:
    db, runner = _db(request), _runner(request)
    terminate_running = False
    async with runner.claim_lock:
        row = await _get_job(db, job_id)
        status = row["status"]

        if status == "cancelled":
            return _job_response(row)
        if status in {"completed", "failed"}:
            raise HTTPException(
                status_code=409, detail=f"A {status} job can no longer be cancelled"
            )

        stages = json.loads(row["stages_json"])
        if status == "running":
            for stage in stages:
                if stage["status"] == "running":
                    stage["status"] = "cancelled"
                elif stage["status"] == "pending":
                    stage["status"] = "skipped"
            await db.execute(
                "UPDATE training_jobs SET status = 'cancelled', finished_at = ?, "
                "current_stage = NULL, stages_json = ? WHERE id = ?",
                (utc_now(), json.dumps(stages), job_id),
            )
            terminate_running = runner.current_job_id == job_id
            if terminate_running:
                runner.cancel_requested = True
        else:  # queued
            await db.execute(
                "UPDATE training_jobs SET status = 'cancelled', finished_at = ?, "
                "queue_position = NULL WHERE id = ?",
                (utc_now(), job_id),
            )
            await renumber_queue(db)
    if terminate_running:
        await runner.cancel_current()
    return _job_response(await _get_job(db, job_id))


@router.post("/jobs/{job_id}/retry", status_code=202)
async def retry_job(job_id: str, request: Request) -> dict:
    db, settings, runner = _db(request), _settings(request), _runner(request)
    row = await _get_job(db, job_id)
    if row["status"] not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=409, detail="Only failed or cancelled jobs can be retried"
        )

    values = json.loads(row["values_json"])
    return await _insert_job(
        db,
        settings,
        runner,
        name=row["name"],
        profile_id=row["profile_id"],
        dataset_config_id=row["dataset_config_id"],
        dataset_config_toml=row["dataset_config_toml"],
        values=values,
        skip_cache_stages=bool(values.get("skipCacheStages", False)),
    )


@router.patch("/jobs/{job_id}")
async def reorder_job(job_id: str, payload: ReorderRequest, request: Request) -> dict:
    db, runner = _db(request), _runner(request)
    async with runner.claim_lock:
        row = await _get_job(db, job_id)
        if row["status"] != "queued":
            raise HTTPException(
                status_code=409, detail="Only queued jobs can be reordered"
            )

        async with db.write_lock:
            rows = await db.fetch_all(
                "SELECT id FROM training_jobs WHERE status = 'queued' "
                "ORDER BY queue_position, created_at"
            )
            ids = [item["id"] for item in rows]
            ids.remove(job_id)
            target = max(0, min(payload.queue_position, len(ids)))
            ids.insert(target, job_id)
            for position, item_id in enumerate(ids):
                await db.connection.execute(
                    "UPDATE training_jobs SET queue_position = ? WHERE id = ?",
                    (position, item_id),
                )
            await db.connection.commit()
    return _job_response(await _get_job(db, job_id))


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, request: Request) -> Response:
    db, settings = _db(request), _settings(request)
    row = await _get_job(db, job_id)
    if row["status"] not in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Cancel the job before deleting it")

    await db.execute("DELETE FROM training_jobs WHERE id = ?", (job_id,))
    if row["log_path"]:
        Path(row["log_path"]).unlink(missing_ok=True)
    snapshot_dir = settings.jobs_dir / job_id
    if snapshot_dir.is_dir():
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    return Response(status_code=204)
