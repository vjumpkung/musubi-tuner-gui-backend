"""Dataset manager: stored musubi-tuner dataset configs with TOML import/export."""

from __future__ import annotations

import json
import os
import tomllib
import uuid
from pathlib import Path

import aiosqlite
import tomli_w
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from .dataset_rules import (
    JSONL_SOURCE_KEYS,
    SOURCE_KEYS,
    normalize_config,
    slugify,
    validate_config,
)
from .db import Database, utc_now

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetPayload(BaseModel):
    name: str
    description: str | None = None
    general: dict = {}
    datasets: list


def _db(request: Request) -> Database:
    return request.app.state.db


def _resource(row) -> dict:
    config = json.loads(row["config_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "general": config.get("general", {}),
        "datasets": config.get("datasets", []),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def _get_row(db: Database, config_id: str):
    row = await db.fetch_one("SELECT * FROM dataset_configs WHERE id = ?", (config_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown dataset config")
    return row


def _validate_or_422(general: object, datasets: object) -> list[str]:
    errors, warnings = validate_config(general, datasets)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))
    try:
        tomli_w.dumps({"general": general or {}, "datasets": datasets})
    except (TypeError, ValueError, OverflowError) as error:
        raise HTTPException(status_code=422, detail=f"Dataset config is not TOML-compatible: {error}")
    return warnings


async def _name_taken(db: Database, name: str, exclude_id: str | None = None) -> bool:
    row = await db.fetch_one(
        "SELECT id FROM dataset_configs WHERE name = ? AND id IS NOT ?",
        (name, exclude_id),
    )
    return row is not None


async def _try_dataset_write(db: Database, sql: str, params: tuple) -> bool:
    """Execute a name-constrained write atomically; False means a name collision."""
    async with db.write_lock:
        try:
            await db.connection.execute(sql, params)
            await db.connection.commit()
            return True
        except aiosqlite.IntegrityError:
            await db.connection.rollback()
            return False


@router.get("")
async def list_datasets(request: Request) -> list[dict]:
    rows = await _db(request).fetch_all(
        "SELECT id, name, description, created_at, updated_at "
        "FROM dataset_configs ORDER BY created_at DESC, id"
    )
    return [dict(row) for row in rows]


@router.post("", status_code=201)
async def create_dataset(payload: DatasetPayload, request: Request) -> dict:
    db = _db(request)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    warnings = _validate_or_422(payload.general, payload.datasets)
    if await _name_taken(db, name):
        raise HTTPException(status_code=409, detail=f"A dataset config named '{name}' already exists")

    config = normalize_config(payload.general, payload.datasets)
    now = utc_now()
    config_id = str(uuid.uuid4())
    inserted = await _try_dataset_write(
        db,
        "INSERT INTO dataset_configs (id, name, description, config_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (config_id, name, payload.description, json.dumps(config), now, now),
    )
    if not inserted:
        raise HTTPException(status_code=409, detail=f"A dataset config named '{name}' already exists")
    row = await _get_row(db, config_id)
    return {**_resource(row), "warnings": warnings}


@router.post("/import", status_code=201)
async def import_dataset(
    request: Request,
    file: UploadFile = File(...),
    name: str | None = Form(None),
) -> dict:
    db = _db(request)
    raw = await file.read()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise HTTPException(status_code=400, detail=f"Invalid TOML: {error}")

    general = document.get("general", {})
    datasets = document.get("datasets", [])
    warnings = _validate_or_422(general, datasets)
    extras = {key: value for key, value in document.items() if key not in {"general", "datasets"}}
    warnings.extend(
        f"top-level: unknown key '{key}' was preserved" for key in extras
    )

    base_name = (name or Path(file.filename or "imported").stem).strip() or "imported"
    candidate = base_name
    suffix = 2
    while await _name_taken(db, candidate):
        candidate = f"{base_name}-{suffix}"
        suffix += 1

    config = normalize_config(general, datasets, extras)
    now = utc_now()
    config_id = str(uuid.uuid4())
    while not await _try_dataset_write(
        db,
        "INSERT INTO dataset_configs (id, name, description, config_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (config_id, candidate, None, json.dumps(config), now, now),
    ):
        candidate = f"{base_name}-{suffix}"
        suffix += 1
    row = await _get_row(db, config_id)
    return {**_resource(row), "warnings": warnings}


@router.get("/{config_id}")
async def read_dataset(config_id: str, request: Request) -> dict:
    row = await _get_row(_db(request), config_id)
    return _resource(row)


@router.put("/{config_id}")
async def update_dataset(config_id: str, payload: DatasetPayload, request: Request) -> dict:
    db = _db(request)
    existing_row = await _get_row(db, config_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    warnings = _validate_or_422(payload.general, payload.datasets)
    if await _name_taken(db, name, exclude_id=config_id):
        raise HTTPException(status_code=409, detail=f"A dataset config named '{name}' already exists")

    existing_config = json.loads(existing_row["config_json"])
    extras = {
        key: value
        for key, value in existing_config.items()
        if key not in {"general", "datasets"}
    }
    config = normalize_config(payload.general, payload.datasets, extras)
    updated = await _try_dataset_write(
        db,
        "UPDATE dataset_configs SET name = ?, description = ?, config_json = ?, updated_at = ? "
        "WHERE id = ?",
        (name, payload.description, json.dumps(config), utc_now(), config_id),
    )
    if not updated:
        raise HTTPException(status_code=409, detail=f"A dataset config named '{name}' already exists")
    row = await _get_row(db, config_id)
    return {**_resource(row), "warnings": warnings}


@router.delete("/{config_id}", status_code=204)
async def delete_dataset(config_id: str, request: Request) -> Response:
    db = _db(request)
    await _get_row(db, config_id)
    # Always allowed: training jobs keep their own TOML snapshot and their
    # dataset_config_id becomes NULL via ON DELETE SET NULL.
    await db.execute("DELETE FROM dataset_configs WHERE id = ?", (config_id,))
    return Response(status_code=204)


def render_toml(config: dict) -> str:
    document = {
        key: value for key, value in config.items() if key not in {"general", "datasets"}
    }
    if config.get("general"):
        document["general"] = config["general"]
    document["datasets"] = config.get("datasets", [])
    return tomli_w.dumps(document)


@router.get("/{config_id}/export")
async def export_dataset(config_id: str, request: Request) -> Response:
    row = await _get_row(_db(request), config_id)
    body = render_toml(json.loads(row["config_json"]))
    filename = f"{slugify(row['name'])}.toml"
    return Response(
        content=body,
        media_type="application/toml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _creatable(path: Path) -> bool:
    """True when the directory exists or could be created without touching anything."""
    if path.is_dir():
        return os.access(path, os.W_OK)
    if path.exists():
        return False
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            return False
        ancestor = ancestor.parent
    return ancestor.is_dir() and os.access(ancestor, os.W_OK)


def _check_dataset_paths(dataset: dict, general: dict) -> dict:
    source_key = next((key for key in SOURCE_KEYS if key in dataset), None)
    result: dict = {"source_key": source_key, "source_path": dataset.get(source_key)}
    if source_key is None:
        result["exists"] = False
        return result

    source = Path(str(dataset[source_key]))
    if source_key in JSONL_SOURCE_KEYS:
        result["exists"] = source.is_file()
        if result["exists"]:
            try:
                with source.open("r", encoding="utf-8", errors="replace") as handle:
                    entries = sum(1 for line in handle if line.strip())
            except OSError:
                entries = 0
            key = "image_count" if source_key == "image_jsonl_file" else "video_count"
            result[key] = entries
    else:
        result["exists"] = source.is_dir()
        if result["exists"]:
            files = [entry for entry in source.iterdir() if entry.is_file()]
            result["image_count"] = sum(1 for f in files if f.suffix.lower() in IMAGE_EXTENSIONS)
            result["video_count"] = sum(1 for f in files if f.suffix.lower() in VIDEO_EXTENSIONS)
            caption_extension = dataset.get("caption_extension") or general.get("caption_extension")
            if caption_extension:
                result["caption_count"] = sum(
                    1 for f in files if f.suffix.lower() == str(caption_extension).lower()
                )

    cache_directory = dataset.get("cache_directory")
    if cache_directory:
        result["cache_directory"] = cache_directory
        result["cache_directory_creatable"] = _creatable(Path(str(cache_directory)))
    return result


@router.post("/{config_id}/validate")
async def validate_dataset_paths(config_id: str, request: Request) -> dict:
    row = await _get_row(_db(request), config_id)
    config = json.loads(row["config_json"])
    general = config.get("general", {})
    return {
        "id": row["id"],
        "datasets": [
            {"index": index, **_check_dataset_paths(dataset, general)}
            for index, dataset in enumerate(config.get("datasets", []))
        ],
    }
