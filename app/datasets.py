"""Dataset manager: stored musubi-tuner dataset configs with TOML import/export."""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
import uuid
from pathlib import Path

import aiosqlite
import tomli_w
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from .config import Settings
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
CAPTION_EXTENSION = ".txt"
MAX_CAPTION_FILE_BYTES = 1024 * 1024
STALE_MANAGED_PREFIXES = (".orphan-", ".pending-delete-", ".tombstone-")

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetPayload(BaseModel):
    name: str
    description: str | None = None
    general: dict = {}
    datasets: list


class ManagedCleanupError(OSError):
    def __init__(self, path: Path):
        super().__init__(f"Managed storage cleanup failed: {path.name}")
        self.path = path


def _stale_state(entry: Path) -> tuple[str, str] | None:
    prefix = next(
        (
            candidate
            for candidate in STALE_MANAGED_PREFIXES
            if entry.name.startswith(candidate)
        ),
        None,
    )
    if prefix is None:
        return None
    remainder = entry.name[len(prefix) :]
    candidate = remainder[:36]
    try:
        config_id = str(uuid.UUID(candidate))
    except ValueError as error:
        raise OSError(f"Unrecognized managed storage state: {entry.name}") from error
    if len(remainder) <= 36 or remainder[36] != "-":
        raise OSError(f"Unrecognized managed storage state: {entry.name}")
    return prefix, config_id


async def reconcile_managed_storage(db: Database, root: Path) -> None:
    """Reconcile interrupted filesystem state against committed config ownership."""
    async with db.write_lock:
        for entry in root.iterdir():
            state = _stale_state(entry)
            if state is None:
                continue
            prefix, config_id = state
            row = await db.fetch_one(
                "SELECT id FROM dataset_configs WHERE id = ?",
                (config_id,),
            )
            if row is not None and prefix in {".pending-delete-", ".orphan-"}:
                destination = root / config_id
                if destination.exists():
                    raise OSError(
                        f"Cannot restore '{entry.name}': managed destination already exists"
                    )
                entry.rename(destination)
                continue
            if row is not None:
                raise OSError(
                    f"Preserving '{entry.name}' because its dataset config still exists"
                )
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                shutil.rmtree(entry)


def _rename_managed_directory(source: Path, destination: Path) -> None:
    source.rename(destination)


def _managed_storage_bytes(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            total += path.stat(follow_symlinks=False).st_size
    return total


def _cleanup_failed_managed_creation(root: Path, directory: Path) -> None:
    if not directory.exists():
        return
    try:
        shutil.rmtree(directory)
        return
    except OSError:
        orphan = root / f".orphan-{directory.name}-{uuid.uuid4()}"
        try:
            directory.rename(orphan)
        except OSError:
            orphan = directory
        raise ManagedCleanupError(orphan)


def _json_form_value(raw: str, field_name: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be valid JSON: {error.msg}",
        ) from error


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _upload_basename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=422, detail="Every uploaded file must have a filename"
        )
    return Path(filename.replace("\\", "/")).name.strip()


def _managed_filename(filename: str | None, used_stems: set[str]) -> str:
    basename = _upload_basename(filename)

    extension = Path(basename).suffix.lower()
    stem = Path(basename).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_. ") or "upload"
    # Windows device names are unsafe even when followed by an extension.
    if safe_stem.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }:
        safe_stem = f"upload-{safe_stem}"

    candidate = safe_stem
    suffix = 2
    while candidate.casefold() in used_stems:
        candidate = f"{safe_stem}-{suffix}"
        suffix += 1
    used_stems.add(candidate.casefold())
    return f"{candidate}{extension}"


def _media_stem_indexes(files: list[UploadFile]) -> dict[str, list[int]]:
    indexes: dict[str, list[int]] = {}
    for index, upload in enumerate(files):
        stem = Path(_upload_basename(upload.filename)).stem.casefold()
        indexes.setdefault(stem, []).append(index)
    return indexes


async def _captions_from_sidecars(
    caption_files: list[UploadFile],
    media_indexes: dict[str, list[int]],
) -> dict[int, str]:
    captions: dict[int, str] = {}
    seen_stems: set[str] = set()
    for upload in caption_files:
        basename = _upload_basename(upload.filename)
        path = Path(basename)
        if path.suffix.lower() != CAPTION_EXTENSION:
            raise HTTPException(
                status_code=422,
                detail=f"Caption sidecar '{basename}' must use the .txt extension",
            )
        stem = path.stem.casefold()
        if stem in seen_stems:
            raise HTTPException(
                status_code=422,
                detail=f"More than one caption sidecar was supplied for '{path.stem}'",
            )
        seen_stems.add(stem)
        matches = media_indexes.get(stem, [])
        if not matches:
            raise HTTPException(
                status_code=422,
                detail=f"Caption sidecar '{basename}' has no matching media file",
            )
        if len(matches) != 1:
            raise HTTPException(
                status_code=422,
                detail=f"Caption sidecar '{basename}' matches more than one media file",
            )
        raw = await upload.read(MAX_CAPTION_FILE_BYTES + 1)
        if len(raw) > MAX_CAPTION_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Caption sidecar '{basename}' exceeds the 1 MiB limit",
            )
        try:
            caption = raw.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=422,
                detail=f"Caption sidecar '{basename}' must be UTF-8 text",
            ) from error
        if not caption:
            raise HTTPException(
                status_code=422,
                detail=f"Caption sidecar '{basename}' must not be empty",
            )
        captions[matches[0]] = caption
    return captions


def _managed_control_filenames(
    control_files: list[UploadFile],
    media_indexes: dict[str, list[int]],
    managed_media_filenames: list[str],
) -> list[str]:
    results: list[str] = []
    controls_by_media: dict[int, list[str]] = {}
    seen_control_stems: set[str] = set()

    for upload in control_files:
        basename = _upload_basename(upload.filename)
        path = Path(basename)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported control image extension: {path.suffix or '(none)'}",
            )
        stem = path.stem
        folded_stem = stem.casefold()
        if folded_stem in seen_control_stems:
            raise HTTPException(
                status_code=422,
                detail=f"Duplicate control image name: '{basename}'",
            )
        seen_control_stems.add(folded_stem)

        suffix = ""
        matches = media_indexes.get(folded_stem, [])
        if not matches:
            numbered = re.fullmatch(r"(.+)_([0-9]+)", stem)
            if numbered:
                matches = media_indexes.get(numbered.group(1).casefold(), [])
                suffix = f"_{numbered.group(2)}"
        if not matches:
            raise HTTPException(
                status_code=422,
                detail=f"Control image '{basename}' has no matching target image",
            )
        if len(matches) != 1:
            raise HTTPException(
                status_code=422,
                detail=f"Control image '{basename}' matches more than one target image",
            )

        media_index = matches[0]
        target_stem = Path(managed_media_filenames[media_index]).stem
        managed_name = f"{target_stem}{suffix}{path.suffix.lower()}"
        controls_by_media.setdefault(media_index, []).append(suffix)
        results.append(managed_name)

    missing = [
        Path(managed_media_filenames[index]).name
        for index in range(len(managed_media_filenames))
        if index not in controls_by_media
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Every target image needs a control image; missing: {', '.join(missing)}",
        )
    for media_index, suffixes in controls_by_media.items():
        if "" in suffixes and len(suffixes) > 1:
            target = Path(managed_media_filenames[media_index]).name
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Control images for '{target}' must use either the exact stem or "
                    "numbered stems, not both"
                ),
            )
    return results


async def _prepare_managed_batch_dataset(
    spec: object,
    files: list[UploadFile],
    caption_files: list[UploadFile],
    control_files: list[UploadFile],
    index: int,
) -> dict:
    label = f"dataset_specs[{index}]"
    if not isinstance(spec, dict):
        raise HTTPException(status_code=422, detail=f"{label} must be an object")

    media_type = spec.get("media_type")
    if media_type not in {"image", "video"}:
        raise HTTPException(
            status_code=422,
            detail=f"{label}.media_type must be 'image' or 'video'",
        )
    if not files:
        raise HTTPException(
            status_code=422, detail=f"{label} must contain at least one media file"
        )

    resolution = spec.get("resolution")
    if not (
        isinstance(resolution, list)
        and len(resolution) == 2
        and all(_positive_int(value) for value in resolution)
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.resolution must be an array of two positive integers",
        )

    num_repeats = spec.get("num_repeats", 1)
    if not _positive_int(num_repeats):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.num_repeats must be a positive integer",
        )

    captions = spec.get("captions")
    if captions is None:
        parsed_captions = [""] * len(files)
    elif not isinstance(captions, list) or not all(
        isinstance(caption, str) for caption in captions
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.captions must be an array of strings",
        )
    elif len(captions) != len(files):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.captions must contain one entry for each media file",
        )
    else:
        parsed_captions = list(captions)

    target_frames = spec.get("target_frames")
    if target_frames is not None and (
        not isinstance(target_frames, list)
        or not target_frames
        or not all(_positive_int(value) for value in target_frames)
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.target_frames must be a non-empty array of positive integers",
        )
    if media_type == "video" and target_frames is None:
        target_frames = [1]
    if media_type == "image" and target_frames is not None:
        raise HTTPException(
            status_code=422,
            detail=f"{label}.target_frames only applies to video datasets",
        )

    allowed_extensions = IMAGE_EXTENSIONS if media_type == "image" else VIDEO_EXTENSIONS
    managed_filenames: list[str] = []
    used_stems: set[str] = set()
    for upload in files:
        filename = _managed_filename(upload.filename, used_stems)
        if Path(filename).suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported {media_type} file extension in {label}: "
                    f"{Path(filename).suffix or '(none)'}"
                ),
            )
        managed_filenames.append(filename)

    media_indexes = _media_stem_indexes(files)
    sidecar_captions = await _captions_from_sidecars(caption_files, media_indexes)
    for media_index, caption in sidecar_captions.items():
        parsed_captions[media_index] = caption
    if any(not caption.strip() for caption in parsed_captions):
        raise HTTPException(
            status_code=422,
            detail=(
                f"{label}.captions must not contain empty or whitespace-only entries; "
                "every media file needs a caption or matching .txt sidecar"
            ),
        )

    if control_files and media_type != "image":
        raise HTTPException(
            status_code=422,
            detail=f"{label}: control images are supported only for image datasets",
        )
    managed_control_filenames = (
        _managed_control_filenames(control_files, media_indexes, managed_filenames)
        if control_files
        else []
    )

    return {
        "media_type": media_type,
        "resolution": resolution,
        "num_repeats": num_repeats,
        "target_frames": target_frames,
        "files": files,
        "captions": parsed_captions,
        "control_files": control_files,
        "managed_filenames": managed_filenames,
        "managed_control_filenames": managed_control_filenames,
    }


async def _write_managed_upload(
    upload: UploadFile,
    destination: Path,
    display_name: str,
    settings: Settings,
    total_bytes: int,
    existing_bytes: int,
    fixed_bytes: int,
) -> int:
    file_bytes = 0
    with destination.open("xb") as output:
        while chunk := await upload.read(1024 * 1024):
            file_bytes += len(chunk)
            total_bytes += len(chunk)
            if file_bytes > settings.managed_max_file_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Uploaded file '{display_name}' exceeds the "
                        f"{settings.managed_max_file_bytes}-byte limit"
                    ),
                )
            if total_bytes > settings.managed_max_total_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Managed dataset uploads exceed the "
                        f"{settings.managed_max_total_bytes}-byte total limit"
                    ),
                )
            if (
                existing_bytes + fixed_bytes + total_bytes
                > settings.managed_max_storage_bytes
            ):
                raise HTTPException(
                    status_code=413,
                    detail="Managed dataset storage quota would be exceeded",
                )
            output.write(chunk)
    return total_bytes


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
        raise HTTPException(
            status_code=422, detail=f"Dataset config is not TOML-compatible: {error}"
        )
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


async def _commit_managed_delete(db: Database) -> None:
    await db.connection.commit()


def _owned_managed_directory(
    request: Request, config_id: str, config: dict
) -> Path | None:
    """Return the managed directory when every config source is confined there."""
    managed_root = request.app.state.settings.managed_datasets_dir.resolve()
    candidate = (managed_root / config_id).resolve()
    if candidate.parent != managed_root:
        return None

    source_paths = [
        dataset[key]
        for dataset in config.get("datasets", [])
        if isinstance(dataset, dict)
        for key in SOURCE_KEYS
        if key in dataset and isinstance(dataset[key], str)
    ]
    if not source_paths:
        return None
    for source_path in source_paths:
        source = Path(source_path).resolve()
        if source != candidate and candidate not in source.parents:
            return None
    return candidate


def _source_entries(config: dict) -> list[tuple[str, str]]:
    return [
        (key, dataset[key])
        for dataset in config.get("datasets", [])
        if isinstance(dataset, dict)
        for key in SOURCE_KEYS
        if key in dataset and isinstance(dataset[key], str)
    ]


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
        raise HTTPException(
            status_code=409, detail=f"A dataset config named '{name}' already exists"
        )

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
        raise HTTPException(
            status_code=409, detail=f"A dataset config named '{name}' already exists"
        )
    row = await _get_row(db, config_id)
    return {**_resource(row), "warnings": warnings}


@router.post("/managed", status_code=201)
async def create_managed_dataset(
    request: Request,
    name: str = Form(...),
    media_type: str = Form(...),
    resolution: str = Form(...),
    files: list[UploadFile] = File(...),
    captions: str | None = Form(None),
    description: str | None = Form(None),
    target_frames: str | None = Form(None),
    caption_files: list[UploadFile] | None = File(None),
    control_files: list[UploadFile] | None = File(None),
) -> dict:
    """Upload captioned media and create a config pointing at managed storage."""
    db = _db(request)
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    if media_type not in {"image", "video"}:
        raise HTTPException(
            status_code=422, detail="media_type must be 'image' or 'video'"
        )
    if not files:
        raise HTTPException(
            status_code=422, detail="At least one media file is required"
        )
    caption_files = caption_files or []
    control_files = control_files or []
    settings = request.app.state.settings
    upload_count = len(files) + len(caption_files) + len(control_files)
    if upload_count > settings.managed_max_files:
        raise HTTPException(
            status_code=413,
            detail=f"A managed dataset upload may contain at most {settings.managed_max_files} files",
        )

    parsed_resolution = _json_form_value(resolution, "resolution")
    if not (
        isinstance(parsed_resolution, list)
        and len(parsed_resolution) == 2
        and all(_positive_int(value) for value in parsed_resolution)
    ):
        raise HTTPException(
            status_code=422,
            detail="resolution must be a JSON array of two positive integers",
        )

    if captions is None:
        parsed_captions = [""] * len(files)
    else:
        value = _json_form_value(captions, "captions")
        if not isinstance(value, list) or not all(
            isinstance(caption, str) for caption in value
        ):
            raise HTTPException(
                status_code=422, detail="captions must be a JSON array of strings"
            )
        if len(value) != len(files):
            raise HTTPException(
                status_code=422,
                detail="captions must contain exactly one entry for each uploaded file",
            )
        parsed_captions = list(value)

    parsed_target_frames: list[int] | None = None
    if target_frames is not None:
        value = _json_form_value(target_frames, "target_frames")
        if (
            not isinstance(value, list)
            or not value
            or not all(_positive_int(item) for item in value)
        ):
            raise HTTPException(
                status_code=422,
                detail="target_frames must be a non-empty JSON array of positive integers",
            )
        parsed_target_frames = value
    if media_type == "video" and parsed_target_frames is None:
        parsed_target_frames = [1]
    if media_type == "image" and parsed_target_frames is not None:
        raise HTTPException(
            status_code=422,
            detail="target_frames only applies to video datasets",
        )

    allowed_extensions = IMAGE_EXTENSIONS if media_type == "image" else VIDEO_EXTENSIONS
    managed_filenames: list[str] = []
    used_stems: set[str] = set()
    for upload in files:
        filename = _managed_filename(upload.filename, used_stems)
        if Path(filename).suffix.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported {media_type} file extension: {Path(filename).suffix or '(none)'}",
            )
        managed_filenames.append(filename)

    media_indexes = _media_stem_indexes(files)
    sidecar_captions = await _captions_from_sidecars(caption_files, media_indexes)
    for index, caption in sidecar_captions.items():
        parsed_captions[index] = caption
    if any(not caption.strip() for caption in parsed_captions):
        raise HTTPException(
            status_code=422,
            detail=(
                "captions must not contain empty or whitespace-only entries; every media "
                "file needs a caption or matching .txt sidecar"
            ),
        )

    if control_files and media_type != "image":
        raise HTTPException(
            status_code=422,
            detail="Control image uploads are supported only for image datasets",
        )
    managed_control_filenames = (
        _managed_control_filenames(control_files, media_indexes, managed_filenames)
        if control_files
        else []
    )

    if await _name_taken(db, clean_name):
        raise HTTPException(
            status_code=409,
            detail=f"A dataset config named '{clean_name}' already exists",
        )

    config_id = str(uuid.uuid4())
    managed_directory = request.app.state.settings.managed_datasets_dir / config_id
    media_directory = managed_directory / "media"
    cache_directory = managed_directory / "cache"
    control_directory = managed_directory / "control"
    source_key = "image_directory" if media_type == "image" else "video_directory"
    dataset = {
        source_key: str(media_directory.resolve()),
        "cache_directory": str(cache_directory.resolve()),
    }
    if control_files:
        dataset["control_directory"] = str(control_directory.resolve())
    if media_type == "video":
        dataset.update(
            {
                "target_frames": parsed_target_frames,
                "frame_extraction": "head",
            }
        )
    config = normalize_config(
        {"resolution": parsed_resolution, "caption_extension": ".txt"},
        [dataset],
    )
    warnings = _validate_or_422(config["general"], config["datasets"])

    config_toml = render_toml(config)
    fixed_bytes = len(config_toml.encode("utf-8")) + sum(
        len(caption.encode("utf-8")) for caption in parsed_captions
    )
    try:
        async with request.app.state.managed_storage_lock:
            try:
                await reconcile_managed_storage(db, settings.managed_datasets_dir)
                existing_bytes = _managed_storage_bytes(settings.managed_datasets_dir)
            except OSError as error:
                raise HTTPException(
                    status_code=500,
                    detail="Could not clean or measure managed dataset storage",
                ) from error
            if existing_bytes + fixed_bytes > settings.managed_max_storage_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Managed dataset storage quota would be exceeded",
                )

            try:
                media_directory.mkdir(parents=True)
                cache_directory.mkdir()
                if control_files:
                    control_directory.mkdir()
                total_bytes = 0
                for upload, filename, caption in zip(
                    files, managed_filenames, parsed_captions, strict=True
                ):
                    destination = media_directory / filename
                    file_bytes = 0
                    with destination.open("xb") as output:
                        while chunk := await upload.read(1024 * 1024):
                            file_bytes += len(chunk)
                            total_bytes += len(chunk)
                            if file_bytes > settings.managed_max_file_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        f"Uploaded file '{filename}' exceeds the "
                                        f"{settings.managed_max_file_bytes}-byte limit"
                                    ),
                                )
                            if total_bytes > settings.managed_max_total_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        "Managed dataset uploads exceed the "
                                        f"{settings.managed_max_total_bytes}-byte total limit"
                                    ),
                                )
                            if (
                                existing_bytes + fixed_bytes + total_bytes
                                > settings.managed_max_storage_bytes
                            ):
                                raise HTTPException(
                                    status_code=413,
                                    detail="Managed dataset storage quota would be exceeded",
                                )
                            output.write(chunk)
                    destination.with_suffix(".txt").write_text(
                        caption, encoding="utf-8"
                    )
                for upload, filename in zip(
                    control_files, managed_control_filenames, strict=True
                ):
                    destination = control_directory / filename
                    file_bytes = 0
                    with destination.open("xb") as output:
                        while chunk := await upload.read(1024 * 1024):
                            file_bytes += len(chunk)
                            total_bytes += len(chunk)
                            if file_bytes > settings.managed_max_file_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        f"Uploaded control image '{filename}' exceeds the "
                                        f"{settings.managed_max_file_bytes}-byte limit"
                                    ),
                                )
                            if total_bytes > settings.managed_max_total_bytes:
                                raise HTTPException(
                                    status_code=413,
                                    detail=(
                                        "Managed dataset uploads exceed the "
                                        f"{settings.managed_max_total_bytes}-byte total limit"
                                    ),
                                )
                            if (
                                existing_bytes + fixed_bytes + total_bytes
                                > settings.managed_max_storage_bytes
                            ):
                                raise HTTPException(
                                    status_code=413,
                                    detail="Managed dataset storage quota would be exceeded",
                                )
                            output.write(chunk)
                (managed_directory / "dataset_config.toml").write_text(
                    config_toml,
                    encoding="utf-8",
                )

                now = utc_now()
                inserted = await _try_dataset_write(
                    db,
                    "INSERT INTO dataset_configs "
                    "(id, name, description, config_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        config_id,
                        clean_name,
                        description,
                        json.dumps(config),
                        now,
                        now,
                    ),
                )
                if not inserted:
                    raise HTTPException(
                        status_code=409,
                        detail=f"A dataset config named '{clean_name}' already exists",
                    )
                return {
                    "id": config_id,
                    "name": clean_name,
                    "description": description,
                    "general": config["general"],
                    "datasets": config["datasets"],
                    "created_at": now,
                    "updated_at": now,
                    "warnings": warnings,
                }
            except BaseException as original_error:
                try:
                    _cleanup_failed_managed_creation(
                        settings.managed_datasets_dir,
                        managed_directory,
                    )
                except ManagedCleanupError as cleanup_error:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Managed dataset creation failed and cleanup is pending in "
                            f"'{cleanup_error.path.name}'"
                        ),
                    ) from cleanup_error
                raise original_error
    finally:
        for upload in [*files, *caption_files, *control_files]:
            await upload.close()


@router.post("/managed/batch", status_code=201)
async def create_managed_dataset_batch(
    request: Request,
    name: str = Form(...),
    dataset_specs: str = Form(...),
    files: list[UploadFile] = File(...),
    description: str | None = Form(None),
    caption_files: list[UploadFile] | None = File(None),
    control_files: list[UploadFile] | None = File(None),
) -> dict:
    """Upload one or more datasets into one managed TOML configuration."""
    db = _db(request)
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    parsed_specs = _json_form_value(dataset_specs, "dataset_specs")
    if not isinstance(parsed_specs, list) or not parsed_specs:
        raise HTTPException(
            status_code=422, detail="dataset_specs must be a non-empty JSON array"
        )

    caption_files = caption_files or []
    control_files = control_files or []
    settings = request.app.state.settings
    upload_count = len(files) + len(caption_files) + len(control_files)
    if upload_count > settings.managed_max_files:
        raise HTTPException(
            status_code=413,
            detail=f"A managed dataset upload may contain at most {settings.managed_max_files} files",
        )

    file_offset = 0
    caption_offset = 0
    control_offset = 0
    grouped_uploads: list[
        tuple[object, list[UploadFile], list[UploadFile], list[UploadFile]]
    ] = []
    try:
        for index, spec in enumerate(parsed_specs):
            label = f"dataset_specs[{index}]"
            if not isinstance(spec, dict):
                raise HTTPException(status_code=422, detail=f"{label} must be an object")
            file_count = spec.get("file_count")
            caption_file_count = spec.get("caption_file_count", 0)
            control_file_count = spec.get("control_file_count", 0)
            if not _positive_int(file_count):
                raise HTTPException(
                    status_code=422,
                    detail=f"{label}.file_count must be a positive integer",
                )
            if not _non_negative_int(caption_file_count):
                raise HTTPException(
                    status_code=422,
                    detail=f"{label}.caption_file_count must be a non-negative integer",
                )
            if not _non_negative_int(control_file_count):
                raise HTTPException(
                    status_code=422,
                    detail=f"{label}.control_file_count must be a non-negative integer",
                )

            grouped_uploads.append(
                (
                    spec,
                    files[file_offset : file_offset + file_count],
                    caption_files[
                        caption_offset : caption_offset + caption_file_count
                    ],
                    control_files[
                        control_offset : control_offset + control_file_count
                    ],
                )
            )
            file_offset += file_count
            caption_offset += caption_file_count
            control_offset += control_file_count

        if file_offset != len(files):
            raise HTTPException(
                status_code=422,
                detail="dataset_specs file counts do not match the uploaded media files",
            )
        if caption_offset != len(caption_files):
            raise HTTPException(
                status_code=422,
                detail="dataset_specs caption file counts do not match the uploaded caption files",
            )
        if control_offset != len(control_files):
            raise HTTPException(
                status_code=422,
                detail="dataset_specs control file counts do not match the uploaded control files",
            )

        prepared_datasets = [
            await _prepare_managed_batch_dataset(
                spec,
                dataset_files,
                dataset_caption_files,
                dataset_control_files,
                index,
            )
            for index, (
                spec,
                dataset_files,
                dataset_caption_files,
                dataset_control_files,
            ) in enumerate(grouped_uploads)
        ]

        if await _name_taken(db, clean_name):
            raise HTTPException(
                status_code=409,
                detail=f"A dataset config named '{clean_name}' already exists",
            )

        config_id = str(uuid.uuid4())
        managed_directory = settings.managed_datasets_dir / config_id
        dataset_locations: list[tuple[Path, Path, Path]] = []
        datasets: list[dict] = []
        for index, prepared in enumerate(prepared_datasets, start=1):
            dataset_directory = managed_directory / f"dataset-{index}"
            media_directory = dataset_directory / "media"
            cache_directory = dataset_directory / "cache"
            control_directory = dataset_directory / "control"
            dataset_locations.append(
                (media_directory, cache_directory, control_directory)
            )

            media_type = prepared["media_type"]
            source_key = (
                "image_directory" if media_type == "image" else "video_directory"
            )
            dataset = {
                source_key: str(media_directory.resolve()),
                "cache_directory": str(cache_directory.resolve()),
                "resolution": prepared["resolution"],
                "num_repeats": prepared["num_repeats"],
            }
            if prepared["control_files"]:
                dataset["control_directory"] = str(control_directory.resolve())
            if media_type == "video":
                dataset.update(
                    {
                        "target_frames": prepared["target_frames"],
                        "frame_extraction": "head",
                    }
                )
            datasets.append(dataset)

        config = normalize_config({"caption_extension": ".txt"}, datasets)
        warnings = _validate_or_422(config["general"], config["datasets"])
        config_toml = render_toml(config)
        fixed_bytes = len(config_toml.encode("utf-8")) + sum(
            len(caption.encode("utf-8"))
            for prepared in prepared_datasets
            for caption in prepared["captions"]
        )

        async with request.app.state.managed_storage_lock:
            try:
                await reconcile_managed_storage(db, settings.managed_datasets_dir)
                existing_bytes = _managed_storage_bytes(settings.managed_datasets_dir)
            except OSError as error:
                raise HTTPException(
                    status_code=500,
                    detail="Could not clean or measure managed dataset storage",
                ) from error
            if existing_bytes + fixed_bytes > settings.managed_max_storage_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Managed dataset storage quota would be exceeded",
                )

            try:
                total_bytes = 0
                for prepared, locations in zip(
                    prepared_datasets, dataset_locations, strict=True
                ):
                    media_directory, cache_directory, control_directory = locations
                    media_directory.mkdir(parents=True)
                    cache_directory.mkdir()
                    if prepared["control_files"]:
                        control_directory.mkdir()

                    for upload, filename, caption in zip(
                        prepared["files"],
                        prepared["managed_filenames"],
                        prepared["captions"],
                        strict=True,
                    ):
                        total_bytes = await _write_managed_upload(
                            upload,
                            media_directory / filename,
                            filename,
                            settings,
                            total_bytes,
                            existing_bytes,
                            fixed_bytes,
                        )
                        (media_directory / filename).with_suffix(".txt").write_text(
                            caption, encoding="utf-8"
                        )

                    for upload, filename in zip(
                        prepared["control_files"],
                        prepared["managed_control_filenames"],
                        strict=True,
                    ):
                        total_bytes = await _write_managed_upload(
                            upload,
                            control_directory / filename,
                            filename,
                            settings,
                            total_bytes,
                            existing_bytes,
                            fixed_bytes,
                        )

                (managed_directory / "dataset_config.toml").write_text(
                    config_toml, encoding="utf-8"
                )
                now = utc_now()
                inserted = await _try_dataset_write(
                    db,
                    "INSERT INTO dataset_configs "
                    "(id, name, description, config_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        config_id,
                        clean_name,
                        description,
                        json.dumps(config),
                        now,
                        now,
                    ),
                )
                if not inserted:
                    raise HTTPException(
                        status_code=409,
                        detail=f"A dataset config named '{clean_name}' already exists",
                    )
                return {
                    "id": config_id,
                    "name": clean_name,
                    "description": description,
                    "general": config["general"],
                    "datasets": config["datasets"],
                    "created_at": now,
                    "updated_at": now,
                    "warnings": warnings,
                }
            except BaseException as original_error:
                try:
                    _cleanup_failed_managed_creation(
                        settings.managed_datasets_dir,
                        managed_directory,
                    )
                except ManagedCleanupError as cleanup_error:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Managed dataset creation failed and cleanup is pending in "
                            f"'{cleanup_error.path.name}'"
                        ),
                    ) from cleanup_error
                raise original_error
    finally:
        for upload in [*files, *caption_files, *control_files]:
            await upload.close()


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
    extras = {
        key: value
        for key, value in document.items()
        if key not in {"general", "datasets"}
    }
    warnings.extend(f"top-level: unknown key '{key}' was preserved" for key in extras)

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
async def update_dataset(
    config_id: str, payload: DatasetPayload, request: Request
) -> dict:
    db = _db(request)
    existing_row = await _get_row(db, config_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    warnings = _validate_or_422(payload.general, payload.datasets)
    if await _name_taken(db, name, exclude_id=config_id):
        raise HTTPException(
            status_code=409, detail=f"A dataset config named '{name}' already exists"
        )

    existing_config = json.loads(existing_row["config_json"])
    extras = {
        key: value
        for key, value in existing_config.items()
        if key not in {"general", "datasets"}
    }
    config = normalize_config(payload.general, payload.datasets, extras)
    managed_directory = _owned_managed_directory(request, config_id, existing_config)
    now = utc_now()

    if managed_directory is not None:
        old_sources = _source_entries(existing_config)
        new_sources = _source_entries(config)
        if (
            len(old_sources) != len(new_sources)
            or any(
                old_key != new_key
                or Path(old_path).resolve() != Path(new_path).resolve()
                for (old_key, old_path), (new_key, new_path) in zip(
                    old_sources, new_sources, strict=True
                )
            )
        ):
            raise HTTPException(
                status_code=422,
                detail="The source paths of a managed dataset config cannot be changed",
            )

        config_path = managed_directory / "dataset_config.toml"
        temporary_path = managed_directory / f".dataset_config-{uuid.uuid4()}.tmp"
        async with db.write_lock:
            current_row = await db.fetch_one(
                "SELECT id FROM dataset_configs WHERE id = ?",
                (config_id,),
            )
            if current_row is None:
                raise HTTPException(status_code=404, detail="Unknown dataset config")
            replaced_config = False
            try:
                previous_toml = (
                    config_path.read_text(encoding="utf-8")
                    if config_path.is_file()
                    else render_toml(existing_config)
                )
                temporary_path.write_text(render_toml(config), encoding="utf-8")
                await db.connection.execute(
                    "UPDATE dataset_configs SET name = ?, description = ?, config_json = ?, "
                    "updated_at = ? WHERE id = ?",
                    (name, payload.description, json.dumps(config), now, config_id),
                )
                temporary_path.replace(config_path)
                replaced_config = True
                await db.connection.commit()
            except aiosqlite.IntegrityError:
                await db.connection.rollback()
                temporary_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=409,
                    detail=f"A dataset config named '{name}' already exists",
                )
            except BaseException:
                await db.connection.rollback()
                temporary_path.unlink(missing_ok=True)
                if replaced_config:
                    config_path.write_text(previous_toml, encoding="utf-8")
                raise
        return {
            "id": config_id,
            "name": name,
            "description": payload.description,
            "general": config["general"],
            "datasets": config["datasets"],
            "created_at": existing_row["created_at"],
            "updated_at": now,
            "warnings": warnings,
        }

    updated = await _try_dataset_write(
        db,
        "UPDATE dataset_configs SET name = ?, description = ?, config_json = ?, updated_at = ? "
        "WHERE id = ?",
        (name, payload.description, json.dumps(config), now, config_id),
    )
    if not updated:
        raise HTTPException(
            status_code=409, detail=f"A dataset config named '{name}' already exists"
        )
    row = await _get_row(db, config_id)
    return {**_resource(row), "warnings": warnings}


@router.delete("/{config_id}", status_code=204)
async def delete_dataset(config_id: str, request: Request) -> Response:
    db = _db(request)
    row = await _get_row(db, config_id)
    managed_directory = _owned_managed_directory(
        request,
        config_id,
        json.loads(row["config_json"]),
    )
    # External configs remain deletable because their media is not owned here;
    # their job references become NULL via ON DELETE SET NULL.
    if managed_directory is not None:
        settings = request.app.state.settings
        pending_delete = (
            settings.managed_datasets_dir
            / f".pending-delete-{config_id}-{uuid.uuid4()}"
        )
        tombstone = (
            settings.managed_datasets_dir / f".tombstone-{config_id}-{uuid.uuid4()}"
        )
        renamed = False
        async with request.app.state.managed_storage_lock:
            try:
                await reconcile_managed_storage(db, settings.managed_datasets_dir)
            except OSError as error:
                raise HTTPException(
                    status_code=500,
                    detail="Could not clean stale managed dataset storage",
                ) from error

            # Managed snapshots still point at uploaded media, so serialize
            # reference checking, ownership rename, and DB deletion.
            async with db.write_lock:
                job = await db.fetch_one(
                    "SELECT id FROM training_jobs WHERE dataset_config_id = ? LIMIT 1",
                    (config_id,),
                )
                if job is not None:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Delete referencing training jobs before deleting this "
                            "managed dataset"
                        ),
                    )
                try:
                    if managed_directory.exists():
                        _rename_managed_directory(managed_directory, pending_delete)
                        renamed = True
                    await db.connection.execute(
                        "DELETE FROM dataset_configs WHERE id = ?",
                        (config_id,),
                    )
                    await _commit_managed_delete(db)
                except BaseException as error:
                    await db.connection.rollback()
                    if renamed and pending_delete.exists():
                        try:
                            _rename_managed_directory(pending_delete, managed_directory)
                        except OSError as restore_error:
                            raise HTTPException(
                                status_code=500,
                                detail=(
                                    "Managed dataset deletion failed and its files remain "
                                    f"in '{pending_delete.name}'"
                                ),
                            ) from restore_error
                    if isinstance(error, HTTPException):
                        raise
                    raise HTTPException(
                        status_code=500,
                        detail="Managed dataset deletion failed; its config and files were restored",
                    ) from error

            if renamed:
                try:
                    _rename_managed_directory(pending_delete, tombstone)
                except OSError as error:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Managed dataset was deleted but pending state cleanup is "
                            f"required for '{pending_delete.name}'"
                        ),
                    ) from error
                try:
                    shutil.rmtree(tombstone)
                except OSError as error:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Managed dataset was deleted but tombstone cleanup is pending in "
                            f"'{tombstone.name}'"
                        ),
                    ) from error
    else:
        async with db.write_lock:
            await db.connection.execute(
                "DELETE FROM dataset_configs WHERE id = ?",
                (config_id,),
            )
            await db.connection.commit()
    return Response(status_code=204)


def render_toml(config: dict) -> str:
    document = {
        key: value
        for key, value in config.items()
        if key not in {"general", "datasets"}
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
            result["image_count"] = sum(
                1 for f in files if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            result["video_count"] = sum(
                1 for f in files if f.suffix.lower() in VIDEO_EXTENSIONS
            )
            caption_extension = dataset.get("caption_extension") or general.get(
                "caption_extension"
            )
            if caption_extension:
                result["caption_count"] = sum(
                    1
                    for f in files
                    if f.suffix.lower() == str(caption_extension).lower()
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
