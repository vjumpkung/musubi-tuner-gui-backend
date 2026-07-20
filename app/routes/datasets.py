"""Dataset API routes and request orchestration."""

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
from starlette.datastructures import Headers

from ..config import Settings
from ..db import Database, utc_now
from ..schemas.datasets import DatasetPayload, ManagedFinalizePayload
from ..utils.dataset_rules import (
    JSONL_SOURCE_KEYS,
    SOURCE_KEYS,
    normalize_config,
    slugify,
    validate_config,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
CAPTION_EXTENSION = ".txt"
MAX_CAPTION_FILE_BYTES = 1024 * 1024
MAX_MANAGED_OPTIONS_BYTES = 64 * 1024
STAGED_DATA_SUFFIX = ".data"
STAGED_METADATA_SUFFIX = ".json"
STAGED_FILE_KINDS = ("target", "caption", "control")
STALE_MANAGED_PREFIXES = (".orphan-", ".pending-delete-", ".tombstone-")
EDIT_MANAGED_PREFIXES = (".edit-backup-", ".edit-")
MANAGED_DATASET_OWNED_KEYS = set(SOURCE_KEYS) | {
    "cache_directory",
    "caption_extension",
    "control_directory",
    "resolution",
    "num_repeats",
    "target_frames",
    "batch_size",
    "enable_bucket",
    "bucket_no_upscale",
}

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


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


def _edit_state(entry: Path) -> tuple[str, str] | None:
    prefix = next(
        (
            candidate
            for candidate in EDIT_MANAGED_PREFIXES
            if entry.name.startswith(candidate)
        ),
        None,
    )
    if prefix is None:
        return None
    candidate = entry.name[len(prefix) : len(prefix) + 36]
    try:
        config_id = str(uuid.UUID(candidate))
    except ValueError as error:
        raise OSError(f"Unrecognized managed edit state: {entry.name}") from error
    return prefix, config_id


def _remove_managed_entry(entry: Path) -> None:
    if entry.is_symlink() or entry.is_file():
        entry.unlink()
    else:
        shutil.rmtree(entry)


def cleanup_staged_uploads(root: Path) -> None:
    """Remove incomplete browser upload sessions during application startup."""
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        return
    for entry in resolved_root.iterdir():
        if entry.parent.resolve() != resolved_root:
            raise OSError("Staged upload cleanup escaped its storage root")
        _remove_managed_entry(entry)


def _managed_directory_matches_config(directory: Path, config: dict) -> bool:
    config_path = directory / "dataset_config.toml"
    if not config_path.is_file():
        return False
    try:
        return tomllib.loads(config_path.read_text(encoding="utf-8")) == config
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False


async def _reconcile_managed_edits(db: Database, root: Path) -> None:
    grouped: dict[str, list[Path]] = {}
    for entry in list(root.iterdir()):
        state = _edit_state(entry)
        if state is not None:
            grouped.setdefault(state[1], []).append(entry)

    for config_id, entries in grouped.items():
        row = await db.fetch_one(
            "SELECT config_json FROM dataset_configs WHERE id = ?",
            (config_id,),
        )
        destination = root / config_id
        if row is None:
            for entry in entries:
                _remove_managed_entry(entry)
            continue

        config = json.loads(row["config_json"])
        candidates = ([destination] if destination.exists() else []) + entries
        matching = next(
            (
                candidate
                for candidate in candidates
                if _managed_directory_matches_config(candidate, config)
            ),
            None,
        )
        if matching is not None and matching != destination:
            if destination.exists():
                _remove_managed_entry(destination)
            matching.rename(destination)
        elif matching is None and not destination.exists():
            backup = next(
                (entry for entry in entries if entry.name.startswith(".edit-backup-")),
                None,
            )
            if backup is not None:
                backup.rename(destination)

        for entry in entries:
            if entry.exists():
                _remove_managed_entry(entry)


async def reconcile_managed_storage(db: Database, root: Path) -> None:
    """Reconcile interrupted filesystem state against committed config ownership."""
    async with db.write_lock:
        await _reconcile_managed_edits(db, root)
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
            _remove_managed_entry(entry)


def _rename_managed_directory(source: Path, destination: Path) -> None:
    source.rename(destination)


def _managed_storage_bytes(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(directory) / filename
            total += path.stat(follow_symlinks=False).st_size
    return total


def _uuid_value(raw: str, label: str) -> str:
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError) as error:
        raise HTTPException(status_code=404, detail=f"Unknown {label}") from error


def _staged_session_directory(settings: Settings, session_id: str) -> Path:
    canonical_id = _uuid_value(session_id, "upload session")
    root = settings.managed_uploads_dir.resolve()
    candidate = (root / canonical_id).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="Unknown upload session")
    return candidate


def _staged_token_paths(directory: Path, token: str) -> tuple[Path, Path]:
    canonical_token = _uuid_value(token, "staged upload")
    return (
        directory / f"{canonical_token}{STAGED_DATA_SUFFIX}",
        directory / f"{canonical_token}{STAGED_METADATA_SUFFIX}",
    )


def _available_staged_tokens(directory: Path) -> set[str]:
    tokens: set[str] = set()
    for metadata_path in directory.glob(f"*{STAGED_METADATA_SUFFIX}"):
        token = metadata_path.name.removesuffix(STAGED_METADATA_SUFFIX)
        data_path = directory / f"{token}{STAGED_DATA_SUFFIX}"
        if data_path.is_file():
            tokens.add(token)
    return tokens


def _open_staged_uploads(
    directory: Path, tokens: list[str], expected_kind: str
) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    try:
        for token in tokens:
            data_path, metadata_path = _staged_token_paths(directory, token)
            if not data_path.is_file() or not metadata_path.is_file():
                raise HTTPException(
                    status_code=422, detail="Unknown staged upload token"
                )
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise HTTPException(
                    status_code=500, detail="Staged upload metadata is unavailable"
                ) from error
            filename = metadata.get("filename")
            content_type = metadata.get("content_type")
            kind = metadata.get("kind")
            if (
                not isinstance(filename, str)
                or not isinstance(content_type, str)
                or kind not in STAGED_FILE_KINDS
            ):
                raise HTTPException(
                    status_code=500, detail="Staged upload metadata is invalid"
                )
            if kind != expected_kind:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Staged file '{filename}' is a {kind} file, not a "
                        f"{expected_kind} file"
                    ),
                )
            uploads.append(
                UploadFile(
                    file=data_path.open("rb"),
                    size=data_path.stat().st_size,
                    filename=filename,
                    headers=Headers({"content-type": content_type}),
                )
            )
        return uploads
    except BaseException:
        for upload in uploads:
            upload.file.close()
        raise


async def _claim_staged_uploads(
    request: Request, session_id: str, token_groups: list[list[str]]
) -> tuple[Path, list[list[UploadFile]]]:
    if len(token_groups) != len(STAGED_FILE_KINDS):
        raise RuntimeError("Staged uploads require target, caption, and control groups")
    settings = request.app.state.settings
    async with request.app.state.managed_upload_lock:
        directory = _staged_session_directory(settings, session_id)
        tokens = [token for group in token_groups for token in group]
        canonical_tokens = [_uuid_value(token, "staged upload") for token in tokens]
        if len(canonical_tokens) != len(set(canonical_tokens)):
            raise HTTPException(
                status_code=422, detail="Staged upload tokens must not be reused"
            )
        if set(canonical_tokens) != _available_staged_tokens(directory):
            raise HTTPException(
                status_code=422,
                detail="Finalize request must reference every staged upload exactly once",
            )
        claimed_directory = directory.with_name(
            f".finalizing-{directory.name}-{uuid.uuid4()}"
        )
        directory.rename(claimed_directory)

    uploads: list[list[UploadFile]] = []
    offset = 0
    try:
        for expected_kind, group in zip(STAGED_FILE_KINDS, token_groups, strict=True):
            group_tokens = canonical_tokens[offset : offset + len(group)]
            uploads.append(
                _open_staged_uploads(claimed_directory, group_tokens, expected_kind)
            )
            offset += len(group)
        return claimed_directory, uploads
    except BaseException:
        for group in uploads:
            for upload in group:
                upload.file.close()
        if claimed_directory.exists():
            _remove_managed_entry(claimed_directory)
        raise


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


def _managed_general_config(
    batch_size: int, enable_bucket: bool, bucket_no_upscale: bool
) -> dict:
    if not _positive_int(batch_size):
        raise HTTPException(
            status_code=422, detail="batch_size must be a positive integer"
        )
    return {
        "caption_extension": CAPTION_EXTENSION,
        "batch_size": batch_size,
        "enable_bucket": enable_bucket,
        "bucket_no_upscale": bucket_no_upscale,
    }


def _has_nested_table(value: object) -> bool:
    if isinstance(value, dict):
        return True
    return isinstance(value, list) and any(_has_nested_table(item) for item in value)


def _managed_additional_options(spec: dict, label: str) -> dict:
    """Parse a TOML key/value fragment without allowing managed fields to be replaced."""
    raw = spec.get("additional_options", "")
    if not isinstance(raw, str):
        raise HTTPException(
            status_code=422, detail=f"{label}.additional_options must be TOML text"
        )
    if len(raw.encode("utf-8")) > MAX_MANAGED_OPTIONS_BYTES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{label}.additional_options must not exceed "
                f"{MAX_MANAGED_OPTIONS_BYTES} bytes"
            ),
        )
    try:
        document = tomllib.loads(f"[options]\n{raw}")
    except tomllib.TOMLDecodeError as error:
        raise HTTPException(
            status_code=422,
            detail=f"{label}.additional_options is invalid TOML: {error}",
        ) from error
    if set(document) != {"options"}:
        raise HTTPException(
            status_code=422,
            detail=f"{label}.additional_options may contain only key/value options",
        )
    options = document["options"]
    if any(_has_nested_table(value) for value in options.values()):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.additional_options may not contain tables or inline tables",
        )
    protected = sorted(MANAGED_DATASET_OWNED_KEYS.intersection(options))
    if protected:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{label}.additional_options cannot replace managed option(s): "
                f"{', '.join(protected)}"
            ),
        )
    return options


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


def _managed_control_names(
    control_names: list[str],
    media_names: list[str],
    managed_media_filenames: list[str],
) -> list[str]:
    results: list[str] = []
    controls_by_media: dict[int, list[str]] = {}
    seen_control_stems: set[str] = set()
    media_indexes: dict[str, list[int]] = {}
    for index, name in enumerate(media_names):
        media_indexes.setdefault(Path(name).stem.casefold(), []).append(index)

    for control_name in control_names:
        basename = _upload_basename(control_name)
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


def _managed_control_filenames(
    control_files: list[UploadFile],
    media_indexes: dict[str, list[int]],
    managed_media_filenames: list[str],
) -> list[str]:
    media_names = [""] * len(managed_media_filenames)
    for stem, indexes in media_indexes.items():
        for index in indexes:
            media_names[index] = stem
    return _managed_control_names(
        [_upload_basename(upload.filename) for upload in control_files],
        media_names,
        managed_media_filenames,
    )


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

    additional_options = _managed_additional_options(spec, label)

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
        "additional_options": additional_options,
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


@router.post("/managed/upload-sessions", status_code=201)
async def create_managed_upload_session(request: Request) -> dict:
    settings = request.app.state.settings
    session_id = str(uuid.uuid4())
    directory = settings.managed_uploads_dir / session_id
    async with request.app.state.managed_upload_lock:
        directory.mkdir(exist_ok=False)
    return {"id": session_id}


@router.post("/managed/upload-sessions/{session_id}/files", status_code=201)
async def upload_managed_session_file(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
    kind: str = Form(...),
) -> dict:
    settings = request.app.state.settings
    filename = _upload_basename(file.filename)
    token = str(uuid.uuid4())
    data_path: Path | None = None
    metadata_path: Path | None = None
    partial_path: Path | None = None
    try:
        if kind not in STAGED_FILE_KINDS:
            raise HTTPException(
                status_code=422,
                detail="kind must be 'target', 'caption', or 'control'",
            )
        async with request.app.state.managed_upload_lock:
            directory = _staged_session_directory(settings, session_id)
            if len(_available_staged_tokens(directory)) >= settings.managed_max_files:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "A managed dataset upload may contain at most "
                        f"{settings.managed_max_files} files"
                    ),
                )

            session_bytes = _managed_storage_bytes(directory)
            staged_bytes = _managed_storage_bytes(settings.managed_uploads_dir)
            managed_bytes = _managed_storage_bytes(settings.managed_datasets_dir)
            data_path, metadata_path = _staged_token_paths(directory, token)
            partial_path = directory / f".{token}.part"
            file_bytes = 0
            with partial_path.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    file_bytes += len(chunk)
                    if file_bytes > settings.managed_max_file_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Uploaded file '{filename}' exceeds the "
                                f"{settings.managed_max_file_bytes}-byte limit"
                            ),
                        )
                    if session_bytes + file_bytes > settings.managed_max_total_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Managed dataset uploads exceed the "
                                f"{settings.managed_max_total_bytes}-byte total limit"
                            ),
                        )
                    if (
                        managed_bytes + staged_bytes + file_bytes
                        > settings.managed_max_storage_bytes
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail="Managed dataset storage quota would be exceeded",
                        )
                    output.write(chunk)
            partial_path.rename(data_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "filename": filename,
                        "content_type": file.content_type or "application/octet-stream",
                        "kind": kind,
                    }
                ),
                encoding="utf-8",
            )
        return {
            "token": token,
            "filename": filename,
            "size_bytes": file_bytes,
            "kind": kind,
        }
    except BaseException:
        for path in (partial_path, data_path, metadata_path):
            if path is not None and path.exists():
                path.unlink()
        raise
    finally:
        await file.close()


@router.delete("/managed/upload-sessions/{session_id}", status_code=204)
async def delete_managed_upload_session(session_id: str, request: Request) -> Response:
    settings = request.app.state.settings
    async with request.app.state.managed_upload_lock:
        directory = _staged_session_directory(settings, session_id)
        _remove_managed_entry(directory)
    return Response(status_code=204)


async def _close_staged_groups(groups: list[list[UploadFile]]) -> None:
    for group in groups:
        for upload in group:
            await upload.close()


def _discard_claimed_upload(directory: Path) -> None:
    if not directory.exists():
        return
    try:
        _remove_managed_entry(directory)
    except OSError:
        # The startup cleanup retries this without turning an already-committed
        # dataset into an apparent finalize failure.
        pass


@router.post("/managed/upload-sessions/{session_id}/finalize", status_code=201)
async def finalize_managed_upload_session(
    session_id: str, payload: ManagedFinalizePayload, request: Request
) -> dict:
    claimed_directory, groups = await _claim_staged_uploads(
        request,
        session_id,
        [
            payload.file_tokens,
            payload.caption_file_tokens,
            payload.control_file_tokens,
        ],
    )
    files, caption_files, control_files = groups
    try:
        return await create_managed_dataset_batch(
            request=request,
            name=payload.name,
            dataset_specs=json.dumps(payload.dataset_specs),
            files=files,
            description=payload.description,
            caption_files=caption_files,
            control_files=control_files,
            batch_size=payload.batch_size,
            enable_bucket=payload.enable_bucket,
            bucket_no_upscale=payload.bucket_no_upscale,
        )
    finally:
        await _close_staged_groups(groups)
        _discard_claimed_upload(claimed_directory)


@router.put(
    "/managed/upload-sessions/{session_id}/datasets/{config_id}/finalize"
)
async def finalize_managed_update_session(
    session_id: str,
    config_id: str,
    payload: ManagedFinalizePayload,
    request: Request,
) -> dict:
    claimed_directory, groups = await _claim_staged_uploads(
        request,
        session_id,
        [
            payload.file_tokens,
            payload.caption_file_tokens,
            payload.control_file_tokens,
        ],
    )
    files, caption_files, control_files = groups
    try:
        return await update_managed_dataset_files(
            config_id=config_id,
            request=request,
            name=payload.name,
            dataset_specs=json.dumps(payload.dataset_specs),
            description=payload.description,
            files=files,
            caption_files=caption_files,
            control_files=control_files,
            batch_size=payload.batch_size,
            enable_bucket=payload.enable_bucket,
            bucket_no_upscale=payload.bucket_no_upscale,
        )
    finally:
        await _close_staged_groups(groups)
        _discard_claimed_upload(claimed_directory)


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


def _relative_managed_id(managed_directory: Path, path: Path) -> str:
    return path.resolve().relative_to(managed_directory.resolve()).as_posix()


def _managed_file_manifest(managed_directory: Path, config: dict) -> dict:
    datasets_manifest: list[dict] = []
    for index, dataset in enumerate(config.get("datasets", [])):
        if not isinstance(dataset, dict):
            continue
        if isinstance(dataset.get("image_directory"), str):
            media_type = "image"
            source = Path(dataset["image_directory"]).resolve()
            allowed_extensions = IMAGE_EXTENSIONS
        elif isinstance(dataset.get("video_directory"), str):
            media_type = "video"
            source = Path(dataset["video_directory"]).resolve()
            allowed_extensions = VIDEO_EXTENSIONS
        else:
            raise HTTPException(
                status_code=422,
                detail="Only directory-backed managed datasets can be edited",
            )

        if (
            not source.is_dir()
            or source == managed_directory
            or managed_directory not in source.parents
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Managed dataset {index + 1} media directory is unavailable",
            )

        media_files = []
        for path in sorted(
            source.iterdir(), key=lambda candidate: candidate.name.casefold()
        ):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix.lower() not in allowed_extensions
            ):
                continue
            caption_path = path.with_suffix(CAPTION_EXTENSION)
            try:
                caption = caption_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError) as error:
                raise HTTPException(
                    status_code=409,
                    detail=f"Caption for managed file '{path.name}' is unavailable",
                ) from error
            media_files.append(
                {
                    "path": _relative_managed_id(managed_directory, path),
                    "name": path.name,
                    "size_bytes": path.stat(follow_symlinks=False).st_size,
                    "caption": caption,
                }
            )

        control_files = []
        control_value = dataset.get("control_directory")
        if isinstance(control_value, str):
            control_directory = Path(control_value).resolve()
            if (
                not control_directory.is_dir()
                or control_directory == managed_directory
                or managed_directory not in control_directory.parents
            ):
                raise HTTPException(
                    status_code=409,
                    detail=f"Managed dataset {index + 1} control directory is unavailable",
                )
            for path in sorted(
                control_directory.iterdir(),
                key=lambda candidate: candidate.name.casefold(),
            ):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue
                control_files.append(
                    {
                        "path": _relative_managed_id(managed_directory, path),
                        "name": path.name,
                        "size_bytes": path.stat(follow_symlinks=False).st_size,
                    }
                )

        datasets_manifest.append(
            {
                "index": index,
                "media_type": media_type,
                "files": media_files,
                "control_files": control_files,
            }
        )
    return {"datasets": datasets_manifest}


def _manifest_file_maps(managed_directory: Path, manifest: dict) -> tuple[dict, dict]:
    media_files: dict[str, dict] = {}
    control_files: dict[str, dict] = {}
    for dataset in manifest["datasets"]:
        for media in dataset["files"]:
            media_files[media["path"]] = {
                **media,
                "media_type": dataset["media_type"],
                "source": managed_directory / Path(media["path"]),
            }
        for control in dataset["control_files"]:
            control_files[control["path"]] = {
                **control,
                "source": managed_directory / Path(control["path"]),
            }
    return media_files, control_files


async def _prepare_managed_update_dataset(
    spec: object,
    files: list[UploadFile],
    caption_files: list[UploadFile],
    control_files: list[UploadFile],
    index: int,
    available_media: dict[str, dict],
    available_controls: dict[str, dict],
    claimed_media: set[str],
    claimed_controls: set[str],
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
    additional_options = _managed_additional_options(spec, label)
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

    existing_specs = spec.get("existing_files", [])
    if not isinstance(existing_specs, list):
        raise HTTPException(
            status_code=422, detail=f"{label}.existing_files must be an array"
        )
    existing_files: list[dict] = []
    for existing in existing_specs:
        if not isinstance(existing, dict):
            raise HTTPException(
                status_code=422,
                detail=f"{label}.existing_files entries must be objects",
            )
        path = existing.get("path")
        caption = existing.get("caption")
        current = available_media.get(path) if isinstance(path, str) else None
        if (
            current is None
            or current["media_type"] != media_type
            or path in claimed_media
        ):
            raise HTTPException(
                status_code=422,
                detail=f"{label}.existing_files contains an unknown or duplicate path",
            )
        if not isinstance(caption, str) or not caption.strip():
            raise HTTPException(
                status_code=422,
                detail=f"{label}.existing_files captions must not be empty",
            )
        claimed_media.add(path)
        existing_files.append({**current, "caption": caption.strip()})

    captions = spec.get("captions", [])
    if not isinstance(captions, list) or not all(
        isinstance(caption, str) for caption in captions
    ):
        raise HTTPException(
            status_code=422, detail=f"{label}.captions must be an array of strings"
        )
    if len(captions) != len(files):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.captions must contain one entry for each uploaded file",
        )
    parsed_captions = list(captions)
    media_indexes = _media_stem_indexes(files)
    sidecar_captions = await _captions_from_sidecars(caption_files, media_indexes)
    for media_index, caption in sidecar_captions.items():
        parsed_captions[media_index] = caption
    if any(not caption.strip() for caption in parsed_captions):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.captions must not contain empty or whitespace-only entries",
        )
    if not existing_files and not files:
        raise HTTPException(
            status_code=422, detail=f"{label} must contain at least one media file"
        )

    allowed_extensions = IMAGE_EXTENSIONS if media_type == "image" else VIDEO_EXTENSIONS
    used_stems: set[str] = set()
    managed_filenames: list[str] = []
    media_names: list[str] = []
    for existing in existing_files:
        stem = Path(existing["name"]).stem.casefold()
        if stem in used_stems:
            raise HTTPException(
                status_code=422,
                detail=f"{label} contains duplicate existing media stems",
            )
        used_stems.add(stem)
        managed_filenames.append(existing["name"])
        media_names.append(existing["name"])
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
        media_names.append(_upload_basename(upload.filename))

    existing_control_paths = spec.get("existing_control_files", [])
    if not isinstance(existing_control_paths, list) or not all(
        isinstance(path, str) for path in existing_control_paths
    ):
        raise HTTPException(
            status_code=422,
            detail=f"{label}.existing_control_files must be an array of paths",
        )
    existing_controls: list[dict] = []
    for path in existing_control_paths:
        current = available_controls.get(path)
        if current is None or path in claimed_controls:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{label}.existing_control_files contains an unknown or duplicate path"
                ),
            )
        claimed_controls.add(path)
        existing_controls.append(current)

    if (existing_controls or control_files) and media_type != "image":
        raise HTTPException(
            status_code=422,
            detail=f"{label}: control images are supported only for image datasets",
        )
    control_names = [control["name"] for control in existing_controls] + [
        _upload_basename(upload.filename) for upload in control_files
    ]
    managed_control_filenames = (
        _managed_control_names(control_names, media_names, managed_filenames)
        if control_names
        else []
    )

    return {
        "media_type": media_type,
        "resolution": resolution,
        "num_repeats": num_repeats,
        "target_frames": target_frames,
        "additional_options": additional_options,
        "existing_files": existing_files,
        "files": files,
        "captions": [caption.strip() for caption in parsed_captions],
        "managed_filenames": managed_filenames,
        "existing_controls": existing_controls,
        "control_files": control_files,
        "managed_control_filenames": managed_control_filenames,
    }


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
    batch_size: int = Form(1),
    enable_bucket: bool = Form(True),
    bucket_no_upscale: bool = Form(False),
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
        {
            "resolution": parsed_resolution,
            **_managed_general_config(
                batch_size, enable_bucket, bucket_no_upscale
            ),
        },
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
    batch_size: int = Form(1),
    enable_bucket: bool = Form(True),
    bucket_no_upscale: bool = Form(False),
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
                raise HTTPException(
                    status_code=422, detail=f"{label} must be an object"
                )
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
                    caption_files[caption_offset : caption_offset + caption_file_count],
                    control_files[control_offset : control_offset + control_file_count],
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
                **prepared["additional_options"],
                source_key: str(media_directory.resolve()),
                "cache_directory": str(cache_directory.resolve()),
                "resolution": prepared["resolution"],
                "num_repeats": prepared["num_repeats"],
            }
            if prepared["control_files"]:
                dataset["control_directory"] = str(control_directory.resolve())
            if media_type == "video":
                dataset["target_frames"] = prepared["target_frames"]
                dataset.setdefault("frame_extraction", "head")
            datasets.append(dataset)

        config = normalize_config(
            _managed_general_config(
                batch_size, enable_bucket, bucket_no_upscale
            ),
            datasets,
        )
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
    resource = _resource(row)
    config = json.loads(row["config_json"])
    resource["managed"] = (
        _owned_managed_directory(request, config_id, config) is not None
    )
    return resource


@router.get("/{config_id}/managed-files")
async def read_managed_dataset_files(config_id: str, request: Request) -> dict:
    row = await _get_row(_db(request), config_id)
    config = json.loads(row["config_json"])
    managed_directory = _owned_managed_directory(request, config_id, config)
    if managed_directory is None:
        raise HTTPException(
            status_code=422,
            detail="Only server-managed dataset files can be edited",
        )
    return _managed_file_manifest(managed_directory, config)


@router.put("/{config_id}/managed")
async def update_managed_dataset_files(
    config_id: str,
    request: Request,
    name: str = Form(...),
    dataset_specs: str = Form(...),
    description: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
    caption_files: list[UploadFile] | None = File(None),
    control_files: list[UploadFile] | None = File(None),
    batch_size: int = Form(1),
    enable_bucket: bool = Form(True),
    bucket_no_upscale: bool = Form(False),
) -> dict:
    """Replace a managed config while retaining selected owned files."""
    db = _db(request)
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    clean_description = (
        description.strip() if description and description.strip() else None
    )
    parsed_specs = _json_form_value(dataset_specs, "dataset_specs")
    if not isinstance(parsed_specs, list) or not parsed_specs:
        raise HTTPException(
            status_code=422, detail="dataset_specs must be a non-empty JSON array"
        )

    files = files or []
    caption_files = caption_files or []
    control_files = control_files or []
    settings = request.app.state.settings
    upload_count = len(files) + len(caption_files) + len(control_files)
    if upload_count > settings.managed_max_files:
        raise HTTPException(
            status_code=413,
            detail=f"A managed dataset upload may contain at most {settings.managed_max_files} files",
        )

    try:
        existing_row = await _get_row(db, config_id)
        existing_config = json.loads(existing_row["config_json"])
        managed_directory = _owned_managed_directory(
            request, config_id, existing_config
        )
        if managed_directory is None or not managed_directory.is_dir():
            raise HTTPException(
                status_code=422,
                detail="Only server-managed dataset files can be edited",
            )
        manifest = _managed_file_manifest(managed_directory, existing_config)
        available_media, available_controls = _manifest_file_maps(
            managed_directory, manifest
        )

        file_offset = 0
        caption_offset = 0
        control_offset = 0
        grouped_uploads: list[
            tuple[object, list[UploadFile], list[UploadFile], list[UploadFile]]
        ] = []
        for index, spec in enumerate(parsed_specs):
            label = f"dataset_specs[{index}]"
            if not isinstance(spec, dict):
                raise HTTPException(
                    status_code=422, detail=f"{label} must be an object"
                )
            file_count = spec.get("file_count", 0)
            caption_file_count = spec.get("caption_file_count", 0)
            control_file_count = spec.get("control_file_count", 0)
            if not _non_negative_int(file_count):
                raise HTTPException(
                    status_code=422,
                    detail=f"{label}.file_count must be a non-negative integer",
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
                    caption_files[caption_offset : caption_offset + caption_file_count],
                    control_files[control_offset : control_offset + control_file_count],
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

        claimed_media: set[str] = set()
        claimed_controls: set[str] = set()
        prepared_datasets = [
            await _prepare_managed_update_dataset(
                spec,
                dataset_files,
                dataset_caption_files,
                dataset_control_files,
                index,
                available_media,
                available_controls,
                claimed_media,
                claimed_controls,
            )
            for index, (
                spec,
                dataset_files,
                dataset_caption_files,
                dataset_control_files,
            ) in enumerate(grouped_uploads)
        ]

        token = str(uuid.uuid4())
        temporary_directory = (
            settings.managed_datasets_dir / f".edit-{config_id}-{token}"
        )
        backup_directory = (
            settings.managed_datasets_dir / f".edit-backup-{config_id}-{token}"
        )
        datasets: list[dict] = []
        dataset_locations: list[tuple[Path, Path, Path]] = []
        for index, prepared in enumerate(prepared_datasets, start=1):
            final_dataset_directory = managed_directory / f"dataset-{index}"
            final_media_directory = final_dataset_directory / "media"
            final_cache_directory = final_dataset_directory / "cache"
            final_control_directory = final_dataset_directory / "control"
            temporary_dataset_directory = temporary_directory / f"dataset-{index}"
            media_directory = temporary_dataset_directory / "media"
            cache_directory = temporary_dataset_directory / "cache"
            control_directory = temporary_dataset_directory / "control"
            dataset_locations.append(
                (media_directory, cache_directory, control_directory)
            )

            source_key = (
                "image_directory"
                if prepared["media_type"] == "image"
                else "video_directory"
            )
            dataset = {
                **prepared["additional_options"],
                source_key: str(final_media_directory.resolve()),
                "cache_directory": str(final_cache_directory.resolve()),
                "resolution": prepared["resolution"],
                "num_repeats": prepared["num_repeats"],
            }
            if prepared["existing_controls"] or prepared["control_files"]:
                dataset["control_directory"] = str(final_control_directory.resolve())
            if prepared["media_type"] == "video":
                dataset["target_frames"] = prepared["target_frames"]
                dataset.setdefault("frame_extraction", "head")
            datasets.append(dataset)

        config = normalize_config(
            _managed_general_config(
                batch_size, enable_bucket, bucket_no_upscale
            ),
            datasets,
        )
        warnings = _validate_or_422(config["general"], config["datasets"])
        config_toml = render_toml(config)
        caption_bytes = sum(
            len(file["caption"].encode("utf-8"))
            for prepared in prepared_datasets
            for file in prepared["existing_files"]
        ) + sum(
            len(caption.encode("utf-8"))
            for prepared in prepared_datasets
            for caption in prepared["captions"]
        )

        async with request.app.state.managed_storage_lock:
            try:
                await reconcile_managed_storage(db, settings.managed_datasets_dir)
                latest_row = await db.fetch_one(
                    "SELECT updated_at FROM dataset_configs WHERE id = ?",
                    (config_id,),
                )
                if latest_row is None:
                    raise HTTPException(
                        status_code=404, detail="Unknown dataset config"
                    )
                if latest_row["updated_at"] != existing_row["updated_at"]:
                    raise HTTPException(
                        status_code=409,
                        detail="The dataset changed while it was being edited; reload and try again",
                    )
                existing_bytes = _managed_storage_bytes(settings.managed_datasets_dir)
                current_bytes = _managed_storage_bytes(managed_directory)
            except OSError as error:
                raise HTTPException(
                    status_code=500,
                    detail="Could not clean or measure managed dataset storage",
                ) from error
            retained_bytes = sum(
                file["source"].stat(follow_symlinks=False).st_size
                for prepared in prepared_datasets
                for file in [
                    *prepared["existing_files"],
                    *prepared["existing_controls"],
                ]
            )
            fixed_bytes = (
                retained_bytes + caption_bytes + len(config_toml.encode("utf-8"))
            )
            existing_without_current = max(0, existing_bytes - current_bytes)
            if (
                existing_without_current + fixed_bytes
                > settings.managed_max_storage_bytes
            ):
                raise HTTPException(
                    status_code=413,
                    detail="Managed dataset storage quota would be exceeded",
                )

            replaced_directory = False
            try:
                total_bytes = 0
                for prepared, locations in zip(
                    prepared_datasets, dataset_locations, strict=True
                ):
                    media_directory, cache_directory, control_directory = locations
                    media_directory.mkdir(parents=True)
                    cache_directory.mkdir()
                    if prepared["existing_controls"] or prepared["control_files"]:
                        control_directory.mkdir()

                    existing_count = len(prepared["existing_files"])
                    for existing, filename in zip(
                        prepared["existing_files"],
                        prepared["managed_filenames"][:existing_count],
                        strict=True,
                    ):
                        shutil.copy2(existing["source"], media_directory / filename)
                        (media_directory / filename).with_suffix(".txt").write_text(
                            existing["caption"], encoding="utf-8"
                        )
                    for upload, filename, caption in zip(
                        prepared["files"],
                        prepared["managed_filenames"][existing_count:],
                        prepared["captions"],
                        strict=True,
                    ):
                        total_bytes = await _write_managed_upload(
                            upload,
                            media_directory / filename,
                            filename,
                            settings,
                            total_bytes,
                            existing_without_current,
                            fixed_bytes,
                        )
                        (media_directory / filename).with_suffix(".txt").write_text(
                            caption, encoding="utf-8"
                        )

                    existing_control_count = len(prepared["existing_controls"])
                    for existing, filename in zip(
                        prepared["existing_controls"],
                        prepared["managed_control_filenames"][:existing_control_count],
                        strict=True,
                    ):
                        shutil.copy2(existing["source"], control_directory / filename)
                    for upload, filename in zip(
                        prepared["control_files"],
                        prepared["managed_control_filenames"][existing_control_count:],
                        strict=True,
                    ):
                        total_bytes = await _write_managed_upload(
                            upload,
                            control_directory / filename,
                            filename,
                            settings,
                            total_bytes,
                            existing_without_current,
                            fixed_bytes,
                        )

                (temporary_directory / "dataset_config.toml").write_text(
                    config_toml, encoding="utf-8"
                )
                now = utc_now()
                async with db.write_lock:
                    current_row = await db.fetch_one(
                        "SELECT id FROM dataset_configs WHERE id = ?",
                        (config_id,),
                    )
                    if current_row is None:
                        raise HTTPException(
                            status_code=404, detail="Unknown dataset config"
                        )
                    name_row = await db.fetch_one(
                        "SELECT id FROM dataset_configs WHERE name = ? AND id != ?",
                        (clean_name, config_id),
                    )
                    if name_row is not None:
                        raise HTTPException(
                            status_code=409,
                            detail=f"A dataset config named '{clean_name}' already exists",
                        )
                    job_row = await db.fetch_one(
                        "SELECT id FROM training_jobs WHERE dataset_config_id = ? LIMIT 1",
                        (config_id,),
                    )
                    if job_row is not None:
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "Managed dataset files cannot be edited while referencing "
                                "training jobs exist"
                            ),
                        )
                    try:
                        _rename_managed_directory(managed_directory, backup_directory)
                        _rename_managed_directory(
                            temporary_directory, managed_directory
                        )
                        replaced_directory = True
                        await db.connection.execute(
                            "UPDATE dataset_configs SET name = ?, description = ?, "
                            "config_json = ?, updated_at = ? WHERE id = ?",
                            (
                                clean_name,
                                clean_description,
                                json.dumps(config),
                                now,
                                config_id,
                            ),
                        )
                        await db.connection.commit()
                    except BaseException as error:
                        await db.connection.rollback()
                        try:
                            if replaced_directory and managed_directory.exists():
                                _remove_managed_entry(managed_directory)
                            if backup_directory.exists():
                                _rename_managed_directory(
                                    backup_directory, managed_directory
                                )
                        except OSError as restore_error:
                            raise HTTPException(
                                status_code=500,
                                detail=(
                                    "Managed dataset update failed and recovery is pending"
                                ),
                            ) from restore_error
                        if isinstance(error, HTTPException):
                            raise error
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                "Managed dataset update failed; its original files were restored"
                            ),
                        ) from error

                if backup_directory.exists():
                    try:
                        _remove_managed_entry(backup_directory)
                    except OSError:
                        pass
                return {
                    "id": config_id,
                    "name": clean_name,
                    "description": clean_description,
                    "general": config["general"],
                    "datasets": config["datasets"],
                    "created_at": existing_row["created_at"],
                    "updated_at": now,
                    "warnings": warnings,
                    "managed": True,
                }
            except BaseException:
                if temporary_directory.exists():
                    _remove_managed_entry(temporary_directory)
                raise
    finally:
        for upload in [*files, *caption_files, *control_files]:
            await upload.close()


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
        if len(old_sources) != len(new_sources) or any(
            old_key != new_key or Path(old_path).resolve() != Path(new_path).resolve()
            for (old_key, old_path), (new_key, new_path) in zip(
                old_sources, new_sources, strict=True
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
