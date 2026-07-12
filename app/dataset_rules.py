"""Validation and TOML round-tripping for musubi-tuner dataset configs.

The stored config is the TOML document mapped one-to-one to JSON. Unknown keys
are preserved and reported as warnings; rule violations are errors (HTTP 422).
"""

from __future__ import annotations

import re

SOURCE_KEYS = (
    "image_directory",
    "image_jsonl_file",
    "video_directory",
    "video_jsonl_file",
)
DIRECTORY_SOURCE_KEYS = {"image_directory", "video_directory"}
JSONL_SOURCE_KEYS = {"image_jsonl_file", "video_jsonl_file"}
VIDEO_SOURCE_KEYS = {"video_directory", "video_jsonl_file"}

FRAME_EXTRACTIONS = {"head", "chunk", "slide", "uniform", "full"}

GENERAL_KEYS = {
    "resolution",
    "caption_extension",
    "batch_size",
    "num_repeats",
    "enable_bucket",
    "bucket_no_upscale",
}

# Architecture-specific keys from the dataset config spec are passed through untouched.
ARCHITECTURE_KEYS = {
    "control_directory",
    "control_path",
    "no_resize_control",
    "control_resolution",
    "fp_latent_window_size",
    "fp_1f_clean_indices",
    "fp_1f_target_index",
    "fp_1f_no_post",
    "multiple_target",
}

COMMON_DATASET_KEYS = GENERAL_KEYS | ARCHITECTURE_KEYS | {"cache_directory"}
VIDEO_SAMPLING_KEYS = {
    "target_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
    "source_fps",
}
IMAGE_DATASET_KEYS = COMMON_DATASET_KEYS | {"image_directory", "image_jsonl_file"}
VIDEO_DATASET_KEYS = (
    COMMON_DATASET_KEYS | VIDEO_SAMPLING_KEYS | {"video_directory", "video_jsonl_file"}
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_resolution(value: object) -> bool:
    if _is_number(value):
        return True
    return (
        isinstance(value, list)
        and 1 <= len(value) <= 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def validate_config(general: object, datasets: object) -> tuple[list[str], list[str]]:
    """Apply the dataset config validation rules.

    Returns (errors, warnings). Errors mean HTTP 422; warnings are reported but
    the config is stored as-is.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if general is None:
        general = {}
    if not isinstance(general, dict):
        return (["general must be a table"], [])
    if not isinstance(datasets, list) or not datasets:
        return (["datasets must be a non-empty array"], [])

    for key in general:
        if key not in GENERAL_KEYS:
            warnings.append(f"general: unknown key '{key}' was preserved")

    if "resolution" in general and not _is_valid_resolution(general["resolution"]):
        errors.append(
            "general: resolution must be a number or an array of one or two integers"
        )

    cache_directories: dict[str, int] = {}

    for index, dataset in enumerate(datasets):
        label = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            errors.append(f"{label} must be a table")
            continue

        present_sources = [key for key in SOURCE_KEYS if key in dataset]
        if len(present_sources) != 1:
            errors.append(
                f"{label}: exactly one of image_directory, image_jsonl_file, "
                f"video_directory, or video_jsonl_file is required"
            )
            continue
        source_key = present_sources[0]
        is_video = source_key in VIDEO_SOURCE_KEYS
        if not _is_non_empty_string(dataset[source_key]):
            errors.append(f"{label}: {source_key} must be a non-empty path string")

        known_keys = VIDEO_DATASET_KEYS if is_video else IMAGE_DATASET_KEYS
        for key in dataset:
            if key in known_keys:
                continue
            if not is_video and key in VIDEO_SAMPLING_KEYS:
                warnings.append(
                    f"{label}: key '{key}' does not apply to an image dataset and was preserved"
                )
            else:
                warnings.append(f"{label}: unknown key '{key}' was preserved")

        if "resolution" not in general and "resolution" not in dataset:
            errors.append(f"{label}: resolution must be set here or in [general]")
        if "resolution" in dataset and not _is_valid_resolution(dataset["resolution"]):
            errors.append(
                f"{label}: resolution must be a number or an array of one or two integers"
            )

        if source_key in DIRECTORY_SOURCE_KEYS:
            caption_extension = dataset.get(
                "caption_extension", general.get("caption_extension")
            )
            if not _is_non_empty_string(caption_extension):
                errors.append(
                    f"{label}: caption_extension must be set here or in [general]"
                )
            if not _is_non_empty_string(dataset.get("cache_directory")):
                warnings.append(
                    f"{label}: cache_directory is recommended for directory datasets"
                )
        else:
            if not _is_non_empty_string(dataset.get("cache_directory")):
                errors.append(
                    f"{label}: cache_directory is required for JSONL datasets"
                )

        cache_directory = dataset.get("cache_directory")
        if _is_non_empty_string(cache_directory):
            normalized_cache_directory = cache_directory.strip()
            if normalized_cache_directory in cache_directories:
                errors.append(
                    f"{label}: cache_directory '{cache_directory}' is also used by "
                    f"datasets[{cache_directories[normalized_cache_directory]}]; each dataset needs its own"
                )
            else:
                cache_directories[normalized_cache_directory] = index

        if is_video:
            _validate_video_dataset(dataset, label, errors, warnings)

        if "source_fps" in dataset and not _is_number(dataset["source_fps"]):
            errors.append(f"{label}: source_fps must be a number")

    return errors, warnings


def _validate_video_dataset(
    dataset: dict, label: str, errors: list[str], warnings: list[str]
) -> None:
    frame_extraction = dataset.get("frame_extraction", "head")
    if frame_extraction not in FRAME_EXTRACTIONS:
        errors.append(
            f"{label}: frame_extraction must be one of head, chunk, slide, uniform, full"
        )
        return

    target_frames = dataset.get("target_frames")
    if frame_extraction != "full" and target_frames is None:
        errors.append(
            f"{label}: target_frames is required unless frame_extraction is 'full'"
        )
    if target_frames is not None:
        if not isinstance(target_frames, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 1
            for item in target_frames
        ):
            errors.append(
                f"{label}: target_frames must be an array of positive integers"
            )
        else:
            if frame_extraction == "chunk" and 1 in target_frames:
                warnings.append(
                    f"{label}: target_frames contains 1 with frame_extraction 'chunk'; "
                    f"single frames are better extracted with 'head'"
                )
            for value in target_frames:
                if value % 4 != 1:
                    warnings.append(
                        f"{label}: target_frames value {value} is not N*4+1; "
                        f"the trainer expects frame counts like 1, 5, 25, 45"
                    )

    if "max_frames" in dataset and frame_extraction != "full":
        warnings.append(
            f"{label}: max_frames only applies when frame_extraction is 'full'"
        )
    if "frame_stride" in dataset and frame_extraction != "slide":
        warnings.append(
            f"{label}: frame_stride only applies when frame_extraction is 'slide'"
        )
    if "frame_sample" in dataset and frame_extraction != "uniform":
        warnings.append(
            f"{label}: frame_sample only applies when frame_extraction is 'uniform'"
        )


def normalize_config(general: dict, datasets: list, extras: dict | None = None) -> dict:
    """Canonical stored form: coerce source_fps to float so TOML renders 30.0, not 30."""
    normalized_datasets = []
    for dataset in datasets:
        dataset = dict(dataset)
        if "source_fps" in dataset and _is_number(dataset["source_fps"]):
            dataset["source_fps"] = float(dataset["source_fps"])
        normalized_datasets.append(dataset)
    return {
        **(extras or {}),
        "general": dict(general or {}),
        "datasets": normalized_datasets,
    }


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "dataset-config"
