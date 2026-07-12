"""Builds per-stage argv for training jobs.

Mirrors buildCommands in
musubi-tuner-gui-frontend/src/components/training/shared/training_workspace.tsx,
except that --dataset_config always points at the server-side TOML snapshot and
no shell is involved (argv items are passed to create_subprocess_exec verbatim).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .profiles import CacheCommand, TrainingProfile, When

EXTRA_ARG_TOKEN = re.compile(r"^[A-Za-z0-9_.\/=:,+()\-]+$")

STAGE_KEYS = ("cache_latents", "cache_text_encoder", "train")


class ExtraArgsError(ValueError):
    """An extraArgs token failed the allowlist pattern (HTTP 400)."""


class ValuesError(ValueError):
    """Typed field values are missing or invalid for the profile (HTTP 422)."""


def _text(values: dict, key: str) -> str:
    value = values.get(key)
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _enabled(values: dict, key: str) -> bool:
    return values.get(key) is True


def _matches(values: dict, when: When | None) -> bool:
    return when is None or values.get(when.key) == when.value


def _argument(values: dict, flag: str, key: str) -> list[str]:
    text = _text(values, key)
    return [flag, text] if text else []


def validate_values(profile: TrainingProfile, values: dict) -> None:
    """Check the required fields for the profile. Raises ValuesError / ExtraArgsError."""
    missing = [
        field.key
        for field in profile.model_fields
        if field.required
        and _matches(values, field.when)
        and not _text(values, field.key)
    ]
    for key in ("outputName", "outputDir", "musubiPath"):
        if not _text(values, key):
            missing.append(key)
    if missing:
        raise ValuesError(f"Missing required values: {', '.join(sorted(set(missing)))}")

    if profile.task_options:
        task = _text(values, "task")
        if task not in profile.task_options:
            raise ValuesError(f"task must be one of: {', '.join(profile.task_options)}")
    for selector in profile.selectors:
        selected = _text(values, selector.key)
        if selected not in selector.options:
            raise ValuesError(
                f"{selector.key} must be one of: {', '.join(selector.options)}"
            )

    attention = _text(values, "attention")
    if attention and attention not in profile.attention_options:
        raise ValuesError(
            f"attention must be one of: {', '.join(profile.attention_options)}"
        )

    parse_extra_args(values)


def parse_extra_args(values: dict) -> list[str]:
    raw = _text(values, "extraArgs")
    if not raw:
        return []
    try:
        tokens = shlex.split(raw)
    except ValueError as error:
        raise ExtraArgsError(f"extraArgs could not be parsed: {error}")
    for token in tokens:
        if not EXTRA_ARG_TOKEN.match(token):
            raise ExtraArgsError(f"extraArgs token is not allowed: {token!r}")
    return tokens


def _cache_argv(
    command: CacheCommand,
    values: dict,
    musubi_path: Path,
    dataset_config_path: Path,
) -> list[str]:
    argv = ["python", str(musubi_path / command.script)]
    for option in command.options:
        if not _matches(values, option.when):
            continue
        if option.enabled_key is not None:
            if _enabled(values, option.enabled_key):
                argv.append(option.flag)
            continue
        if option.key is not None:
            if option.key == "datasetConfig":
                argv += [option.flag, str(dataset_config_path)]
            else:
                argv += _argument(values, option.flag, option.key)
            continue
        if option.value is not None:
            argv += [option.flag, option.value]
            continue
        argv.append(option.flag)
    return argv


def _train_argv(
    profile: TrainingProfile,
    values: dict,
    musubi_path: Path,
    dataset_config_path: Path,
) -> list[str]:
    publishing = bool(_text(values, "huggingfaceRepoId"))
    argument = lambda flag, key: _argument(values, flag, key)  # noqa: E731

    train_fields = [*profile.model_fields, *profile.advanced_fields]
    args: list[str] = [
        "--dataset_config",
        str(dataset_config_path),
        *argument("--output_name", "outputName"),
        *argument("--output_dir", "outputDir"),
    ]
    if profile.task_options:
        args += argument("--task", "task")
    for selector in profile.selectors:
        if selector.train_flag:
            args += argument(selector.train_flag, selector.key)
    for field in train_fields:
        if field.train_flag and _matches(values, field.when):
            args += argument(field.train_flag, field.key)
    args += [
        "--network_module",
        profile.network_module,
        *argument("--optimizer_type", "optimizer"),
        *argument("--lr_scheduler", "scheduler"),
        *argument("--mixed_precision", "mixedPrecision"),
        *argument("--timestep_sampling", "timestepSampling"),
        *argument("--weighting_scheme", "weightingScheme"),
        *argument("--logging_dir", "loggingDir"),
        *argument("--max_train_epochs", "epochs"),
        *argument("--save_every_n_epochs", "saveEvery"),
        *argument("--network_dim", "networkDim"),
        *argument("--network_alpha", "networkAlpha"),
        *argument("--learning_rate", "learningRate"),
        *argument("--lr_warmup_steps", "warmupSteps"),
        *argument("--lr_scheduler_power", "schedulerPower"),
        *argument("--lr_scheduler_min_lr_ratio", "schedulerMinRatio"),
        *argument("--lr_scheduler_num_cycles", "schedulerCycles"),
        *argument("--blocks_to_swap", "blocksToSwap"),
        *argument("--discrete_flow_shift", "flowShift"),
        *argument("--seed", "seed"),
        *argument("--max_data_loader_n_workers", "workers"),
        *argument("--guidance_scale", "guidanceScale"),
        *argument("--timestep_boundary", "timestepBoundary"),
        *argument("--min_timestep", "minTimestep"),
        *argument("--max_timestep", "maxTimestep"),
    ]
    if _text(values, "networkArgs"):
        args += ["--network_args", *_text(values, "networkArgs").split()]
    if _text(values, "optimizerArgs"):
        args += ["--optimizer_args", *_text(values, "optimizerArgs").split()]
    if _enabled(values, "gradientCheckpointing"):
        args.append("--gradient_checkpointing")
    if _enabled(values, "persistentWorkers"):
        args.append("--persistent_data_loader_workers")
    for flag in profile.memory_flags:
        if flag.train and _enabled(values, flag.key):
            args.append(flag.flag)
    args += profile.fixed_train_flags
    attention = _text(values, "attention")
    if attention:
        args.append(f"--{attention}")
    if publishing:
        args += argument("--huggingface_repo_id", "huggingfaceRepoId")
        args += argument("--huggingface_repo_type", "huggingfaceRepoType")
        args += argument("--huggingface_path_in_repo", "huggingfacePath")
        args += argument("--huggingface_token", "huggingfaceToken")
        args += argument("--huggingface_repo_visibility", "huggingfaceVisibility")
        if _enabled(values, "asyncUpload"):
            args.append("--async_upload")
    args += parse_extra_args(values)

    accelerate = [
        "accelerate",
        "launch",
        "--dynamo_backend",
        _text(values, "dynamoBackend") or "no",
        "--dynamo_mode",
        _text(values, "dynamoMode") or "default",
        "--mixed_precision",
        _text(values, "mixedPrecision") or "no",
        "--num_processes",
        _text(values, "numProcesses") or "1",
        "--num_machines",
        _text(values, "numMachines") or "1",
        "--num_cpu_threads_per_process",
        _text(values, "numCpuThreadsPerProcess") or str(profile.cpu_threads),
        str(musubi_path / profile.trainer),
    ]
    return accelerate + args


def build_stage_argv(
    profile: TrainingProfile,
    values: dict,
    musubi_path: Path,
    dataset_config_path: Path,
) -> dict[str, list[str]]:
    """argv for each stage key: cache_latents, cache_text_encoder, train."""
    latents, text_encoder = profile.cache_commands
    return {
        "cache_latents": _cache_argv(latents, values, musubi_path, dataset_config_path),
        "cache_text_encoder": _cache_argv(
            text_encoder, values, musubi_path, dataset_config_path
        ),
        "train": _train_argv(profile, values, musubi_path, dataset_config_path),
    }


def redact_argv(argv: list[str]) -> list[str]:
    """Copy of argv safe for logging: the Hugging Face token is masked."""
    redacted = list(argv)
    for index, item in enumerate(redacted[:-1]):
        if item == "--huggingface_token":
            redacted[index + 1] = "***"
    return redacted
