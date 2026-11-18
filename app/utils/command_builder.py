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

from ..profiles import CacheCommand, CacheOption, TrainingProfile, When

EXTRA_ARG_TOKEN = re.compile(r"^[A-Za-z0-9_.\/=:,+()\-]+$")

STAGE_KEYS = ("cache_latents", "cache_text_encoder", "train")
SAVE_PRECISIONS = ("bf16", "fp16", "fp32")

# (value key, train flag) pairs that _train_argv renders as "<flag> <value>", in
# argv order.
TRAIN_VALUE_FLAGS = (
    ("optimizer", "--optimizer_type"),
    ("scheduler", "--lr_scheduler"),
    ("mixedPrecision", "--mixed_precision"),
    ("savePrecision", "--save_precision"),
    ("timestepSampling", "--timestep_sampling"),
    ("weightingScheme", "--weighting_scheme"),
    ("loggingDir", "--logging_dir"),
    ("epochs", "--max_train_epochs"),
    ("saveEvery", "--save_every_n_epochs"),
    ("networkDim", "--network_dim"),
    ("networkAlpha", "--network_alpha"),
    ("learningRate", "--learning_rate"),
    ("warmupSteps", "--lr_warmup_steps"),
    ("schedulerPower", "--lr_scheduler_power"),
    ("schedulerMinRatio", "--lr_scheduler_min_lr_ratio"),
    ("schedulerCycles", "--lr_scheduler_num_cycles"),
    ("blocksToSwap", "--blocks_to_swap"),
    ("flowShift", "--discrete_flow_shift"),
    ("seed", "--seed"),
    ("workers", "--max_data_loader_n_workers"),
    ("guidanceScale", "--guidance_scale"),
    ("timestepBoundary", "--timestep_boundary"),
    ("minTimestep", "--min_timestep"),
    ("maxTimestep", "--max_timestep"),
)
BOOLEAN_TRAIN_FLAGS = (
    ("gradientCheckpointing", "--gradient_checkpointing"),
    ("persistentWorkers", "--persistent_data_loader_workers"),
    ("asyncUpload", "--async_upload"),
)
HUGGINGFACE_VALUE_FLAGS = (
    ("huggingfaceRepoId", "--huggingface_repo_id"),
    ("huggingfaceRepoType", "--huggingface_repo_type"),
    ("huggingfacePath", "--huggingface_path_in_repo"),
    ("huggingfaceToken", "--huggingface_token"),
    ("huggingfaceVisibility", "--huggingface_repo_visibility"),
)
ACCELERATE_VALUE_FLAGS = (
    ("dynamoBackend", "--dynamo_backend"),
    ("dynamoMode", "--dynamo_mode"),
    ("numProcesses", "--num_processes"),
    ("numMachines", "--num_machines"),
    ("numCpuThreadsPerProcess", "--num_cpu_threads_per_process"),
)
# Values the server sets or ignores itself; never reported as unknown.
SERVER_VALUE_KEYS = frozenset({"datasetConfig", "skipCacheStages"})


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


def condition_matches(values: dict, when: When | None) -> bool:
    if when is None:
        return True
    value = values.get(when.key)
    if isinstance(value, str):
        value = value.strip()
    if isinstance(when.value, tuple):
        return value in when.value
    return value == when.value


def _argument(values: dict, flag: str, key: str) -> list[str]:
    text = _text(values, key)
    return [flag, text] if text else []


def describe_values(profile: TrainingProfile) -> dict[str, dict]:
    """Every value key the profile accepts, with its type and the CLI flags it feeds."""
    entries: dict[str, dict] = {}

    def add(
        key: str,
        value_type: str = "string",
        flags: tuple[str, ...] = (),
        *,
        required: bool = False,
        options: tuple[str, ...] = (),
        when: When | None = None,
        workspace_path: bool = False,
    ) -> None:
        entry = entries.setdefault(key, {"type": value_type, "flags": []})
        for flag in flags:
            if flag not in entry["flags"]:
                entry["flags"].append(flag)
        if required:
            entry["required"] = True
        if options:
            entry["options"] = list(options)
        if when is not None:
            entry["when"] = {
                "key": when.key,
                "value": list(when.value)
                if isinstance(when.value, tuple)
                else when.value,
            }
        if workspace_path:
            entry["workspace_path"] = True

    add("outputName", flags=("--output_name",), required=True)
    add("outputDir", flags=("--output_dir",), required=True)
    add("musubiPath", required=True)
    if profile.task_options:
        add("task", flags=("--task",), required=True, options=profile.task_options)
    for selector in profile.selectors:
        add(
            selector.key,
            flags=(selector.train_flag,) if selector.train_flag else (),
            required=True,
            options=selector.options,
        )
    for field in (*profile.model_fields, *profile.advanced_fields):
        add(
            field.key,
            flags=(field.train_flag,) if field.train_flag else (),
            required=field.required,
            when=field.when,
            workspace_path=field.workspace_path,
        )
    for option in (
        *profile.cache_commands[0].options,
        *profile.cache_commands[1].options,
        *profile.train_options,
    ):
        if option.enabled_key is not None:
            add(option.enabled_key, "boolean", (option.flag,))
        elif option.key is not None and option.key != "datasetConfig":
            add(option.key, flags=(option.flag,))
    for key, flag in TRAIN_VALUE_FLAGS:
        add(key, flags=(flag,))
    add("savePrecision", options=SAVE_PRECISIONS)
    add("networkArgs", flags=("--network_args",))
    add("optimizerArgs", flags=("--optimizer_args",))
    for key, flag in BOOLEAN_TRAIN_FLAGS:
        add(key, "boolean", (flag,))
    for memory_flag in profile.memory_flags:
        add(memory_flag.key, "boolean", (memory_flag.flag,))
    add(
        "attention",
        flags=tuple(f"--{name}" for name in profile.attention_options),
        options=profile.attention_options,
    )
    for key, flag in HUGGINGFACE_VALUE_FLAGS:
        add(key, flags=(flag,))
    add("extraArgs")
    for key, flag in ACCELERATE_VALUE_FLAGS:
        add(key, flags=(flag,))
    return entries


def validate_values(profile: TrainingProfile, values: dict) -> list[str]:
    """Check the values for the profile and return warnings for ignored keys.

    Raises ValuesError / ExtraArgsError.
    """
    missing = [
        field.key
        for field in profile.model_fields
        if field.required
        and condition_matches(values, field.when)
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

    save_precision = _text(values, "savePrecision")
    if save_precision and save_precision not in SAVE_PRECISIONS:
        raise ValuesError(
            f"savePrecision must be one of: {', '.join(SAVE_PRECISIONS)}"
        )

    parse_extra_args(values)

    described = describe_values(profile)
    for key, value in values.items():
        entry = described.get(key)
        if entry and entry["type"] == "boolean" and not isinstance(value, bool):
            raise ValuesError(f"{key} must be a JSON boolean (true or false)")
    return [
        f"Unknown value '{key}' is ignored by profile {profile.id}"
        for key in values
        if key not in described and key not in SERVER_VALUE_KEYS
    ]


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


def _option_argv(
    option: CacheOption, values: dict, dataset_config_path: Path
) -> list[str]:
    if not condition_matches(values, option.when):
        return []
    if option.enabled_key is not None:
        return [option.flag] if _enabled(values, option.enabled_key) else []
    if option.key is not None:
        if option.key == "datasetConfig":
            return [option.flag, str(dataset_config_path)]
        return _argument(values, option.flag, option.key)
    if option.value is not None:
        return [option.flag, option.value]
    return [option.flag]


def _cache_argv(
    command: CacheCommand,
    values: dict,
    musubi_path: Path,
    dataset_config_path: Path,
) -> list[str]:
    argv = ["python", str(musubi_path / command.script)]
    for option in command.options:
        argv += _option_argv(option, values, dataset_config_path)
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
        if field.train_flag and condition_matches(values, field.when):
            args += argument(field.train_flag, field.key)
    for option in profile.train_options:
        args += _option_argv(option, values, dataset_config_path)
    args += ["--network_module", profile.network_module]
    for key, flag in TRAIN_VALUE_FLAGS:
        args += argument(flag, key)
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
