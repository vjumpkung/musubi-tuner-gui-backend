from pathlib import Path

import pytest

from app.command_builder import (
    ExtraArgsError,
    ValuesError,
    build_stage_argv,
    parse_extra_args,
    redact_argv,
    validate_values,
)
from app.profiles import TRAINING_PROFILES

WAN22_VALUES = {
    "musubiPath": "./musubi-tuner",
    "task": "t2v-A14B",
    "dit": "./diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
    "ditHigh": "./diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors",
    "vae": "./vae/wan_2.1_vae.safetensors",
    "t5": "./text_encoders/models_t5_umt5-xxl-enc-bf16.pth",
    "outputName": "char_v3",
    "outputDir": "./lora_training/outputs",
    "epochs": "1000",
    "learningRate": "1e-4",
    "optimizer": "adamw",
    "mixedPrecision": "fp16",
    "attention": "sdpa",
    "cacheBatchSize": "16",
    "fp8Base": False,
    "gradientCheckpointing": True,
    "extraArgs": "--preserve_distribution_shape --log_with tensorboard",
}


def build_wan22(values=None):
    profile = TRAINING_PROFILES["wan-22"]
    return build_stage_argv(
        profile,
        values or WAN22_VALUES,
        musubi_path=Path("/ws/musubi-tuner"),
        dataset_config_path=Path("/data/jobs/j1/dataset_config.toml"),
    )


def test_wan22_stage_argv():
    stages = build_wan22()

    latents = stages["cache_latents"]
    assert latents[0] == "python"
    assert latents[1].endswith("wan_cache_latents.py")
    assert latents[2:4] == [
        "--dataset_config",
        str(Path("/data/jobs/j1/dataset_config.toml")),
    ]
    assert "--vae" in latents

    text_encoder = stages["cache_text_encoder"]
    assert text_encoder[1].endswith("wan_cache_text_encoder_outputs.py")
    assert text_encoder[text_encoder.index("--t5") + 1].endswith(
        "umt5-xxl-enc-bf16.pth"
    )
    assert text_encoder[text_encoder.index("--batch_size") + 1] == "16"

    train = stages["train"]
    assert train[:2] == ["accelerate", "launch"]
    # wan-22 defaults to 16 CPU threads per process.
    assert train[train.index("--num_cpu_threads_per_process") + 1] == "16"
    assert train[train.index("--task") + 1] == "t2v-A14B"
    assert train[train.index("--dit_high_noise") + 1].endswith(
        "high_noise_14B_fp16.safetensors"
    )
    assert train[train.index("--network_module") + 1] == "networks.lora_wan"
    # The snapshot path is forced; the client cannot point the trainer elsewhere.
    assert train[train.index("--dataset_config") + 1] == str(
        Path("/data/jobs/j1/dataset_config.toml")
    )
    assert "--gradient_checkpointing" in train
    assert "--fp8_base" not in train
    assert "--sdpa" in train
    assert (
        train[-2:] == ["--preserve_distribution_shape", "--log_with"]
        or "tensorboard" in train
    )


def test_conditional_fields_follow_task():
    profile = TRAINING_PROFILES["hunyuan-video-1-5"]
    values = {
        "musubiPath": "./musubi-tuner",
        "task": "t2v",
        "dit": "./d.safetensors",
        "vae": "./v.safetensors",
        "textEncoder": "./te.safetensors",
        "byt5": "./byt5.safetensors",
        "outputName": "x",
        "outputDir": "./out",
    }
    stages = build_stage_argv(profile, values, Path("/ws/m"), Path("/snap.toml"))
    assert "--i2v" not in stages["cache_latents"]
    assert "--image_encoder" not in stages["train"]

    values_i2v = {**values, "task": "i2v", "imageEncoder": "./ie.safetensors"}
    stages = build_stage_argv(profile, values_i2v, Path("/ws/m"), Path("/snap.toml"))
    assert "--i2v" in stages["cache_latents"]
    assert "--image_encoder" in stages["train"]


def test_validate_values_requires_profile_fields():
    profile = TRAINING_PROFILES["wan-22"]
    with pytest.raises(ValuesError, match="ditHigh"):
        validate_values(profile, {**WAN22_VALUES, "ditHigh": ""})
    with pytest.raises(ValuesError, match="task"):
        validate_values(profile, {**WAN22_VALUES, "task": "bogus"})
    with pytest.raises(ValuesError, match="attention"):
        validate_values(profile, {**WAN22_VALUES, "attention": "rm -rf"})
    # i2v-only field is not required for t2v.
    hv = TRAINING_PROFILES["hunyuan-video-1-5"]
    validate_values(
        hv,
        {
            "musubiPath": "m",
            "task": "t2v",
            "dit": "d",
            "vae": "v",
            "textEncoder": "t",
            "byt5": "b",
            "outputName": "o",
            "outputDir": "od",
        },
    )


def test_extra_args_tokens_are_validated():
    assert parse_extra_args(
        {"extraArgs": "--log_with tensorboard --flow_shift=3.0"}
    ) == [
        "--log_with",
        "tensorboard",
        "--flow_shift=3.0",
    ]
    with pytest.raises(ExtraArgsError):
        parse_extra_args({"extraArgs": "--ok; rm -rf /"})
    with pytest.raises(ExtraArgsError):
        parse_extra_args({"extraArgs": "$(whoami)"})


def test_huggingface_token_redaction():
    values = {
        **WAN22_VALUES,
        "huggingfaceRepoId": "me/repo",
        "huggingfaceToken": "hf_secret123",
    }
    train = build_wan22(values)["train"]
    assert "hf_secret123" in train  # passed to the real process
    redacted = redact_argv(train)
    assert "hf_secret123" not in redacted
    assert redacted[redacted.index("--huggingface_token") + 1] == "***"


def test_fixed_train_flags_and_fixed_cache_values():
    profile = TRAINING_PROFILES["wan-one-frame"]
    values = {
        "musubiPath": "m",
        "task": "i2v-14B",
        "dit": "d",
        "vae": "v",
        "t5": "t",
        "outputName": "o",
        "outputDir": "od",
    }
    stages = build_stage_argv(profile, values, Path("/ws/m"), Path("/snap.toml"))
    assert "--one_frame" in stages["cache_latents"]
    assert "--one_frame" in stages["train"]

    ideogram = TRAINING_PROFILES["ideogram-4"]
    values = {
        "musubiPath": "m",
        "dit": "d",
        "vae": "v",
        "textEncoder": "t",
        "outputName": "o",
        "outputDir": "od",
    }
    stages = build_stage_argv(ideogram, values, Path("/ws/m"), Path("/snap.toml"))
    latents = stages["cache_latents"]
    assert latents[latents.index("--vae_dtype") + 1] == "bfloat16"
