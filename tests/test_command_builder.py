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
    "savePrecision": "bf16",
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
    assert train[train.index("--save_precision") + 1] == "bf16"
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
    with pytest.raises(ValuesError, match="savePrecision"):
        validate_values(profile, {**WAN22_VALUES, "savePrecision": "int8"})
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


def test_flux2_model_versions_match_download_bundles():
    profile = TRAINING_PROFILES["flux-2"]
    assert profile.selectors[0].options == (
        "dev",
        "klein-base-4b",
        "klein-base-9b",
    )

    values = {
        "musubiPath": "./musubi-tuner",
        "dit": "./diffusion_models/flux-2-klein-base-4b.safetensors",
        "vae": "./vae/ae.safetensors",
        "textEncoder": ("./text_encoders/qwen3_4b/model-00001-of-00002.safetensors"),
        "outputName": "flux2_lora",
        "outputDir": "./lora_training/outputs",
    }
    for model_version in profile.selectors[0].options:
        validate_values(profile, {**values, "modelVersion": model_version})

    with pytest.raises(ValuesError, match="modelVersion"):
        validate_values(profile, {**values, "modelVersion": "klein-4b"})


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


H3_VALUES = {
    "musubiPath": "./musubi-tuner",
    "task": "t2va",
    "lossMethod": "guidance",
    "dit": "./diffusion_models/minimax_h3_fl2va_bf16.safetensors",
    "videoVae": "./vae/minimax_h3_video_vae_fp16.safetensors",
    "audioVae": "./vae/minimax_h3_audio_vae_fp32.safetensors",
    "textEncoder": "./text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
    "baseWeights": "./training_adapter.safetensors",
    "guidanceUncondCache": "./uncond.safetensors",
    "h3GuidanceLossScale": "4.0",
    "h3GuidanceLossSigmaMin": "0.15",
    "h3GuidanceLossScaleAudio": "3.0",
    "h3TeacherConditionSigmaMin": "0.15",
    "h3TeacherConditionSigmaMax": "0.75",
    "h3TeacherLossDcWeight": "0.3",
    "h3TeacherLossMagWeight": "0.5",
    "h3TimestepFocusProb": "0.5",
    "audioLossWeight": "1.0",
    "cacheSeed": "42",
    "textEncoderBlocksToSwap": "0",
    "textEncoderAttnMode": "sdpa",
    "outputName": "h3_lora",
    "outputDir": "./outputs",
    "cacheBatchSize": "1",
    "attention": "sdpa",
    "extraArgs": "--example_arg value",
}


def build_h3(values):
    profile = TRAINING_PROFILES["minimax-h3"]
    validate_values(profile, values)
    return build_stage_argv(profile, values, Path("/ws/m"), Path("/snapshot.toml"))


@pytest.mark.parametrize(
    ("method", "task", "one_frame"),
    [
        (method, task, one_frame)
        for method in (
            "guidance",
            "adapter",
            "teacher-first-last",
            "teacher-ref",
            "teacher-subject-ref",
        )
        for task in ("t2va", "fl2va", "ref2va")
        for one_frame in (False, True)
        if not (one_frame and method in {"teacher-first-last", "teacher-ref"})
    ],
)
def test_h3_command_matrix(method, task, one_frame):
    values = {**H3_VALUES, "lossMethod": method, "task": task, "oneFrame": one_frame}
    stages = build_h3(values)
    latents, text, train = (
        stages["cache_latents"],
        stages["cache_text_encoder"],
        stages["train"],
    )
    teacher_conditions = {
        "teacher-first-last": ("fl2va", "first,last"),
        "teacher-ref": ("t2va", "ref"),
        "teacher-subject-ref": ("ref2va", "subject_ref"),
    }
    teacher = method in teacher_conditions
    expected_latent_task = teacher_conditions[method][0] if teacher else task
    for argv, expected_task in (
        (latents, expected_latent_task),
        (text, "t2va" if teacher else task),
        (train, "t2va" if teacher else task),
    ):
        assert argv.count("--task") == 1
        assert argv[argv.index("--task") + 1] == expected_task
        assert ("--one_frame" in argv) is one_frame
        assert argv[argv.index("--dataset_config") + 1] == str(Path("/snapshot.toml"))
    assert latents[1].endswith("minimax_h3_cache_latents.py")
    assert text[1].endswith("minimax_h3_cache_text_encoder_outputs.py")
    assert any(item.endswith("minimax_h3_train_network.py") for item in train)
    assert latents[latents.index("--video_vae") + 1] == values["videoVae"]
    assert latents[latents.index("--audio_vae") + 1] == values["audioVae"]
    assert latents[latents.index("--cache_seed") + 1] == "42"
    assert latents[-1] == text[-1] == "--skip_existing"
    assert text[text.index("--text_encoder") + 1] == values["textEncoder"]
    assert text[text.index("--batch_size") + 1] == "1"
    assert text[text.index("--text_encoder_blocks_to_swap") + 1] == "0"
    assert text[text.index("--text_encoder_attn_mode") + 1] == "sdpa"
    assert train[train.index("--network_module") + 1] == "networks.lora_minimax_h3"
    assert train[train.index("--num_cpu_threads_per_process") + 1] == "1"
    assert train[train.index("--audio_loss_weight") + 1] == "1.0"
    assert train.index("--task") < train.index("--example_arg")
    assert train[-2:] == ["--example_arg", "value"]
    for flag in (
        "--video_vae",
        "--audio_vae",
        "--text_encoder",
        "--cache_seed",
        "--fp8_base",
        "--fp8_scaled",
    ):
        assert flag not in train
    assert ("--base_weights" in train) is (method == "adapter")
    assert ("--uncond_output" in text) is (method == "guidance")
    guidance_flags = {
        "--h3_guidance_loss_uncond_cache": "./uncond.safetensors",
        "--h3_guidance_loss_scale": "4.0",
        "--h3_guidance_loss_sigma_min": "0.15",
        "--h3_guidance_loss_scale_audio": "3.0",
    }
    for flag, value in guidance_flags.items():
        assert (flag in train) is (method == "guidance")
        if method == "guidance":
            assert train[train.index(flag) + 1] == value
    if method == "guidance":
        assert text[text.index("--uncond_output") + 1] == "./uncond.safetensors"
    if method == "adapter":
        assert train[train.index("--base_weights") + 1] == values["baseWeights"]
    assert ("--h3_teacher_matching" in train) is teacher
    assert ("--h3_teacher_conditions" in train) is teacher
    assert ("--teacher_conditions" in text) is teacher
    if teacher:
        condition = teacher_conditions[method][1]
        assert train[train.index("--h3_teacher_conditions") + 1] == condition
        assert text[text.index("--teacher_conditions") + 1] == condition
    for flag, value in {
        "--h3_teacher_condition_sigma_min": "0.15",
        "--h3_teacher_condition_sigma_max": "0.75",
        "--h3_teacher_loss_dc_weight": "0.3",
        "--h3_teacher_loss_mag_weight": "0.5",
        "--h3_timestep_focus_prob": "0.5",
    }.items():
        assert (flag in train) is teacher
        if teacher:
            assert train[train.index(flag) + 1] == value


def test_h3_selector_whitespace_renders_identically():
    canonical = {**H3_VALUES, "lossMethod": "teacher-ref"}
    spaced = {**H3_VALUES, "lossMethod": "teacher-ref ", "task": " t2va "}
    assert build_h3(spaced) == build_h3(canonical)
    assert "--h3_teacher_matching" in build_h3(spaced)["train"]


def test_h3_managed_task_precedes_extra_args():
    values = {**H3_VALUES, "lossMethod": "teacher-ref", "extraArgs": "--task ref2va"}
    train = build_h3(values)["train"]
    assert train[-2:] == ["--task", "ref2va"]
    managed = train[:-2]
    assert managed.count("--task") == 1
    assert managed[managed.index("--task") + 1] == "t2va"


@pytest.mark.parametrize(
    ("updates", "key"),
    [
        ({"lossMethod": "adapter", "baseWeights": ""}, "baseWeights"),
        ({"guidanceUncondCache": ""}, "guidanceUncondCache"),
        ({"lossMethod": "invalid"}, "lossMethod"),
        ({"task": "invalid"}, "task"),
        ({"videoVae": ""}, "videoVae"),
        ({"audioVae": ""}, "audioVae"),
        ({"textEncoder": ""}, "textEncoder"),
    ],
)
def test_h3_required_values_and_selectors(updates, key):
    with pytest.raises(ValuesError, match=key):
        validate_values(TRAINING_PROFILES["minimax-h3"], {**H3_VALUES, **updates})


def test_h3_memory_flags():
    values = {
        **H3_VALUES,
        "videoOnly": True,
        "convrotInt8": True,
        "pruneAdaln": True,
        "splitAttn": True,
    }
    train = build_h3(values)["train"]
    for flag in ("--video_only", "--convrot_int8", "--prune_adaln", "--split_attn"):
        assert flag in train
