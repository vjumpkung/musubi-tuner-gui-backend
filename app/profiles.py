"""Mirror of the frontend training profile definitions.

Ported from musubi-tuner-gui-frontend/src/components/training/profiles.ts and the
hardcoded cache-command builders in training_workspace.tsx. Only the data the
backend needs to validate requests and build argv is mirrored; labels, helper
text, and form defaults stay in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class When:
    key: str
    value: str | bool


@dataclass(frozen=True)
class ProfileField:
    key: str
    train_flag: str | None = None
    required: bool = False
    when: When | None = None


@dataclass(frozen=True)
class Selector:
    key: str
    train_flag: str | None
    options: tuple[str, ...]


@dataclass(frozen=True)
class CacheOption:
    flag: str
    key: str | None = None
    value: str | None = None
    enabled_key: str | None = None
    when: When | None = None


@dataclass(frozen=True)
class CacheCommand:
    script: str
    options: tuple[CacheOption, ...]


@dataclass(frozen=True)
class MemoryFlag:
    key: str
    flag: str
    train: bool = True


@dataclass(frozen=True)
class TrainingProfile:
    id: str
    name: str
    trainer: str
    network_module: str
    cache_commands: tuple[CacheCommand, CacheCommand]
    model_fields: tuple[ProfileField, ...]
    attention_options: tuple[str, ...]
    advanced_fields: tuple[ProfileField, ...] = ()
    selectors: tuple[Selector, ...] = ()
    memory_flags: tuple[MemoryFlag, ...] = ()
    fixed_train_flags: tuple[str, ...] = ()
    cpu_threads: int = 2
    task_options: tuple[str, ...] = ()


TRAINING_PROFILES: dict[str, TrainingProfile] = {
    "hunyuan-video": TrainingProfile(
        id="hunyuan-video",
        name="Hunyuan Video",
        trainer="hv_train_network.py",
        network_module="networks.lora",
        cache_commands=(
            CacheCommand(
                script="cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--vae_chunk_size", key="vaeChunkSize"),
                    CacheOption(flag="--vae_tiling", enabled_key="vaeTiling"),
                ),
            ),
            CacheCommand(
                script="cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder1", key="textEncoder1"),
                    CacheOption(flag="--text_encoder2", key="textEncoder2"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                    CacheOption(flag="--fp8_llm", enabled_key="fp8Llm"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", required=True),
            ProfileField(key="textEncoder1", required=True),
            ProfileField(key="textEncoder2", required=True),
        ),
        attention_options=("sdpa", "flash_attn", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Llm", flag="--fp8_llm", train=False),
            MemoryFlag(key="vaeTiling", flag="--vae_tiling", train=False),
        ),
    ),
    "framepack": TrainingProfile(
        id="framepack",
        name="FramePack",
        trainer="fpack_train_network.py",
        network_module="networks.lora_framepack",
        cache_commands=(
            CacheCommand(
                script="fpack_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--image_encoder", key="imageEncoder"),
                    CacheOption(flag="--vae_chunk_size", key="vaeChunkSize"),
                ),
            ),
            CacheCommand(
                script="fpack_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder1", key="textEncoder1"),
                    CacheOption(flag="--text_encoder2", key="textEncoder2"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                    CacheOption(flag="--fp8_llm", enabled_key="fp8Llm"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(
                key="textEncoder1", train_flag="--text_encoder1", required=True
            ),
            ProfileField(
                key="textEncoder2", train_flag="--text_encoder2", required=True
            ),
            ProfileField(
                key="imageEncoder", train_flag="--image_encoder", required=True
            ),
        ),
        attention_options=("sdpa", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="fp8Llm", flag="--fp8_llm", train=False),
        ),
    ),
    "wan-22": TrainingProfile(
        id="wan-22",
        name="WAN 2.2",
        trainer="wan_train_network.py",
        network_module="networks.lora_wan",
        cache_commands=(
            CacheCommand(
                script="wan_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                ),
            ),
            CacheCommand(
                script="wan_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--t5", key="t5"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="ditHigh", train_flag="--dit_high_noise", required=True),
            ProfileField(key="vae", required=True),
            ProfileField(key="t5", required=True),
        ),
        attention_options=("sdpa", "xformers"),
        memory_flags=(MemoryFlag(key="fp8Base", flag="--fp8_base"),),
        cpu_threads=16,
        task_options=(
            "t2v-A14B",
            "i2v-A14B",
            "t2v-1.3B",
            "t2v-14B",
            "i2v-14B",
            "t2i-14B",
        ),
    ),
    "flux-kontext": TrainingProfile(
        id="flux-kontext",
        name="Flux Kontext Dev",
        trainer="flux_kontext_train_network.py",
        network_module="networks.lora_flux",
        cache_commands=(
            CacheCommand(
                script="flux_kontext_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                ),
            ),
            CacheCommand(
                script="flux_kontext_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder1", key="textEncoder1"),
                    CacheOption(flag="--text_encoder2", key="textEncoder2"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(
                key="textEncoder1", train_flag="--text_encoder1", required=True
            ),
            ProfileField(
                key="textEncoder2", train_flag="--text_encoder2", required=True
            ),
        ),
        attention_options=("sdpa", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="fp8T5", flag="--fp8_t5"),
        ),
    ),
    "qwen-image": TrainingProfile(
        id="qwen-image",
        name="QWEN Image",
        trainer="qwen_image_train_network.py",
        network_module="networks.lora_qwen_image",
        cache_commands=(
            CacheCommand(
                script="qwen_image_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                ),
            ),
            CacheCommand(
                script="qwen_image_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder", key="textEncoder"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", required=True),
            ProfileField(key="textEncoder", required=True),
        ),
        attention_options=("xformers", "sdpa"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="textEncoderCpu", flag="--text_encoder_cpu"),
            MemoryFlag(key="vaeEnableTiling", flag="--vae_enable_tiling"),
        ),
    ),
    "flux-2": TrainingProfile(
        id="flux-2",
        name="FLUX.2",
        trainer="flux_2_train_network.py",
        network_module="networks.lora_flux_2",
        cache_commands=(
            CacheCommand(
                script="flux_2_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--model_version", key="modelVersion"),
                ),
            ),
            CacheCommand(
                script="flux_2_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder", key="textEncoder"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                    CacheOption(flag="--model_version", key="modelVersion"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(key="textEncoder", train_flag="--text_encoder", required=True),
        ),
        attention_options=("sdpa", "xformers", "flash_attn"),
        selectors=(
            Selector(
                key="modelVersion",
                train_flag="--model_version",
                options=(
                    "dev",
                    "klein-4b",
                    "klein-base-4b",
                    "klein-9b",
                    "klein-base-9b",
                ),
            ),
        ),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="fp8TextEncoder", flag="--fp8_text_encoder"),
        ),
    ),
    "framepack-one-frame": TrainingProfile(
        id="framepack-one-frame",
        name="FramePack One-Frame",
        trainer="fpack_train_network.py",
        network_module="networks.lora_framepack",
        cache_commands=(
            CacheCommand(
                script="fpack_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--image_encoder", key="imageEncoder"),
                    CacheOption(flag="--vae_chunk_size", key="vaeChunkSize"),
                    CacheOption(flag="--one_frame"),
                    CacheOption(flag="--one_frame_no_2x", enabled_key="oneFrameNo2x"),
                    CacheOption(flag="--one_frame_no_4x", enabled_key="oneFrameNo4x"),
                ),
            ),
            CacheCommand(
                script="fpack_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder1", key="textEncoder1"),
                    CacheOption(flag="--text_encoder2", key="textEncoder2"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                    CacheOption(flag="--fp8_llm", enabled_key="fp8Llm"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(
                key="textEncoder1", train_flag="--text_encoder1", required=True
            ),
            ProfileField(
                key="textEncoder2", train_flag="--text_encoder2", required=True
            ),
            ProfileField(
                key="imageEncoder", train_flag="--image_encoder", required=True
            ),
        ),
        attention_options=("sdpa", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="fp8Llm", flag="--fp8_llm", train=False),
            MemoryFlag(key="oneFrameNo2x", flag="--one_frame_no_2x", train=False),
            MemoryFlag(key="oneFrameNo4x", flag="--one_frame_no_4x", train=False),
        ),
        fixed_train_flags=("--one_frame",),
    ),
    "wan-one-frame": TrainingProfile(
        id="wan-one-frame",
        name="WAN One-Frame",
        trainer="wan_train_network.py",
        network_module="networks.lora_wan",
        cache_commands=(
            CacheCommand(
                script="wan_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--one_frame"),
                ),
            ),
            CacheCommand(
                script="wan_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--t5", key="t5"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", required=True),
            ProfileField(key="t5", required=True),
        ),
        attention_options=("sdpa", "xformers"),
        memory_flags=(MemoryFlag(key="fp8Base", flag="--fp8_base"),),
        fixed_train_flags=("--one_frame",),
        task_options=("i2v-14B", "flf2v-14B"),
    ),
    "z-image-turbo": TrainingProfile(
        id="z-image-turbo",
        name="Z-Image Turbo",
        trainer="zimage_train_network.py",
        network_module="networks.lora_zimage",
        cache_commands=(
            CacheCommand(
                script="zimage_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                ),
            ),
            CacheCommand(
                script="zimage_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder", key="textEncoder"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", required=True),
            ProfileField(key="textEncoder", required=True),
        ),
        attention_options=("sdpa", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="textEncoderCpu", flag="--text_encoder_cpu"),
            MemoryFlag(key="vaeEnableTiling", flag="--vae_enable_tiling"),
        ),
    ),
    "hidream-o1": TrainingProfile(
        id="hidream-o1",
        name="HiDream O1",
        trainer="hidream_o1_train_network.py",
        network_module="networks.lora_hidream_o1",
        cache_commands=(
            CacheCommand(
                script="hidream_o1_cache_pixel.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--batch_size", key="cachePixelBatchSize"),
                ),
            ),
            CacheCommand(
                script="hidream_o1_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                    CacheOption(flag="--model_type", key="modelType"),
                ),
            ),
        ),
        model_fields=(ProfileField(key="dit", train_flag="--dit", required=True),),
        attention_options=("sdpa", "flash_attn"),
        advanced_fields=(
            ProfileField(key="noiseScaleStart", train_flag="--noise_scale_start"),
            ProfileField(key="noiseScaleEnd", train_flag="--noise_scale_end"),
            ProfileField(key="noiseClipStd", train_flag="--noise_clip_std"),
        ),
        selectors=(
            Selector(
                key="modelType", train_flag="--model_type", options=("full", "dev")
            ),
        ),
        memory_flags=(
            MemoryFlag(
                key="pinnedBlockSwap", flag="--use_pinned_memory_for_block_swap"
            ),
        ),
        task_options=("t2i", "i2i"),
    ),
    "hunyuan-video-1-5": TrainingProfile(
        id="hunyuan-video-1-5",
        name="HunyuanVideo 1.5",
        trainer="hv_1_5_train_network.py",
        network_module="networks.lora_hv_1_5",
        cache_commands=(
            CacheCommand(
                script="hv_1_5_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--i2v", when=When(key="task", value="i2v")),
                    CacheOption(
                        flag="--image_encoder",
                        key="imageEncoder",
                        when=When(key="task", value="i2v"),
                    ),
                ),
            ),
            CacheCommand(
                script="hv_1_5_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder", key="textEncoder"),
                    CacheOption(flag="--byt5", key="byt5"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                    CacheOption(flag="--fp8_vl", enabled_key="fp8Vl"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(key="textEncoder", train_flag="--text_encoder", required=True),
            ProfileField(key="byt5", train_flag="--byt5", required=True),
            ProfileField(
                key="imageEncoder",
                train_flag="--image_encoder",
                required=True,
                when=When(key="task", value="i2v"),
            ),
        ),
        attention_options=("sdpa", "flash_attn", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
            MemoryFlag(key="fp8Vl", flag="--fp8_vl", train=False),
            MemoryFlag(key="splitAttn", flag="--split_attn"),
        ),
        task_options=("t2v", "i2v"),
    ),
    "ideogram-4": TrainingProfile(
        id="ideogram-4",
        name="Ideogram 4",
        trainer="ideogram4_train_network.py",
        network_module="networks.lora_ideogram4",
        cache_commands=(
            CacheCommand(
                script="ideogram4_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                    CacheOption(flag="--vae_dtype", value="bfloat16"),
                ),
            ),
            CacheCommand(
                script="ideogram4_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder", key="textEncoder"),
                    CacheOption(flag="--text_cache_dtype", value="bf16"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", required=True),
            ProfileField(key="textEncoder", required=True),
        ),
        attention_options=("sdpa", "flash_attn", "xformers"),
    ),
    "kandinsky-5": TrainingProfile(
        id="kandinsky-5",
        name="Kandinsky 5",
        trainer="kandinsky5_train_network.py",
        network_module="networks.lora_kandinsky",
        cache_commands=(
            CacheCommand(
                script="kandinsky5_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                ),
            ),
            CacheCommand(
                script="kandinsky5_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder_qwen", key="textEncoderQwen"),
                    CacheOption(flag="--text_encoder_clip", key="textEncoderClip"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(
                key="textEncoderQwen", train_flag="--text_encoder_qwen", required=True
            ),
            ProfileField(
                key="textEncoderClip", train_flag="--text_encoder_clip", required=True
            ),
        ),
        attention_options=("sdpa", "flash_attn", "sage_attn", "xformers"),
        advanced_fields=(
            ProfileField(key="maxGradNorm", train_flag="--max_grad_norm"),
            ProfileField(key="schedulerScale", train_flag="--scheduler_scale"),
        ),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
        ),
        task_options=("k5-pro-t2v-5s-sd", "k5-pro-i2v-5s-sd"),
    ),
    "krea-2": TrainingProfile(
        id="krea-2",
        name="Krea 2",
        trainer="krea2_train_network.py",
        network_module="networks.lora_krea2",
        cache_commands=(
            CacheCommand(
                script="krea2_cache_latents.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--vae", key="vae"),
                ),
            ),
            CacheCommand(
                script="krea2_cache_text_encoder_outputs.py",
                options=(
                    CacheOption(flag="--dataset_config", key="datasetConfig"),
                    CacheOption(flag="--text_encoder", key="textEncoder"),
                    CacheOption(flag="--batch_size", key="cacheBatchSize"),
                ),
            ),
        ),
        model_fields=(
            ProfileField(key="dit", train_flag="--dit", required=True),
            ProfileField(key="vae", train_flag="--vae", required=True),
            ProfileField(key="textEncoder", required=True),
        ),
        attention_options=("sdpa", "flash_attn", "sage_attn", "xformers"),
        memory_flags=(
            MemoryFlag(key="fp8Base", flag="--fp8_base"),
            MemoryFlag(key="fp8Scaled", flag="--fp8_scaled"),
        ),
    ),
}
