"""Dataset API request schemas."""

from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class DatasetPayload(BaseModel):
    name: str
    description: str | None = None
    general: dict = Field(default_factory=dict)
    datasets: list


class ManagedFinalizePayload(BaseModel):
    name: str
    description: str | None = None
    batch_size: int = Field(default=1, gt=0)
    enable_bucket: bool = True
    bucket_no_upscale: bool = False
    dataset_specs: list[dict]
    file_tokens: list[str] = Field(default_factory=list)
    caption_file_tokens: list[str] = Field(default_factory=list)
    control_file_tokens: list[str] = Field(default_factory=list)


class DatasetOptions(BaseModel):
    name: str
    description: str | None = None
    media_type: Literal["image", "video"] = "image"
    resolution: list[PositiveInt] = Field(min_length=2, max_length=2)
    num_repeats: PositiveInt = 1
    target_frames: list[PositiveInt] | None = Field(None, min_length=1)
    batch_size: PositiveInt = 1
    enable_bucket: bool = True
    bucket_no_upscale: bool = False


class DatasetFromFolderPayload(DatasetOptions):
    path: str = Field(min_length=1)


class ManagedSimpleFinalizePayload(DatasetOptions):
    pass
