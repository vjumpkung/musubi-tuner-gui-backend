"""Dataset API request schemas."""

from pydantic import BaseModel, Field


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
