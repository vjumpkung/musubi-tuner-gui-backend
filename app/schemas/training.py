"""Training API request schemas."""

from typing import Any

from pydantic import BaseModel, Field


class TrainingJobRequest(BaseModel):
    name: str
    profile_id: str
    dataset_config_id: str
    skip_cache_stages: bool = False
    values: dict[str, Any] = Field(default_factory=dict)


class ReorderRequest(BaseModel):
    queue_position: int
