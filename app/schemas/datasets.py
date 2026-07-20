"""Dataset API request schemas."""

from pydantic import BaseModel, Field


class DatasetPayload(BaseModel):
    name: str
    description: str | None = None
    general: dict = Field(default_factory=dict)
    datasets: list
