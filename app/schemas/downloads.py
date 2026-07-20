"""Download API request schemas."""

from pydantic import BaseModel, Field, SecretStr


class DownloadRequest(BaseModel):
    script_id: str
    destination: str = "."
    hf_token: SecretStr | None = Field(default=None, min_length=1, max_length=2048)
