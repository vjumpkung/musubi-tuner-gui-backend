"""Download API request schemas."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class DownloadRequest(BaseModel):
    script_id: str
    destination: str = "."
    hf_token: SecretStr | None = Field(default=None, min_length=1, max_length=2048)


class DownloadFile(BaseModel):
    url: str
    path: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "huggingface.co"
            or parsed.netloc.lower() != "huggingface.co"
            or "?" in value
            or "#" in value
        ):
            raise ValueError("Download URL must use HTTPS on huggingface.co")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if (
            not value
            or any(character in value for character in ("\\", ":", "\0"))
            or any(segment in {"", ".", ".."} for segment in value.split("/"))
        ):
            raise ValueError("Download path must be a relative POSIX path")
        return value


class DownloadModel(BaseModel):
    id: str
    name: str
    files: list[DownloadFile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_paths(self) -> DownloadModel:
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Download paths must be unique within a model")
        return self
