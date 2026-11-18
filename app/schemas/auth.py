"""Authentication API request schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr


class LoginRequest(BaseModel):
    password: SecretStr = Field(min_length=1, max_length=1024)
