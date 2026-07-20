"""Download API request schemas."""

from pydantic import BaseModel


class DownloadRequest(BaseModel):
    script_id: str
    destination: str = "."
