"""Server configuration resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).resolve() if raw else default.resolve()


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    # Database, job logs, and dataset snapshots live under this directory.
    data_root: Path = field(
        default_factory=lambda: _env_path("MUSUBI_GUI_DATA_ROOT", BACKEND_ROOT / "data")
    )
    # Allowlisted download scripts are looked up here by fixed filename.
    scripts_dir: Path = field(
        default_factory=lambda: _env_path(
            "MUSUBI_GUI_SCRIPTS_DIR", BACKEND_ROOT / "scripts"
        )
    )
    # Download destinations, musubiPath, and output/logging dirs must resolve inside this root.
    workspace_root: Path = field(
        default_factory=lambda: _env_path("MUSUBI_GUI_WORKSPACE_ROOT", Path.cwd())
    )
    # Built Vite frontend served in production when this directory exists.
    web_dir: Path = field(
        default_factory=lambda: _env_path("MUSUBI_GUI_WEB_DIR", BACKEND_ROOT / "web")
    )
    managed_max_files: int = field(
        default_factory=lambda: _env_positive_int("MUSUBI_GUI_MANAGED_MAX_FILES", 1000)
    )
    managed_max_file_bytes: int = field(
        default_factory=lambda: _env_positive_int(
            "MUSUBI_GUI_MANAGED_MAX_FILE_BYTES", 100 * 1024**3
        )
    )
    managed_max_total_bytes: int = field(
        default_factory=lambda: _env_positive_int(
            "MUSUBI_GUI_MANAGED_MAX_TOTAL_BYTES", 1024 * 1024**3
        )
    )
    managed_max_request_bytes: int = field(
        default_factory=lambda: _env_positive_int(
            "MUSUBI_GUI_MANAGED_MAX_REQUEST_BYTES", 1100 * 1024**3
        )
    )
    managed_max_storage_bytes: int = field(
        default_factory=lambda: _env_positive_int(
            "MUSUBI_GUI_MANAGED_MAX_STORAGE_BYTES", 10 * 1024**4
        )
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip()
            for origin in os.environ.get(
                "MUSUBI_GUI_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
    )

    @property
    def database_path(self) -> Path:
        return self.data_root / "musubi_gui.db"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def managed_datasets_dir(self) -> Path:
        return self.data_root / "managed_datasets"

    @property
    def managed_uploads_dir(self) -> Path:
        return self.data_root / "managed_uploads"

    def resolve_inside_workspace(self, raw: str) -> Path:
        """Resolve a client-supplied path against the workspace root.

        Raises ValueError when the result escapes the configured root.
        """
        candidate = Path(raw)
        resolved = (
            candidate if candidate.is_absolute() else self.workspace_root / candidate
        ).resolve()
        if (
            resolved != self.workspace_root
            and self.workspace_root not in resolved.parents
        ):
            raise ValueError(f"Path is outside the configured workspace root: {raw}")
        return resolved


def load_settings() -> Settings:
    return Settings()
