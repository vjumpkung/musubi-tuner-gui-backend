"""Async SQLite persistence for dataset configs and training jobs."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_configs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT,
    config_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_jobs (
    id                    TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    profile_id            TEXT NOT NULL,
    dataset_config_id     TEXT REFERENCES dataset_configs(id) ON DELETE SET NULL,
    dataset_config_toml   TEXT NOT NULL,
    values_json           TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'queued',
    queue_position        INTEGER,
    current_stage         TEXT,
    stages_json           TEXT NOT NULL,
    progress_json         TEXT,
    error                 TEXT,
    log_path              TEXT,
    created_at            TEXT NOT NULL,
    started_at            TEXT,
    finished_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs (status, created_at);

CREATE TABLE IF NOT EXISTS app_settings (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL
);
"""


def utc_now() -> str:
    """ISO-8601 UTC timestamp with microseconds, e.g. 2026-07-12T10:00:00.123456Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Database:
    """A single shared aiosqlite connection with a lock for multi-statement writes."""

    def __init__(self, connection: aiosqlite.Connection):
        self.connection = connection
        self.write_lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: Path) -> "Database":
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode = WAL")
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute("PRAGMA busy_timeout = 5000")
        await connection.executescript(SCHEMA)
        await connection.commit()
        return cls(connection)

    async def close(self) -> None:
        await self.connection.close()

    async def fetch_one(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        async with self.connection.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        async with self.connection.execute(sql, params) as cursor:
            return list(await cursor.fetchall())

    async def execute(self, sql: str, params: tuple = ()) -> None:
        await self.connection.execute(sql, params)
        await self.connection.commit()

    async def get_setting(self, key: str, default: str) -> str:
        row = await self.fetch_one(
            "SELECT value FROM app_settings WHERE key = ?", (key,)
        )
        return row["value"] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        await self.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
