"""The training queue runner: run one queued job per explicit start request."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from collections import deque
from pathlib import Path
from typing import AsyncIterator

from .command_builder import build_stage_argv, redact_argv
from .config import Settings
from .db import Database, utc_now
from .process import spawn, terminate_tree
from .profiles import TRAINING_PROFILES

QUEUE_STATE_KEY = "training_queue_state"
MAX_LOG_BYTES = 50 * 1024 * 1024
PROGRESS_UPDATE_INTERVAL = 1.0

LINE_SPLIT = re.compile(rb"\r\n|\r|\n")
STEP_PATTERN = re.compile(
    r"(\d+)/(\d+)\s*\["
)  # tqdm: "steps: 35%|... | 350/1000 [00:10<...]"
FALLBACK_STEP_PATTERN = re.compile(r"steps?[:\s].*?(\d+)/(\d+)", re.IGNORECASE)
EPOCH_PATTERN = re.compile(r"epoch[:\s]+(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE)


async def _iter_lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    """Yield lines split on \\n or \\r so tqdm carriage-return updates stream too."""
    buffer = b""
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            if buffer:
                yield buffer.decode("utf-8", errors="replace")
            return
        buffer += chunk
        parts = LINE_SPLIT.split(buffer)
        buffer = parts.pop()
        for part in parts:
            if part:
                yield part.decode("utf-8", errors="replace")
        if len(buffer) > 1024 * 1024:
            yield buffer.decode("utf-8", errors="replace")
            buffer = b""


def parse_progress(line: str, progress: dict) -> bool:
    """Best-effort progress from trainer output. Returns True when anything changed."""
    changed = False
    match = STEP_PATTERN.search(line) or FALLBACK_STEP_PATTERN.search(line)
    if match:
        step, total = int(match.group(1)), int(match.group(2))
        if total > 0 and step <= total:
            changed = (
                progress.get("step") != step or progress.get("total_steps") != total
            )
            progress["step"] = step
            progress["total_steps"] = total
            progress["percent"] = round(step / total * 100, 1)
    epoch_match = EPOCH_PATTERN.search(line)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        if progress.get("epoch") != epoch:
            progress["epoch"] = epoch
            changed = True
        if epoch_match.group(2):
            total_epochs = int(epoch_match.group(2))
            if progress.get("total_epochs") != total_epochs:
                progress["total_epochs"] = total_epochs
                changed = True
    return changed


class QueueRunner:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.queue_running = asyncio.Event()
        self.wake = asyncio.Event()
        # Serializes pause/start, claiming, cancellation, and reorder decisions
        # so endpoints cannot act on a stale queued state.
        self.claim_lock = asyncio.Lock()
        self.current_job_id: str | None = None
        self.current_process: asyncio.subprocess.Process | None = None
        self.cancel_requested = False

    async def restore_state(self) -> None:
        state = await self.db.get_setting(QUEUE_STATE_KEY, "paused")
        if state == "running":
            self.queue_running.set()

    @property
    def state(self) -> str:
        return "running" if self.queue_running.is_set() else "paused"

    async def set_state(self, state: str) -> None:
        async with self.claim_lock:
            await self.db.set_setting(QUEUE_STATE_KEY, state)
            if state == "running":
                self.queue_running.set()
                self.wake.set()
            else:
                self.queue_running.clear()

    async def cancel_current(self) -> None:
        """Terminate the currently running job's process tree (cancel endpoint)."""
        self.cancel_requested = True
        await self.set_state("paused")
        if self.current_process is not None:
            await terminate_tree(self.current_process)

    async def run(self) -> None:
        while True:
            await self.queue_running.wait()
            row = await self.db.fetch_one(
                "SELECT * FROM training_jobs WHERE status = 'queued' "
                "ORDER BY queue_position, created_at LIMIT 1"
            )
            if row is None:
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=5)
                except TimeoutError:
                    pass
                continue
            try:
                await self._execute(row)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # keep the runner alive on unexpected failures
                await self._fail_job(row["id"], f"Internal runner error: {error}")

    async def _execute(self, row) -> None:
        job_id = row["id"]
        profile = TRAINING_PROFILES.get(row["profile_id"])
        if profile is None:
            await self._fail_job(
                job_id, f"Unknown training profile: {row['profile_id']}"
            )
            return

        values = json.loads(row["values_json"])
        stages = json.loads(row["stages_json"])
        log_path = self.settings.jobs_dir / f"{job_id}.log"
        snapshot_path = self.settings.jobs_dir / job_id / "dataset_config.toml"

        if not await self._claim_job(job_id, log_path):
            return
        await renumber_queue(self.db)

        if not snapshot_path.is_file():
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(row["dataset_config_toml"], encoding="utf-8")

        try:
            musubi_path = self.settings.resolve_inside_workspace(
                str(values.get("musubiPath", ""))
            )
            argv_by_stage = build_stage_argv(
                profile, values, musubi_path, snapshot_path
            )
        except ValueError as error:
            await self._fail_job(job_id, str(error), stages)
            return
        if not musubi_path.is_dir():
            await self._fail_job(
                job_id, f"musubi-tuner directory not found: {musubi_path}", stages
            )
            return
        if shutil.which("accelerate") is None:
            await self._fail_job(
                job_id, "accelerate is not available on the server", stages
            )
            return

        progress = {
            "epoch": None,
            "total_epochs": None,
            "step": None,
            "total_steps": None,
            "percent": None,
        }
        last_lines: deque[str] = deque(maxlen=20)

        with log_path.open("ab") as log_file:
            for index, stage in enumerate(stages):
                if self.cancel_requested:
                    self._clear_current(job_id)
                    return
                if stage["status"] != "pending":
                    continue
                stage["status"] = "running"
                if not await self._update_stages(
                    job_id, stages, current_stage=stage["key"]
                ):
                    self._clear_current(job_id)
                    return
                argv = argv_by_stage[stage["key"]]
                command_text = self._redact_output(" ".join(redact_argv(argv)), values)
                self._log(log_file, f"=== stage {stage['key']}: {command_text}")

                try:
                    self.current_process = await spawn(
                        argv, cwd=self.settings.workspace_root
                    )
                except OSError as error:
                    stage["status"] = "failed"
                    self._skip_remaining(stages, index + 1)
                    await self._fail_job(
                        job_id, f"Could not start {stage['key']}: {error}", stages
                    )
                    return

                if self.cancel_requested:
                    await terminate_tree(self.current_process)
                    self._clear_current(job_id)
                    return

                last_update = 0.0
                assert self.current_process.stdout is not None
                async for line in _iter_lines(self.current_process.stdout):
                    safe_line = self._redact_output(line, values)
                    self._log(log_file, safe_line)
                    last_lines.append(safe_line)
                    if stage["key"] == "train" and parse_progress(line, progress):
                        now = asyncio.get_running_loop().time()
                        if now - last_update >= PROGRESS_UPDATE_INTERVAL:
                            last_update = now
                            await self.db.execute(
                                "UPDATE training_jobs SET progress_json = ? WHERE id = ?",
                                (json.dumps(progress), job_id),
                            )

                return_code = await self.current_process.wait()
                self.current_process = None

                if self.cancel_requested:
                    # The cancel endpoint already marked the job and stages.
                    self._clear_current(job_id)
                    return
                if return_code != 0:
                    stage["status"] = "failed"
                    self._skip_remaining(stages, index + 1)
                    error_text = (
                        "\n".join(last_lines)
                        or f"{stage['key']} exited with code {return_code}"
                    )
                    await self._fail_job(job_id, error_text, stages, progress)
                    return
                stage["status"] = "completed"
                if not await self._update_stages(
                    job_id, stages, current_stage=stage["key"]
                ):
                    self._clear_current(job_id)
                    return

        # A successful job consumes the explicit Start action. Persist the pause before
        # publishing completion so clients never observe a completed job with a running queue.
        await self.set_state("paused")
        await self.db.execute(
            "UPDATE training_jobs SET status = 'completed', finished_at = ?, "
            "current_stage = NULL, stages_json = ?, progress_json = ? "
            "WHERE id = ? AND status = 'running'",
            (utc_now(), json.dumps(stages), json.dumps(progress), job_id),
        )
        self.current_job_id = None

    async def _claim_job(self, job_id: str, log_path: Path) -> bool:
        """Atomically claim a queued job while respecting pause and cancellation."""
        async with self.claim_lock:
            if not self.queue_running.is_set():
                return False
            # Publish ownership before the running status is visible so a
            # concurrent running-job cancellation can always reach this runner.
            self.current_job_id = job_id
            self.current_process = None
            self.cancel_requested = False
            cursor = await self.db.connection.execute(
                "UPDATE training_jobs SET status = 'running', started_at = ?, "
                "queue_position = NULL, log_path = ? "
                "WHERE id = ? AND status = 'queued'",
                (utc_now(), str(log_path), job_id),
            )
            await self.db.connection.commit()
            claimed = cursor.rowcount == 1
            await cursor.close()
            if not claimed:
                self._clear_current(job_id)
            return claimed

    def _clear_current(self, job_id: str) -> None:
        if self.current_job_id == job_id:
            self.current_job_id = None
            self.current_process = None

    def _log(self, log_file, line: str) -> None:
        if log_file.tell() < MAX_LOG_BYTES:
            log_file.write(line.encode("utf-8", errors="replace") + b"\n")
            log_file.flush()

    @staticmethod
    def _redact_output(line: str, values: dict) -> str:
        token = values.get("huggingfaceToken")
        if isinstance(token, str) and token:
            return line.replace(token, "***")
        return line

    @staticmethod
    def _skip_remaining(stages: list[dict], start: int) -> None:
        for stage in stages[start:]:
            if stage["status"] == "pending":
                stage["status"] = "skipped"

    async def _update_stages(
        self, job_id: str, stages: list[dict], current_stage: str
    ) -> bool:
        cursor = await self.db.connection.execute(
            "UPDATE training_jobs SET stages_json = ?, current_stage = ? "
            "WHERE id = ? AND status = 'running'",
            (json.dumps(stages), current_stage, job_id),
        )
        await self.db.connection.commit()
        updated = cursor.rowcount == 1
        await cursor.close()
        return updated

    async def _fail_job(
        self,
        job_id: str,
        error: str,
        stages: list[dict] | None = None,
        progress: dict | None = None,
    ) -> None:
        self.current_job_id = None
        self.current_process = None
        row = await self.db.fetch_one(
            "SELECT status FROM training_jobs WHERE id = ?", (job_id,)
        )
        if row is None or row["status"] == "cancelled":
            return
        await self.set_state("paused")
        await self.db.execute(
            "UPDATE training_jobs SET status = 'failed', error = ?, finished_at = ?, "
            "queue_position = NULL, current_stage = NULL, "
            "stages_json = COALESCE(?, stages_json), "
            "progress_json = COALESCE(?, progress_json) "
            "WHERE id = ? AND status != 'cancelled'",
            (
                error[-4000:],
                utc_now(),
                json.dumps(stages) if stages is not None else None,
                json.dumps(progress) if progress is not None else None,
                job_id,
            ),
        )


async def renumber_queue(db: Database) -> None:
    """Re-assign 0-based queue positions to all queued jobs, preserving order."""
    async with db.write_lock:
        rows = await db.fetch_all(
            "SELECT id FROM training_jobs WHERE status = 'queued' "
            "ORDER BY queue_position, created_at"
        )
        for position, row in enumerate(rows):
            await db.connection.execute(
                "UPDATE training_jobs SET queue_position = ? WHERE id = ?",
                (position, row["id"]),
            )
        await db.connection.commit()


async def recover_interrupted_jobs(db: Database) -> None:
    """Jobs still marked running were interrupted by a server restart."""
    now = utc_now()
    async with db.write_lock:
        rows = await db.fetch_all(
            "SELECT id, stages_json FROM training_jobs WHERE status = 'running'"
        )
        for row in rows:
            stages = json.loads(row["stages_json"])
            for stage in stages:
                if stage["status"] == "running":
                    stage["status"] = "failed"
                elif stage["status"] == "pending":
                    stage["status"] = "skipped"
            await db.connection.execute(
                "UPDATE training_jobs SET status = 'failed', error = ?, finished_at = ?, "
                "current_stage = NULL, stages_json = ? WHERE id = ?",
                ("Interrupted by server restart", now, json.dumps(stages), row["id"]),
            )
        await db.connection.commit()
    await renumber_queue(db)
