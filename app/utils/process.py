"""Cross-platform subprocess spawning and process-tree termination.

The production target is Linux (training subprocesses, POSIX process groups);
the Windows branch keeps local development working.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
from pathlib import Path

TERMINATE_GRACE_SECONDS = 10.0


async def spawn(
    argv: list[str], cwd: Path, env: dict[str, str] | None = None
) -> asyncio.subprocess.Process:
    """Spawn argv in its own process group with combined stdout/stderr."""
    # Resolve a bare program name against PATH ourselves: on Windows,
    # CreateProcess searches the parent executable's directory before PATH, so a
    # bare ``python`` would run the backend's interpreter instead of the one the
    # trainer environment put on PATH.
    if not os.path.isabs(argv[0]):
        resolved = shutil.which(argv[0])
        if resolved is None:
            raise OSError(f"{argv[0]} was not found on PATH")
        argv = [resolved, *argv[1:]]
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    merged_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        **(env or {}),
    }
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=merged_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **kwargs,
    )


async def terminate_tree(process: asyncio.subprocess.Process) -> None:
    """Terminate a process and its descendants: SIGTERM, then SIGKILL after a grace period."""
    if process.returncode is not None:
        return
    if os.name == "nt":
        # No process-group SIGTERM on Windows; kill the tree forcefully.
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            process.kill()
        return

    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
    except TimeoutError:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
