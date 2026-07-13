"""Host CPU, memory, and NVIDIA GPU utilization metrics."""

from __future__ import annotations

import asyncio
import csv
import shutil
import subprocess
from datetime import datetime, timezone
from io import StringIO
from typing import Any

import psutil
from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])

MIB = 1024 * 1024
NVIDIA_SMI_TIMEOUT_SECONDS = 3
NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
)


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _parse_nvidia_smi(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(StringIO(output)):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(NVIDIA_SMI_FIELDS):
            continue

        index, name, utilization, memory_used, memory_total, temperature = (
            value.strip() for value in row
        )
        used_mib = _number(memory_used)
        total_mib = _number(memory_total)
        used_bytes = round(used_mib * MIB) if used_mib is not None else None
        total_bytes = round(total_mib * MIB) if total_mib is not None else None
        memory_percent = (
            round(used_bytes / total_bytes * 100, 1)
            if used_bytes is not None and total_bytes
            else None
        )

        try:
            gpu_index = int(index)
        except ValueError:
            gpu_index = len(gpus)

        gpus.append(
            {
                "index": gpu_index,
                "name": name,
                "utilization_percent": _number(utilization),
                "memory_used_bytes": used_bytes,
                "memory_total_bytes": total_bytes,
                "memory_percent": memory_percent,
                "temperature_c": _number(temperature),
            }
        )
    return gpus


def _query_nvidia_gpus() -> tuple[list[dict[str, Any]], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return (
            [],
            "NVIDIA GPU metrics are unavailable because nvidia-smi was not found.",
        )

    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [], "NVIDIA GPU metrics are temporarily unavailable."

    if result.returncode != 0:
        return [], "NVIDIA GPU metrics are temporarily unavailable."

    gpus = _parse_nvidia_smi(result.stdout)
    if not gpus:
        return [], "No NVIDIA GPUs were detected."
    return gpus, None


def _collect_resource_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    gpus, gpu_error = _query_nvidia_gpus()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu": {
            "percent": round(psutil.cpu_percent(interval=0.1), 1),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "ram": {
            "percent": round(memory.percent, 1),
            "used_bytes": memory.used,
            "available_bytes": memory.available,
            "total_bytes": memory.total,
        },
        "gpus": gpus,
        "gpu_error": gpu_error,
    }


@router.get("/resources")
async def read_system_resources() -> dict[str, Any]:
    """Return a point-in-time host resource snapshot without blocking the event loop."""

    return await asyncio.to_thread(_collect_resource_snapshot)
