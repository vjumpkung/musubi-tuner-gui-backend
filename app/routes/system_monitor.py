"""Host resource monitoring API routes."""

import asyncio
from typing import Any

from fastapi import APIRouter

from ..services import system_monitor

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/resources")
async def read_system_resources() -> dict[str, Any]:
    """Return a point-in-time snapshot without blocking the event loop."""

    return await asyncio.to_thread(system_monitor._collect_resource_snapshot)
