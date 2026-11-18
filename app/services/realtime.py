"""Socket.IO realtime hub: ordered server-to-client event delivery."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

import socketio

from ..config import Settings
from ..utils.origins import origin_allowed
from . import system_monitor

logger = logging.getLogger(__name__)

SOCKET_PATH = "/api/socket.io"
SYSTEM_ROOM = "system"
SYSTEM_PUSH_INTERVAL_SECONDS = 5.0
LOG_FLUSH_DELAY_SECONDS = 0.25
LOG_CHUNK_MAX_CHARS = 64 * 1024
MAX_JOB_ID_LENGTH = 64


def _topic_room(data: Any) -> str | None:
    """Map a subscribe/unsubscribe payload to a room name; None when invalid."""
    if not isinstance(data, dict):
        return None
    topic = data.get("topic")
    if topic == "system":
        return SYSTEM_ROOM
    if topic == "job-log":
        job_id = data.get("job_id")
        if isinstance(job_id, str) and 1 <= len(job_id) <= MAX_JOB_ID_LENGTH:
            return f"job-log:{job_id}"
    return None


class RealtimeHub:
    def __init__(self, settings: Settings):
        self.settings = settings
        cors_origins = settings.cors_origins
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            # A callable keeps same-origin working behind proxies; a list would
            # replace it and `[]` would disable the origin check entirely.
            cors_allowed_origins=lambda origin, environ: origin_allowed(
                origin, environ, cors_origins
            ),
            logger=False,
            engineio_logger=False,
        )
        self.sio.on("subscribe", self._on_subscribe)
        self.sio.on("unsubscribe", self._on_unsubscribe)
        self._started = False
        self._queue: asyncio.Queue[tuple[str, Any, str | None]] | None = None
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._system_task: asyncio.Task[None] | None = None
        self._log_pending: dict[str, dict[str, Any]] = {}
        self._log_timers: dict[str, asyncio.TimerHandle] = {}

    def asgi_app(self) -> socketio.ASGIApp:
        # The path includes the mount prefix because the mounted child still
        # sees the full request path.
        return socketio.ASGIApp(self.sio, socketio_path=SOCKET_PATH)

    def start(self) -> None:
        if self._started:
            return
        self._queue = asyncio.Queue()
        self._dispatcher_task = asyncio.create_task(self._dispatch())
        self._system_task = asyncio.create_task(self._system_loop())
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        self._flush_all_logs()
        self._started = False
        for task in (self._dispatcher_task, self._system_task):
            if task is None:
                continue
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._dispatcher_task = None
        self._system_task = None
        for timer in self._log_timers.values():
            timer.cancel()
        self._log_timers.clear()
        self._log_pending.clear()
        self._queue = None
        try:
            await self.sio.shutdown()
        except Exception:
            logger.exception("Socket.IO shutdown failed")

    def publish(self, event: str, data: Any, room: str | None = None) -> None:
        """Queue an event for ordered delivery. Safe to call from sync code."""
        if not self._started or self._queue is None:
            return
        self._queue.put_nowait((event, data, room))

    async def _deliver(self, event: str, data: Any, room: str | None) -> None:
        await self.sio.emit(event, data, room=room)

    async def _dispatch(self) -> None:
        assert self._queue is not None
        queue = self._queue
        while True:
            event, data, room = await queue.get()
            try:
                await self._deliver(event, data, room)
            except Exception:
                logger.exception("Failed to deliver realtime event %s", event)

    def has_subscribers(self, room: str) -> bool:
        try:
            return any(True for _ in self.sio.manager.get_participants("/", room))
        except Exception:
            return False

    def publish_log(
        self, job_id: str, offset: int, next_offset: int, text: str
    ) -> None:
        """Coalesce contiguous log text per job and publish it shortly after."""
        if not self._started or not self.has_subscribers(f"job-log:{job_id}"):
            return
        pending = self._log_pending.get(job_id)
        if (
            pending is not None
            and pending["next_offset"] == offset
            and len(pending["content"]) + len(text) <= LOG_CHUNK_MAX_CHARS
        ):
            pending["content"] += text
            pending["next_offset"] = next_offset
            return
        if pending is not None:
            self._flush_log(job_id)
        self._log_pending[job_id] = {
            "job_id": job_id,
            "offset": offset,
            "next_offset": next_offset,
            "content": text,
        }
        self._log_timers[job_id] = asyncio.get_running_loop().call_later(
            LOG_FLUSH_DELAY_SECONDS, self._flush_log, job_id
        )

    def _flush_log(self, job_id: str) -> None:
        timer = self._log_timers.pop(job_id, None)
        if timer is not None:
            timer.cancel()
        pending = self._log_pending.pop(job_id, None)
        if pending is not None:
            self.publish("job:log", pending, room=f"job-log:{job_id}")

    def _flush_all_logs(self) -> None:
        for job_id in list(self._log_pending):
            self._flush_log(job_id)

    async def _on_subscribe(self, sid: str, data: Any = None) -> None:
        room = _topic_room(data)
        if room is not None:
            await self.sio.enter_room(sid, room)

    async def _on_unsubscribe(self, sid: str, data: Any = None) -> None:
        room = _topic_room(data)
        if room is not None:
            await self.sio.leave_room(sid, room)

    async def _system_loop(self) -> None:
        while True:
            await asyncio.sleep(SYSTEM_PUSH_INTERVAL_SECONDS)
            try:
                if not self.has_subscribers(SYSTEM_ROOM):
                    continue
                snapshot = await asyncio.to_thread(
                    system_monitor._collect_resource_snapshot
                )
                self.publish("system:resources", snapshot, room=SYSTEM_ROOM)
            except Exception:
                logger.exception("Failed to publish system resources")
