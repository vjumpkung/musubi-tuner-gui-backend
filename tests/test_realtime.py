import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest

from app.config import Settings
from app.services import auth as auth_service
from app.services import downloads as downloads_service
from app.services import realtime, system_monitor
from app.services.downloads import DownloadManager
from app.services.realtime import RealtimeHub
from app.utils.origins import origin_allowed

from . import test_downloads, test_training
from .conftest import wait_for

# Reuse the fixtures and helpers of the training and download suites.
training_client = test_training.training_client
model = test_downloads.model
make_dataset = test_training.make_dataset
make_job = test_training.make_job
get_job = test_training.get_job

PASSWORD = "correct horse battery"
SOCKET_URL = "/api/socket.io/?EIO=4&transport=polling"
TEXT_PLAIN = {"Content-Type": "text/plain;charset=UTF-8"}
SNAPSHOT = {"timestamp": "2026-01-01T00:00:00+00:00", "cpu": {"percent": 1.5}}


def capture(client) -> list[tuple[str, object, str | None]]:
    """Replace the hub's delivery seam so emitted events are recorded."""
    captured: list[tuple[str, object, str | None]] = []

    async def fake_deliver(event, data, room):
        captured.append((event, data, room))

    client.app.state.realtime._deliver = fake_deliver
    return captured


def names(captured, start=0) -> list[str]:
    return [event for event, _, _ in list(captured)[start:]]


def payloads(captured, event) -> list:
    return [data for name, data, _ in list(captured) if name == event]


def job_updates(captured, job_id, status=None) -> list[dict]:
    return [
        data
        for data in payloads(captured, "job:update")
        if data["id"] == job_id and (status is None or data["status"] == status)
    ]


def poll_session(client) -> str:
    """Open a polling session and consume the CONNECT acknowledgement."""
    handshake = client.get(SOCKET_URL)
    assert handshake.status_code == 200
    assert handshake.text.startswith("0{")
    sid = json.loads(handshake.text[1:])["sid"]
    post = client.post(f"{SOCKET_URL}&sid={sid}", content="40", headers=TEXT_PLAIN)
    assert post.status_code == 200
    ack = client.get(f"{SOCKET_URL}&sid={sid}")
    assert "40{" in ack.text
    return sid


async def wait_until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for condition")


def standalone_settings(paths) -> Settings:
    return Settings(
        data_root=paths["data"],
        workspace_root=paths["workspace"],
        web_dir=paths["workspace"] / "no-web",
        cors_origins=(),
    )


@pytest.fixture
async def live_hub(paths):
    hub = RealtimeHub(standalone_settings(paths))
    delivered: list[tuple[str, object, str | None]] = []

    async def fake_deliver(event, data, room):
        delivered.append((event, data, room))

    hub._deliver = fake_deliver
    hub.start()
    yield hub, delivered
    await hub.stop()


# --- origin_allowed -------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "environ", "expected"),
    [
        (None, {}, True),
        ("", {"HTTP_HOST": "example.test"}, True),
        ("http://example.test", {"HTTP_HOST": "example.test"}, True),
        ("https://example.test:8443", {"HTTP_HOST": "example.test:8443"}, True),
        (
            "https://pod.proxy.test",
            {
                "HTTP_HOST": "127.0.0.1:8000",
                "HTTP_X_FORWARDED_HOST": "pod.proxy.test, other.test",
            },
            True,
        ),
        (
            "https://other.test",
            {"HTTP_X_FORWARDED_HOST": "pod.proxy.test, other.test"},
            False,
        ),
        ("http://localhost:5173", {"HTTP_HOST": "example.test"}, True),
        ("https://evil.test", {"HTTP_HOST": "example.test"}, False),
        ("null", {}, False),
        ("https://evil.test", {}, False),
    ],
)
def test_origin_allowed(origin, environ, expected):
    assert origin_allowed(origin, environ, ("http://localhost:5173",)) is expected


# --- handshake and auth ---------------------------------------------------


def test_polling_handshake_opens_session(client):
    response = client.get(SOCKET_URL)
    assert response.status_code == 200
    assert response.text.startswith("0{")


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        (None, 200),
        ("http://testserver", 200),
        ("http://localhost:5173", 200),
        ("https://evil.example", 400),
    ],
)
def test_handshake_origin_check(client, origin, status):
    headers = {"Origin": origin} if origin else {}
    assert client.get(SOCKET_URL, headers=headers).status_code == status


def test_handshake_requires_login_when_password_set(make_client, monkeypatch):
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", PASSWORD)
    monkeypatch.setattr(auth_service, "FAILURE_DELAY_SECONDS", 0)
    with make_client() as client:
        assert client.get(SOCKET_URL).status_code == 401
        login = client.post("/api/auth/login", json={"password": PASSWORD})
        assert login.status_code == 200

        # Cookie-authenticated socket POSTs are exempt from the CSRF header.
        poll_session(client)


def test_app_wires_hub_into_runner_and_downloads(client):
    hub = client.app.state.realtime
    assert isinstance(hub, RealtimeHub)
    assert client.app.state.runner.events is hub
    assert client.app.state.downloads.events is hub


# --- subscriptions --------------------------------------------------------


async def test_subscribe_and_unsubscribe_manage_rooms(live_hub, monkeypatch):
    hub, _ = live_hub
    entered: list[tuple[str, str]] = []
    left: list[tuple[str, str]] = []

    async def enter_room(sid, room, namespace=None):
        entered.append((sid, room))

    async def leave_room(sid, room, namespace=None):
        left.append((sid, room))

    monkeypatch.setattr(hub.sio, "enter_room", enter_room)
    monkeypatch.setattr(hub.sio, "leave_room", leave_room)
    subscribe = hub.sio.handlers["/"]["subscribe"]
    unsubscribe = hub.sio.handlers["/"]["unsubscribe"]

    await subscribe("sid-1", {"topic": "system"})
    await subscribe("sid-1", {"topic": "job-log", "job_id": "job-abc"})
    await unsubscribe("sid-1", {"topic": "system"})
    await unsubscribe("sid-1", {"topic": "job-log", "job_id": "job-abc"})
    assert entered == [("sid-1", "system"), ("sid-1", "job-log:job-abc")]
    assert left == [("sid-1", "system"), ("sid-1", "job-log:job-abc")]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "system",
        [],
        {},
        {"topic": "unknown"},
        {"topic": "job-log"},
        {"topic": "job-log", "job_id": ""},
        {"topic": "job-log", "job_id": 7},
        {"topic": "job-log", "job_id": "x" * 65},
    ],
)
async def test_invalid_subscription_payloads_are_ignored(
    live_hub, monkeypatch, payload
):
    hub, _ = live_hub
    calls: list[str] = []

    async def record(sid, room, namespace=None):
        calls.append(room)

    monkeypatch.setattr(hub.sio, "enter_room", record)
    monkeypatch.setattr(hub.sio, "leave_room", record)
    await hub.sio.handlers["/"]["subscribe"]("sid-1", payload)
    await hub.sio.handlers["/"]["unsubscribe"]("sid-1", payload)
    assert calls == []


async def test_subscribe_accepts_maximum_length_job_id(live_hub, monkeypatch):
    hub, _ = live_hub
    entered: list[str] = []

    async def enter_room(sid, room, namespace=None):
        entered.append(room)

    monkeypatch.setattr(hub.sio, "enter_room", enter_room)
    await hub.sio.handlers["/"]["subscribe"](
        "sid-1", {"topic": "job-log", "job_id": "x" * 64}
    )
    assert entered == ["job-log:" + "x" * 64]


async def test_has_subscribers_is_false_without_clients_or_on_error(
    live_hub, monkeypatch
):
    hub, _ = live_hub
    assert hub.has_subscribers("system") is False

    def broken(namespace, room):
        raise RuntimeError("boom")

    monkeypatch.setattr(hub.sio.manager, "get_participants", broken)
    assert hub.has_subscribers("system") is False


# --- hub delivery ---------------------------------------------------------


async def test_publish_is_noop_before_start_and_after_stop(paths):
    hub = RealtimeHub(standalone_settings(paths))
    delivered: list[tuple] = []

    async def fake_deliver(event, data, room):
        delivered.append((event, data, room))

    hub._deliver = fake_deliver
    hub.publish("early", {})
    hub.start()
    hub.publish("live", {"n": 1})
    await wait_until(lambda: len(delivered) == 1)
    await hub.stop()
    hub.publish("late", {})
    await hub.stop()  # idempotent
    await asyncio.sleep(0.05)
    assert [event for event, _, _ in delivered] == ["live"]


async def test_publish_preserves_order_and_rooms(live_hub):
    hub, delivered = live_hub
    hub.publish("a", {"n": 1})
    hub.publish("b", {"n": 2}, room="system")
    hub.publish("c", {"n": 3})
    await wait_until(lambda: len(delivered) == 3)
    assert delivered == [
        ("a", {"n": 1}, None),
        ("b", {"n": 2}, "system"),
        ("c", {"n": 3}, None),
    ]


async def test_dispatcher_survives_delivery_errors(live_hub):
    hub, delivered = live_hub

    async def flaky(event, data, room):
        if event == "bad":
            raise RuntimeError("boom")
        delivered.append((event, data, room))

    hub._deliver = flaky
    hub.publish("bad", {})
    hub.publish("good", {})
    await wait_until(lambda: len(delivered) == 1)
    assert delivered[0][0] == "good"


async def test_publish_log_requires_subscribers(live_hub):
    hub, delivered = live_hub
    hub.publish_log("job-1", 0, 3, "abc")
    assert hub._log_pending == {}
    assert hub._log_timers == {}
    await asyncio.sleep(0.05)
    assert delivered == []


async def test_publish_log_coalesces_contiguous_chunks(live_hub, monkeypatch):
    hub, delivered = live_hub
    monkeypatch.setattr(hub, "has_subscribers", lambda room: True)
    monkeypatch.setattr(realtime, "LOG_FLUSH_DELAY_SECONDS", 0.02)
    hub.publish_log("job-1", 0, 3, "abc")
    hub.publish_log("job-1", 3, 6, "def")
    await wait_until(lambda: len(delivered) == 1)
    assert delivered == [
        (
            "job:log",
            {"job_id": "job-1", "offset": 0, "next_offset": 6, "content": "abcdef"},
            "job-log:job-1",
        )
    ]
    assert hub._log_pending == {}
    assert hub._log_timers == {}


async def test_publish_log_flushes_on_gap_and_size_cap(live_hub, monkeypatch):
    hub, delivered = live_hub
    monkeypatch.setattr(hub, "has_subscribers", lambda room: True)
    monkeypatch.setattr(realtime, "LOG_FLUSH_DELAY_SECONDS", 0.02)

    hub.publish_log("job-1", 0, 3, "abc")
    hub.publish_log("job-1", 10, 13, "xyz")  # gap: flush the old chunk first
    big = "a" * 60_000
    hub.publish_log("job-2", 0, 60_000, big)
    hub.publish_log("job-2", 60_000, 70_000, "b" * 10_000)  # would exceed 64 KiB

    await wait_until(lambda: len(delivered) == 4)
    by_job: dict[str, list[dict]] = {}
    for event, data, room in delivered:
        assert event == "job:log"
        assert room == f"job-log:{data['job_id']}"
        by_job.setdefault(data["job_id"], []).append(data)
    assert [(c["offset"], c["next_offset"], c["content"]) for c in by_job["job-1"]] == [
        (0, 3, "abc"),
        (10, 13, "xyz"),
    ]
    assert [(c["offset"], c["next_offset"]) for c in by_job["job-2"]] == [
        (0, 60_000),
        (60_000, 70_000),
    ]
    assert by_job["job-2"][0]["content"] == big


async def test_stop_flushes_and_cancels_log_timers_without_raising(
    live_hub, monkeypatch
):
    hub, _ = live_hub
    monkeypatch.setattr(hub, "has_subscribers", lambda room: True)
    hub.publish_log("job-1", 0, 3, "abc")
    assert "job-1" in hub._log_timers
    await hub.stop()
    assert hub._log_pending == {}
    assert hub._log_timers == {}


# --- system resources -----------------------------------------------------


def test_system_loop_only_publishes_with_subscribers(make_client, monkeypatch):
    monkeypatch.setattr(realtime, "SYSTEM_PUSH_INTERVAL_SECONDS", 0.05)
    calls: list[int] = []

    def fake_snapshot():
        calls.append(1)
        return SNAPSHOT

    monkeypatch.setattr(system_monitor, "_collect_resource_snapshot", fake_snapshot)
    with make_client() as client:
        hub = client.app.state.realtime
        captured = capture(client)

        # Without subscribers the loop must not even collect a snapshot.
        time.sleep(0.3)
        assert calls == []
        assert "system:resources" not in names(captured)

        monkeypatch.setattr(hub, "has_subscribers", lambda room: room == "system")
        wait_for(lambda: payloads(captured, "system:resources"))
        event, data, room = next(
            item for item in list(captured) if item[0] == "system:resources"
        )
        assert data == SNAPSHOT
        assert room == "system"


def test_subscribe_over_polling_receives_system_resources(make_client, monkeypatch):
    monkeypatch.setattr(realtime, "SYSTEM_PUSH_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(system_monitor, "_collect_resource_snapshot", lambda: SNAPSHOT)
    with make_client() as client:
        hub = client.app.state.realtime
        sid = poll_session(client)
        packet = '42["subscribe",{"topic":"system"}]'
        post = client.post(
            f"{SOCKET_URL}&sid={sid}", content=packet, headers=TEXT_PLAIN
        )
        assert post.status_code == 200
        wait_for(lambda: hub.has_subscribers("system"))

        body = ""
        for _ in range(20):
            body += client.get(f"{SOCKET_URL}&sid={sid}").text
            if "system:resources" in body:
                break
        assert '42["system:resources",' in body
        assert '"percent":1.5' in body


# --- training events ------------------------------------------------------


def test_job_lifecycle_publishes_state_events(training_client):
    client = training_client
    captured = capture(client)
    dataset_id = make_dataset(client)

    job = make_job(client, dataset_id, name="lifecycle")
    job_id = job["id"]
    wait_for(lambda: len(captured) >= 3)
    assert names(captured) == ["queue:update", "job:update", "queue:update"]
    assert job_updates(captured, job_id)[0]["status"] == "queued"
    assert payloads(captured, "queue:update")[-1] == {
        "state": "paused",
        "queued": 1,
        "running_job_id": None,
    }
    assert "job:log" not in names(captured)  # nobody subscribed to the log

    client.post("/api/training/queue/start")
    wait_for(lambda: job_updates(captured, job_id, "completed"))
    wait_for(
        lambda: (
            payloads(captured, "queue:update")[-1]
            == {"state": "paused", "queued": 0, "running_job_id": None}
        )
    )

    events = list(captured)
    queue_states = [data["state"] for data in payloads(captured, "queue:update")]
    assert "running" in queue_states
    first_running_queue = next(
        index
        for index, (event, data, _) in enumerate(events)
        if event == "queue:update" and data["state"] == "running"
    )
    first_running_job = next(
        index
        for index, (event, data, _) in enumerate(events)
        if event == "job:update"
        and data["id"] == job_id
        and data["status"] == "running"
    )
    jobs_changed = next(
        index for index, (event, _, _) in enumerate(events) if event == "jobs:changed"
    )
    completed = next(
        index
        for index, (event, data, _) in enumerate(events)
        if event == "job:update" and data["status"] == "completed"
    )
    assert first_running_queue < first_running_job < jobs_changed < completed
    assert events[-1][0] == "queue:update"
    assert completed < len(events) - 1

    statuses = [data["status"] for data in job_updates(captured, job_id)]
    assert statuses[0] == "queued"
    assert statuses[-1] == "completed"
    assert set(statuses) == {"queued", "running", "completed"}
    stages = {data["current_stage"] for data in job_updates(captured, job_id)}
    assert {"cache_latents", "cache_text_encoder", "train"} <= stages
    assert any(
        data["progress"]["epoch"] is not None for data in job_updates(captured, job_id)
    )
    final = job_updates(captured, job_id, "completed")[-1]
    assert final["queue_position"] is None
    assert final["finished_at"]
    assert "job:log" not in names(captured)


def test_cancel_reorder_and_delete_publish_events(training_client):
    client = training_client
    captured = capture(client)
    dataset_id = make_dataset(client)
    first = make_job(client, dataset_id, name="first")
    second = make_job(client, dataset_id, name="second")
    wait_for(lambda: len(captured) >= 6)
    assert payloads(captured, "queue:update")[-1]["queued"] == 2

    def expect(start, expected):
        wait_for(lambda: len(captured) >= start + len(expected))
        assert names(captured, start) == expected

    start = len(captured)
    reorder = client.patch(
        f"/api/training/jobs/{second['id']}", json={"queue_position": 0}
    )
    assert reorder.status_code == 200
    expect(start, ["jobs:changed"])

    start = len(captured)
    cancel = client.post(f"/api/training/jobs/{first['id']}/cancel")
    assert cancel.status_code == 200
    expect(start, ["job:update", "queue:update", "jobs:changed", "queue:update"])
    cancelled = job_updates(captured, first["id"], "cancelled")
    assert cancelled and cancelled[-1]["queue_position"] is None
    assert payloads(captured, "queue:update")[-1]["queued"] == 1

    # Cancelling an already cancelled job publishes nothing, so the delete
    # sequence below must follow directly.
    start = len(captured)
    assert client.post(f"/api/training/jobs/{first['id']}/cancel").status_code == 200
    assert client.delete(f"/api/training/jobs/{first['id']}").status_code == 204
    expect(start, ["job:removed", "queue:update"])
    assert payloads(captured, "job:removed")[-1] == {"id": first["id"]}

    start = len(captured)
    retry_source = client.post(f"/api/training/jobs/{second['id']}/cancel")
    assert retry_source.status_code == 200
    expect(start, ["job:update", "queue:update", "jobs:changed", "queue:update"])
    start = len(captured)
    retried = client.post(f"/api/training/jobs/{second['id']}/retry")
    assert retried.status_code == 202
    expect(start, ["queue:update", "job:update", "queue:update"])
    assert job_updates(captured, retried.json()["id"], "queued")


def test_dataset_delete_publishes_jobs_changed(training_client):
    client = training_client
    captured = capture(client)
    dataset_id = make_dataset(client)
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 204
    wait_for(lambda: "jobs:changed" in names(captured))
    assert payloads(captured, "jobs:changed") == [{}]


def test_running_job_cancel_publishes_cancelled_state(training_client):
    client = training_client
    captured = capture(client)
    dataset_id = make_dataset(client)
    job = make_job(client, dataset_id, name="slow", stub_mode="slow")
    client.post("/api/training/queue/start")
    wait_for(
        lambda: any(
            data["current_stage"] == "train"
            for data in job_updates(captured, job["id"], "running")
        )
    )
    cancel = client.post(f"/api/training/jobs/{job['id']}/cancel")
    assert cancel.status_code == 200
    wait_for(lambda: job_updates(captured, job["id"], "cancelled"))
    wait_for(
        lambda: (
            payloads(captured, "queue:update")[-1]
            == {"state": "paused", "queued": 0, "running_job_id": None}
        )
    )
    cancelled = job_updates(captured, job["id"], "cancelled")[-1]
    assert cancelled["current_stage"] is None
    assert cancelled["finished_at"]


def test_failed_job_publishes_without_leaking_token(training_client, monkeypatch):
    client = training_client
    token = "hf_realtime_secret_token"
    hub = client.app.state.realtime
    monkeypatch.setattr(
        hub, "has_subscribers", lambda room: room.startswith("job-log:")
    )
    captured = capture(client)
    dataset_id = make_dataset(client)
    job = make_job(
        client,
        dataset_id,
        stub_mode="leak_fail",
        values={"huggingfaceToken": token},
    )
    client.post("/api/training/queue/start")
    wait_for(lambda: job_updates(captured, job["id"], "failed"))
    wait_for(
        lambda: any("***" in data["content"] for data in payloads(captured, "job:log"))
    )
    wait_for(
        lambda: (
            payloads(captured, "queue:update")[-1]
            == {"state": "paused", "queued": 0, "running_job_id": None}
        )
    )

    failed = job_updates(captured, job["id"], "failed")[-1]
    assert failed["error"]
    serialized = json.dumps(list(captured), default=str)
    assert token not in serialized


def test_job_log_chunks_are_contiguous_and_match_http_logs(
    training_client, monkeypatch
):
    client = training_client
    hub = client.app.state.realtime
    monkeypatch.setattr(
        hub, "has_subscribers", lambda room: room.startswith("job-log:")
    )
    monkeypatch.setattr(realtime, "LOG_FLUSH_DELAY_SECONDS", 0.05)
    captured = capture(client)
    dataset_id = make_dataset(client)
    job = make_job(client, dataset_id, name="logged")
    job_id = job["id"]
    client.post("/api/training/queue/start")
    wait_for(lambda: get_job(client, job_id)["status"] == "completed")

    full, offset = "", 0
    while True:
        chunk = client.get(
            f"/api/training/jobs/{job_id}/logs", params={"offset": offset}
        ).json()
        if chunk["next_offset"] == offset:
            break
        full += chunk["content"]
        offset = chunk["next_offset"]
    assert "train done" in full

    wait_for(
        lambda: (
            (chunks := payloads(captured, "job:log"))
            and chunks[-1]["next_offset"] == offset
        )
    )
    events = [item for item in list(captured) if item[0] == "job:log"]
    chunks = [data for _, data, _ in events]
    assert all(room == f"job-log:{job_id}" for _, _, room in events)
    assert all(chunk["job_id"] == job_id for chunk in chunks)
    assert chunks[0]["offset"] == 0
    for previous, current in zip(chunks, chunks[1:]):
        assert current["offset"] == previous["next_offset"]
    assert "".join(chunk["content"] for chunk in chunks) == full


# --- downloads ------------------------------------------------------------


def download_updates(captured, job_id) -> list[dict]:
    return [
        data for data in payloads(captured, "download:update") if data["id"] == job_id
    ]


def test_download_publishes_queued_running_and_completed(client, model):
    captured = capture(client)
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"model data")
    )
    response = client.post(
        "/api/downloads",
        json={"script_id": model.id, "destination": ".", "hf_token": "hf_dl_secret"},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]
    wait_for(
        lambda: any(
            data["status"] == "completed" for data in download_updates(captured, job_id)
        )
    )
    updates = download_updates(captured, job_id)
    statuses = [data["status"] for data in updates]
    assert statuses[0] == "queued"
    assert "running" in statuses
    assert statuses[-1] == "completed"
    assert updates[-1]["progress"] == 100
    assert "hf_dl_secret" not in json.dumps(list(captured), default=str)
    assert all(room is None for _, _, room in list(captured))


def test_failed_download_publishes_failed(client, model):
    captured = capture(client)
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(404, content=b"missing")
    )
    response = client.post(
        "/api/downloads", json={"script_id": model.id, "destination": "."}
    )
    job_id = response.json()["id"]
    wait_for(
        lambda: any(
            data["status"] == "failed" for data in download_updates(captured, job_id)
        )
    )
    failed = download_updates(captured, job_id)[-1]
    assert (
        failed["error"]
        == f"File not found on Hugging Face: {test_downloads.MODEL_PATH}"
    )


async def test_cancel_before_start_publishes_cancelled(live_hub, model, paths):
    hub, delivered = live_hub
    manager = DownloadManager(standalone_settings(paths), events=hub)
    manager.transport = httpx.MockTransport(
        lambda request: pytest.fail("Must not send")
    )
    job = manager.start(model.id, ".")
    await manager.cancel(job.id)
    assert not job.started
    await wait_until(
        lambda: [d["status"] for _, d, _ in delivered] == ["queued", "cancelled"]
    )
    assert {data["id"] for _, data, _ in delivered} == {job.id}
    assert "last_publish" not in delivered[0][1]


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: int):
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            yield b"x" * (1024 * 1024)


@pytest.mark.parametrize(
    ("interval", "expect_progress"), [(0.0, True), (3600.0, False)]
)
async def test_download_progress_updates_are_throttled(
    live_hub, model, paths, monkeypatch, interval, expect_progress
):
    hub, delivered = live_hub
    monkeypatch.setattr(
        downloads_service, "PROGRESS_PUBLISH_INTERVAL_SECONDS", interval
    )
    manager = DownloadManager(standalone_settings(paths), events=hub)
    chunks = 3
    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Length": str(chunks * 1024 * 1024)},
            stream=ChunkedStream(chunks),
        )
    )
    job = manager.start(model.id, ".")
    await job.task
    await wait_until(lambda: delivered and delivered[-1][1]["status"] == "completed")
    intermediate = [
        data
        for _, data, _ in delivered
        if data["status"] == "running" and 0 < data.get("progress", 0) < 100
    ]
    assert bool(intermediate) is expect_progress
