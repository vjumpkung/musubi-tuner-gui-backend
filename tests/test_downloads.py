import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.download_catalog import DOWNLOAD_MODELS
from app.schemas.downloads import DownloadFile, DownloadModel
from app.services.downloads import DownloadManager

from .conftest import wait_for

MODEL_PATH = "diffusion_models/model.safetensors"
MODEL_URL = "https://huggingface.co/test/repo/resolve/main/model.safetensors"


@pytest.fixture
def model(monkeypatch) -> DownloadModel:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model = DownloadModel(
        id="test-model",
        name="Test model",
        files=[DownloadFile(url=MODEL_URL, path=MODEL_PATH)],
    )
    monkeypatch.setitem(DOWNLOAD_MODELS, model.id, model)
    return model


@pytest.fixture
def manager(paths, monkeypatch) -> DownloadManager:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return DownloadManager(
        Settings(
            data_root=paths["data"],
            workspace_root=paths["workspace"],
            web_dir=paths["workspace"] / "no-web",
        )
    )


def start(client, model, **values):
    return client.post(
        "/api/downloads",
        json={"script_id": model.id, "destination": ".", **values},
    )


def finished(client, job_id):
    def poll():
        job = client.get(f"/api/downloads/{job_id}").json()
        return job if job["status"] in {"completed", "failed", "cancelled"} else None

    return wait_for(poll)


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content


def test_unknown_model(client):
    response = client.post("/api/downloads", json={"script_id": "unknown"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown download model"}


def test_destination_escape(client, model, tmp_path):
    response = start(client, model, destination=str(tmp_path / "outside"))
    assert response.status_code == 400


def test_success_writes_all_files(client, model, paths, monkeypatch):
    bundle = DownloadModel(
        id=model.id,
        name=model.name,
        files=[
            *model.files,
            DownloadFile(
                url="https://huggingface.co/test/repo/resolve/main/vae.bin",
                path="vae/nested/vae.bin",
            ),
        ],
    )
    monkeypatch.setitem(DOWNLOAD_MODELS, model.id, bundle)
    contents = {"model.safetensors": b"model data", "vae.bin": b"vae data"}

    def handle(request):
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(
            200, content=contents[request.url.path.rsplit("/", 1)[-1]]
        )

    client.app.state.downloads.transport = httpx.MockTransport(handle)
    response = start(client, model, destination="models")
    assert response.status_code == 202
    job = finished(client, response.json()["id"])
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["message"] == "Download completed"
    assert job["current_file"] == "vae/nested/vae.bin"
    destination = paths["workspace"] / "models"
    assert (destination / MODEL_PATH).read_bytes() == b"model data"
    assert (destination / "vae/nested/vae.bin").read_bytes() == b"vae data"
    assert not list(destination.rglob("*.part"))
    assert client.get("/api/downloads").json() == [job]
    assert client.post(f"/api/downloads/{job['id']}/cancel").status_code == 409


def test_token_sent_only_to_original_origin(client, model):
    seen = []
    token = "hf_test_secret"

    def handle(request):
        seen.append((request.url.host, request.headers.get("Authorization")))
        if request.url.host == "huggingface.co":
            return httpx.Response(
                302, headers={"Location": "https://cdn.example.test/signed-file"}
            )
        return httpx.Response(200, content=b"checkpoint")

    client.app.state.downloads.transport = httpx.MockTransport(handle)
    response = start(client, model, hf_token=token)
    job = finished(client, response.json()["id"])
    assert job["status"] == "completed"
    assert seen == [("huggingface.co", f"Bearer {token}"), ("cdn.example.test", None)]
    assert token not in response.text
    assert token not in json.dumps(job)
    assert token not in client.get("/api/downloads").text


def test_token_absent_from_start_list_get_cancel(client, model):
    token = "hf_cancel_secret"

    async def handle(request):
        await asyncio.Event().wait()

    client.app.state.downloads.transport = httpx.MockTransport(handle)
    response = start(client, model, hf_token=token)
    job_id = response.json()["id"]
    responses = [
        response,
        client.get("/api/downloads"),
        client.get(f"/api/downloads/{job_id}"),
        client.post(f"/api/downloads/{job_id}/cancel"),
        client.post(f"/api/downloads/{job_id}/cancel"),
    ]
    assert responses[-1].json()["status"] == "cancelled"
    assert all(token not in response.text for response in responses)


def test_invalid_download_request_does_not_echo_token(client, model):
    token = "hf_" + "secret" * 400
    response = start(client, model, hf_token=token)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid download request"}
    assert token not in response.text
    other = client.post("/api/training/jobs", json={})
    assert other.status_code == 422
    assert isinstance(other.json()["detail"], list)


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (
            401,
            f"Access denied for {MODEL_PATH}: accept the model license on Hugging Face and provide a token",
        ),
        (
            403,
            f"Access denied for {MODEL_PATH}: accept the model license on Hugging Face and provide a token",
        ),
        (404, f"File not found on Hugging Face: {MODEL_PATH}"),
        (503, f"Download failed for {MODEL_PATH} (HTTP 503)"),
    ],
)
def test_http_failures_are_safe(client, model, status, error):
    token = "hf_error_secret"
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(status, content=f"upstream secret: {token}")
    )
    response = start(client, model, hf_token=token)
    job = finished(client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == error
    assert token not in json.dumps(job)
    assert client.post(f"/api/downloads/{job['id']}/cancel").status_code == 409


def test_network_error_does_not_leak_exception(client, model):
    def handle(request):
        raise httpx.ConnectError("hf_secret signed upstream URL", request=request)

    client.app.state.downloads.transport = httpx.MockTransport(handle)
    response = start(client, model)
    job = finished(client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == f"Download failed for {MODEL_PATH}: network error"


def test_truncated_download_preserves_existing_file(client, model, paths):
    target = paths["workspace"] / MODEL_PATH
    target.parent.mkdir()
    target.write_bytes(b"previous model")
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Length": "20"}, stream=ByteStream(b"short")
        )
    )
    response = start(client, model)
    job = finished(client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == f"Incomplete download for {MODEL_PATH}"
    assert target.read_bytes() == b"previous model"
    assert not list(paths["workspace"].rglob("*.part"))


def test_same_size_existing_file_is_replaced(client, model, paths):
    target = paths["workspace"] / MODEL_PATH
    target.parent.mkdir()
    target.write_bytes(b"wrong")
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"right")
    )
    response = start(client, model)
    assert finished(client, response.json()["id"])["status"] == "completed"
    assert target.read_bytes() == b"right"


def test_missing_content_length(client, model, paths):
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=ByteStream(b"streamed"))
    )
    response = start(client, model)
    assert finished(client, response.json()["id"])["status"] == "completed"
    assert (paths["workspace"] / MODEL_PATH).read_bytes() == b"streamed"


def test_unexpected_content_encoding(client, model, paths):
    client.app.state.downloads.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=ByteStream(b"encoded")
        )
    )
    response = start(client, model)
    job = finished(client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["error"] == f"Unexpected content encoding for {MODEL_PATH}"
    assert not (paths["workspace"] / MODEL_PATH).exists()
    assert not list(paths["workspace"].rglob("*.part"))


def test_environment_token_fallback(client, model, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_environment_secret")
    seen = []

    def handle(request):
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, content=b"model")

    client.app.state.downloads.transport = httpx.MockTransport(handle)
    response = start(client, model)
    assert finished(client, response.json()["id"])["status"] == "completed"
    assert seen == ["Bearer hf_environment_secret"]


def test_unknown_job(client):
    assert client.get("/api/downloads/missing").status_code == 404
    assert client.post("/api/downloads/missing/cancel").status_code == 404


async def test_overlap_blocked_during_cancel_cleanup(manager, model, monkeypatch):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    second_started = asyncio.Event()
    cancellations = []

    class ControlledStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * (1024 * 1024)
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellations.append("stream")
                raise

        async def aclose(self) -> None:
            cleaning.set()
            await release.wait()

    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=ControlledStream())
    )
    other = DownloadModel(id="overlap", name="Overlap", files=model.files)
    monkeypatch.setitem(DOWNLOAD_MODELS, other.id, other)
    job = manager.start(model.id, ".")
    await entered.wait()
    assert list(job.destination.rglob("*.part"))
    first_cancel = asyncio.create_task(manager.cancel(job.id))
    await cleaning.wait()
    assert job.status == "cancelled"
    assert not first_cancel.done()
    with pytest.raises(HTTPException) as error:
        manager.start(other.id, ".")
    assert error.value.status_code == 409

    async def cancel_again():
        second_started.set()
        return await manager.cancel(job.id)

    second_cancel = asyncio.create_task(cancel_again())
    await second_started.wait()
    assert not second_cancel.done()
    assert job.task.cancelling() == 1
    release.set()
    assert await first_cancel is job
    assert await second_cancel is job
    assert cancellations == ["stream"]
    assert not list(job.destination.rglob("*.part"))
    assert not manager.reservations
    assert (await manager.cancel(job.id)).status == "cancelled"
    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"new")
    )
    replacement = manager.start(other.id, ".")
    await replacement.task
    assert replacement.status == "completed"


async def test_cancel_before_coroutine_start_releases_reservation(manager, model):
    manager.transport = httpx.MockTransport(
        lambda request: pytest.fail("Must not send")
    )
    job = manager.start(model.id, ".")
    await manager.cancel(job.id)
    assert not job.started
    assert job.task.done()
    assert job.status == "cancelled"
    assert not manager.reservations
    replacement = manager.start(model.id, ".")
    await manager.cancel(replacement.id)


async def test_target_escape_rejected_before_start(
    manager, model, monkeypatch, tmp_path
):
    original_resolve = Path.resolve
    target = manager.settings.workspace_root / MODEL_PATH

    def resolve(path, *args, **kwargs):
        if path == target:
            return tmp_path / "outside.safetensors"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(HTTPException) as error:
        manager.start(model.id, ".")
    assert error.value.status_code == 400
    assert not manager.jobs


@pytest.mark.parametrize("escaped_path", ["temporary", "final"])
async def test_install_rechecks_paths(
    manager, model, monkeypatch, tmp_path, escaped_path
):
    original_resolve = Path.resolve
    target = manager.settings.workspace_root / MODEL_PATH
    target.parent.mkdir()
    target.write_bytes(b"old")
    transferring = False

    def resolve(path, *args, **kwargs):
        if transferring and (
            (escaped_path == "final" and path == target)
            or (escaped_path == "temporary" and path.name.endswith(".part"))
        ):
            return tmp_path / "outside.safetensors"
        return original_resolve(path, *args, **kwargs)

    def handle(request):
        nonlocal transferring
        transferring = True
        return httpx.Response(200, content=b"new")

    monkeypatch.setattr(Path, "resolve", resolve)
    manager.transport = httpx.MockTransport(handle)
    job = manager.start(model.id, ".")
    await job.task
    assert job.status == "failed"
    assert job.error == f"Download path is outside destination: {MODEL_PATH}"
    assert target.read_bytes() == b"old"
    assert not list(job.destination.rglob("*.part"))
    assert not manager.reservations


async def test_progress_stays_below_complete_until_install(manager, model):
    entered = asyncio.Event()
    release = asyncio.Event()
    size = 1024 * 1024

    class ProgressStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * size
            entered.set()
            await release.wait()

    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Length": str(size)}, stream=ProgressStream()
        )
    )
    job = manager.start(model.id, ".")
    await entered.wait()
    assert job.status == "running"
    assert job.progress == 99.9
    assert job.current_file == MODEL_PATH
    assert job.message == f"Downloading {MODEL_PATH} (1/1)"
    assert not (job.destination / MODEL_PATH).exists()
    release.set()
    await job.task
    assert job.progress == 100
    assert job.status == "completed"


async def test_shutdown_cleans_temporary_files(manager, model):
    entered = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * (1024 * 1024)
            entered.set()
            await asyncio.Event().wait()

    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=SlowStream())
    )
    job = manager.start(model.id, ".")
    await entered.wait()
    assert list(job.destination.rglob("*.part"))
    await manager.shutdown()
    assert job.status == "cancelled"
    assert job.task.done()
    assert not list(job.destination.rglob("*.part"))
    assert not manager.reservations


async def test_shutdown_awaits_already_cancelled_cleanup(manager, model):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    shutdown_started = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self) -> None:
            cleaning.set()
            await release.wait()

    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=SlowStream())
    )
    job = manager.start(model.id, ".")
    await entered.wait()
    cancellation = asyncio.create_task(manager.cancel(job.id))
    await cleaning.wait()

    async def shutdown():
        shutdown_started.set()
        await manager.shutdown()

    shutdown_task = asyncio.create_task(shutdown())
    await shutdown_started.wait()
    assert not shutdown_task.done()
    assert job.task.cancelling() == 1
    release.set()
    await cancellation
    await shutdown_task
    assert not manager.reservations
    assert not list(job.destination.rglob("*.part"))


async def test_shutdown_cancels_all_jobs_before_awaiting_cleanup(manager, model):
    entered = [asyncio.Event(), asyncio.Event()]
    closing = [asyncio.Event(), asyncio.Event()]
    streams = iter(range(2))

    class CoordinatedStream(httpx.AsyncByteStream):
        def __init__(self, index: int):
            self.index = index

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * (1024 * 1024)
            entered[self.index].set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            closing[self.index].set()
            await asyncio.gather(*(event.wait() for event in closing))

    manager.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=CoordinatedStream(next(streams)))
    )
    jobs = [manager.start(model.id, destination) for destination in ("first", "second")]
    await asyncio.gather(*(event.wait() for event in entered))
    await asyncio.wait_for(manager.shutdown(), timeout=5)
    assert all(job.status == "cancelled" and job.task.done() for job in jobs)
    assert not manager.reservations
    assert not list(manager.settings.workspace_root.rglob("*.part"))


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute",
        "a\\b",
        "C:drive",
        "a:ads",
        "a/../b",
        "./a",
        "a//b",
        "a/",
        "a\0b",
    ],
)
def test_download_path_validation(path):
    with pytest.raises(ValidationError):
        DownloadFile(url=MODEL_URL, path=path)


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/test",
        "https://example.com/test",
        "https://huggingface.co.evil.test/test",
        "https://user:secret@huggingface.co/test",
        "https://huggingface.co:443/test",
        "https://huggingface.co/test?token=secret",
        "https://huggingface.co/test#fragment",
    ],
)
def test_download_url_validation(url):
    with pytest.raises(ValidationError):
        DownloadFile(url=url, path=MODEL_PATH)


def test_model_rejects_empty_files_and_duplicate_paths():
    file = DownloadFile(url=MODEL_URL, path=MODEL_PATH)
    with pytest.raises(ValidationError):
        DownloadModel(id="bad", name="Bad", files=[])
    with pytest.raises(ValidationError):
        DownloadModel(id="bad", name="Bad", files=[file, file])


def test_catalog_sanity():
    expected = {
        "flux2-dev",
        "flux2-klein-base-4b",
        "flux2-klein-base-9b",
        "flux-kontext-dev",
        "framepack",
        "hidream-o1",
        "hunyuan-video",
        "hunyuan-video-1-5",
        "ideogram-4",
        "kandinsky-5-t2v",
        "krea-2",
        "qwen-image-bf16",
        "qwen-image-fp8",
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "wan-2-1-t2v-14b",
        "wan-2-1-i2v-14b",
        "wan-2-2-t2v-14b",
        "wan-2-2-i2v-14b",
        "z-image-base",
        "z-image-turbo",
        "minimax-h3-fl2va-bf16",
        "minimax-h3-fl2va-int8",
        "minimax-h3-ref2va-bf16",
        "minimax-h3-ref2va-int8",
    }
    assert set(DOWNLOAD_MODELS) == expected
    for model_id, model in DOWNLOAD_MODELS.items():
        assert model_id == model.id
        assert DownloadModel.model_validate(model.model_dump()) == model
        assert all(
            file.url.startswith("https://huggingface.co/") for file in model.files
        )
