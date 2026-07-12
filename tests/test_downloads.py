import asyncio
import shutil

import pytest

from app.config import Settings
from app.downloads import DownloadManager

from .conftest import wait_for

FAST_SCRIPT = """#!/usr/bin/env bash
echo "Downloading FLUX.2 dev DiT"
echo "diffusion_models/flux2_dev_fp8mixed.safetensors 50%"
echo "done 100%"
"""

SLOW_SCRIPT = """#!/usr/bin/env bash
echo "starting"
sleep 30
"""

FAILING_SCRIPT = """#!/usr/bin/env bash
echo "something went wrong"
exit 3
"""


@pytest.fixture(autouse=True)
def hf_cli_available(monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *args, **kwargs: (
            "hf" if name == "hf" else real_which(name, *args, **kwargs)
        ),
    )


def install_script(paths, filename, body):
    script = paths["scripts"] / filename
    script.write_text(body, newline="\n")
    return script


def test_unknown_script_id_returns_404(client):
    response = client.post(
        "/api/downloads", json={"script_id": "nope", "destination": "."}
    )
    assert response.status_code == 404


def test_missing_script_file_returns_503(client):
    response = client.post(
        "/api/downloads", json={"script_id": "flux2-dev", "destination": "."}
    )
    assert response.status_code == 503


def test_missing_hf_cli_returns_503(client, paths, monkeypatch):
    install_script(paths, "download_flux2_dev.sh", FAST_SCRIPT)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *args, **kwargs: (
            None if name == "hf" else real_which(name, *args, **kwargs)
        ),
    )
    response = client.post(
        "/api/downloads", json={"script_id": "flux2-dev", "destination": "."}
    )
    assert response.status_code == 503
    assert "hf CLI" in response.json()["detail"]


def test_destination_outside_root_returns_400(client, paths):
    install_script(paths, "download_flux2_dev.sh", FAST_SCRIPT)
    response = client.post(
        "/api/downloads", json={"script_id": "flux2-dev", "destination": "../escape"}
    )
    assert response.status_code == 400


def test_download_completes_with_progress(client, paths):
    install_script(paths, "download_flux2_dev.sh", FAST_SCRIPT)
    response = client.post(
        "/api/downloads", json={"script_id": "flux2-dev", "destination": "."}
    )
    assert response.status_code == 202
    job = response.json()
    assert job["script_id"] == "flux2-dev"
    assert job["status"] in {"queued", "running"}

    done = wait_for(
        lambda: (
            (data := client.get(f"/api/downloads/{job['id']}").json())["status"]
            == "completed"
            and data
        )
    )
    assert done["progress"] == 100
    assert done["current_file"] == "diffusion_models/flux2_dev_fp8mixed.safetensors"


def test_failed_download_reports_error(client, paths):
    install_script(paths, "download_flux2_dev.sh", FAILING_SCRIPT)
    job = client.post(
        "/api/downloads", json={"script_id": "flux2-dev", "destination": "."}
    ).json()
    failed = wait_for(
        lambda: (
            (data := client.get(f"/api/downloads/{job['id']}").json())["status"]
            == "failed"
            and data
        )
    )
    assert "something went wrong" in failed["error"]

    cancel = client.post(f"/api/downloads/{job['id']}/cancel")
    assert cancel.status_code == 409


def test_duplicate_active_job_conflicts_and_cancel_is_idempotent(client, paths):
    install_script(paths, "download_framepack.sh", SLOW_SCRIPT)
    first = client.post(
        "/api/downloads", json={"script_id": "framepack", "destination": "."}
    )
    assert first.status_code == 202
    job = first.json()
    wait_for(
        lambda: client.get(f"/api/downloads/{job['id']}").json()["status"] == "running"
    )

    duplicate = client.post(
        "/api/downloads", json={"script_id": "framepack", "destination": "."}
    )
    assert duplicate.status_code == 409

    cancelled = client.post(f"/api/downloads/{job['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    again = client.post(f"/api/downloads/{job['id']}/cancel")
    assert again.status_code == 200
    assert again.json()["status"] == "cancelled"

    # The slot is free once the job is no longer active.
    retry = client.post(
        "/api/downloads", json={"script_id": "framepack", "destination": "."}
    )
    assert retry.status_code == 202
    client.post(f"/api/downloads/{retry.json()['id']}/cancel")


def test_unknown_job_returns_404(client):
    assert client.get("/api/downloads/does-not-exist").status_code == 404


@pytest.mark.asyncio
async def test_cancel_while_process_is_spawning_stays_cancelled(paths, monkeypatch):
    install_script(paths, "download_flux2_dev.sh", FAST_SCRIPT)
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()
    terminated = asyncio.Event()

    class FakeProcess:
        returncode = None

    process = FakeProcess()

    async def delayed_spawn(*args, **kwargs):
        spawn_started.set()
        await release_spawn.wait()
        return process

    async def fake_terminate(received):
        assert received is process
        received.returncode = -1
        terminated.set()

    from app import downloads

    monkeypatch.setattr(downloads, "spawn", delayed_spawn)
    monkeypatch.setattr(downloads, "terminate_tree", fake_terminate)
    monkeypatch.setattr(
        downloads.shutil,
        "which",
        lambda name: name if name in {"bash", "hf"} else None,
    )
    manager = DownloadManager(
        Settings(
            data_root=paths["data"],
            scripts_dir=paths["scripts"],
            workspace_root=paths["workspace"],
            web_dir=paths["workspace"] / "web",
            cors_origins=(),
        )
    )

    job = manager.start("flux2-dev", ".")
    await spawn_started.wait()
    await manager.cancel(job.id)
    release_spawn.set()
    assert job.task is not None
    await job.task

    assert terminated.is_set()
    assert job.status == "cancelled"
