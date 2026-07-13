import asyncio
import io
import json
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from .conftest import wait_for

STUB_SCRIPT = """
import sys
import time

stage = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "ok"
print(f"{stage} starting", flush=True)
if mode == "fail" and stage == "cache_latents":
    print("boom failure", flush=True)
    sys.exit(1)
if mode == "leak_fail" and stage == "cache_latents":
    print(f"token was {sys.argv[3]}", flush=True)
    sys.exit(1)
if mode == "slow" and stage == "train":
    time.sleep(120)
if stage == "train":
    print("epoch 1/2", flush=True)
    print("steps:  50%|#####     | 5/10 [00:01<00:01,  5.00it/s]", flush=True)
    print("epoch 2/2", flush=True)
    print("steps: 100%|##########| 10/10 [00:02<00:00,  5.00it/s]", flush=True)
print(f"{stage} done", flush=True)
"""

JOB_VALUES = {
    "musubiPath": "./musubi-tuner",
    "task": "t2v-A14B",
    "dit": "./diffusion_models/low.safetensors",
    "ditHigh": "./diffusion_models/high.safetensors",
    "vae": "./vae/wan_2.1_vae.safetensors",
    "t5": "./text_encoders/umt5.pth",
    "outputName": "char_v3",
    "outputDir": "./lora_training/outputs",
}

DATASET_PAYLOAD = {
    "name": "queue-dataset",
    "general": {"resolution": [512, 512], "caption_extension": ".txt"},
    "datasets": [
        {"image_directory": "/data/images", "cache_directory": "/cache/images"}
    ],
}


@pytest.fixture
def training_client(client, paths, monkeypatch, tmp_path):
    musubi_path = paths["workspace"] / "musubi-tuner"
    musubi_path.mkdir()
    from app.profiles import TRAINING_PROFILES

    profile = TRAINING_PROFILES["wan-22"]
    for filename in (
        profile.cache_commands[0].script,
        profile.cache_commands[1].script,
        profile.trainer,
    ):
        (musubi_path / filename).touch()
    stub = tmp_path / "stage_stub.py"
    stub.write_text(STUB_SCRIPT)

    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *args, **kwargs: (
            "accelerate" if name == "accelerate" else real_which(name, *args, **kwargs)
        ),
    )

    from app import queue_runner

    def fake_build(profile, values, musubi_path, dataset_config_path):
        mode = values.get("stubMode", "ok")
        return {
            key: [
                sys.executable,
                str(stub),
                key,
                mode,
                *([values["huggingfaceToken"]] if mode == "leak_fail" else []),
            ]
            for key in ("cache_latents", "cache_text_encoder", "train")
        }

    monkeypatch.setattr(queue_runner, "build_stage_argv", fake_build)
    return client


def make_dataset(client):
    response = client.post("/api/datasets", json=DATASET_PAYLOAD)
    assert response.status_code == 201
    return response.json()["id"]


def make_job(
    client, dataset_id, name="job", stub_mode="ok", skip_cache=False, values=None
):
    response = client.post(
        "/api/training/jobs",
        json={
            "name": name,
            "profile_id": "wan-22",
            "dataset_config_id": dataset_id,
            "skip_cache_stages": skip_cache,
            "values": {**JOB_VALUES, "stubMode": stub_mode, **(values or {})},
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def get_job(client, job_id):
    return client.get(f"/api/training/jobs/{job_id}").json()


def test_job_creation_rechecks_dataset_after_concurrent_managed_delete(
    training_client, paths, monkeypatch
):
    managed_response = training_client.post(
        "/api/datasets/managed",
        data={
            "name": "racing managed dataset",
            "media_type": "image",
            "resolution": json.dumps([512, 512]),
            "captions": json.dumps(["caption"]),
        },
        files={"files": ("sample.png", io.BytesIO(b"image"), "image/png")},
    )
    assert managed_response.status_code == 201
    dataset_id = managed_response.json()["id"]

    from app import training

    real_insert_job = training._insert_job
    reached_insert = threading.Event()
    allow_insert = threading.Event()

    async def delayed_insert_job(*args, **kwargs):
        reached_insert.set()
        await asyncio.to_thread(allow_insert.wait)
        return await real_insert_job(*args, **kwargs)

    monkeypatch.setattr(training, "_insert_job", delayed_insert_job)
    request_body = {
        "name": "racing job",
        "profile_id": "wan-22",
        "dataset_config_id": dataset_id,
        "values": JOB_VALUES,
    }

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            training_client.post, "/api/training/jobs", json=request_body
        )
        assert reached_insert.wait(timeout=5)
        deleted = training_client.delete(f"/api/datasets/{dataset_id}")
        allow_insert.set()
        created = future.result(timeout=10)

    assert deleted.status_code == 204
    assert created.status_code == 404
    assert created.json()["detail"] == "Unknown dataset config"
    assert training_client.get("/api/training/jobs").json() == []
    assert list(paths["data"].joinpath("jobs").iterdir()) == []


def test_create_validations(training_client):
    dataset_id = make_dataset(training_client)

    unknown_profile = training_client.post(
        "/api/training/jobs",
        json={
            "name": "x",
            "profile_id": "nope",
            "dataset_config_id": dataset_id,
            "values": {},
        },
    )
    assert unknown_profile.status_code == 404

    unknown_dataset = training_client.post(
        "/api/training/jobs",
        json={
            "name": "x",
            "profile_id": "wan-22",
            "dataset_config_id": "missing",
            "values": JOB_VALUES,
        },
    )
    assert unknown_dataset.status_code == 404

    missing_fields = training_client.post(
        "/api/training/jobs",
        json={
            "name": "x",
            "profile_id": "wan-22",
            "dataset_config_id": dataset_id,
            "values": {**JOB_VALUES, "dit": ""},
        },
    )
    assert missing_fields.status_code == 422

    bad_extra_args = training_client.post(
        "/api/training/jobs",
        json={
            "name": "x",
            "profile_id": "wan-22",
            "dataset_config_id": dataset_id,
            "values": {**JOB_VALUES, "extraArgs": "--fine; rm -rf /"},
        },
    )
    assert bad_extra_args.status_code == 400

    escaping_path = training_client.post(
        "/api/training/jobs",
        json={
            "name": "x",
            "profile_id": "wan-22",
            "dataset_config_id": dataset_id,
            "values": {**JOB_VALUES, "outputDir": "../../outside"},
        },
    )
    assert escaping_path.status_code == 400

    missing_musubi = training_client.post(
        "/api/training/jobs",
        json={
            "name": "x",
            "profile_id": "wan-22",
            "dataset_config_id": dataset_id,
            "values": {**JOB_VALUES, "musubiPath": "./missing-musubi"},
        },
    )
    assert missing_musubi.status_code == 503


def test_jobs_queue_until_started_then_run_fifo(training_client):
    dataset_id = make_dataset(training_client)
    first = make_job(training_client, dataset_id, name="first")
    second = make_job(training_client, dataset_id, name="second")

    assert first["status"] == "queued"
    assert first["queue_position"] == 0
    assert second["queue_position"] == 1
    assert [stage["status"] for stage in first["stages"]] == [
        "pending",
        "pending",
        "pending",
    ]

    queue = training_client.get("/api/training/queue").json()
    assert queue == {"state": "paused", "queued": 2, "running_job_id": None}

    # Nothing runs while the queue is paused.
    time.sleep(1.0)
    assert get_job(training_client, first["id"])["status"] == "queued"

    started = training_client.post("/api/training/queue/start")
    assert started.status_code == 200
    assert started.json()["state"] == "running"

    done_first = wait_for(
        lambda: (
            (job := get_job(training_client, first["id"]))["status"] == "completed"
            and job
        )
    )
    done_second = wait_for(
        lambda: (
            (job := get_job(training_client, second["id"]))["status"] == "completed"
            and job
        )
    )
    assert done_first["finished_at"] <= done_second["started_at"]
    assert [stage["status"] for stage in done_first["stages"]] == [
        "completed",
        "completed",
        "completed",
    ]
    assert done_first["progress"]["percent"] == 100
    assert done_first["progress"]["epoch"] == 2
    assert done_first["progress"]["total_epochs"] == 2
    assert done_first["progress"]["step"] == 10

    logs = training_client.get(
        f"/api/training/jobs/{first['id']}/logs", params={"offset": 0}
    )
    body = logs.json()
    assert "train done" in body["content"]
    assert body["eof"] is True
    assert body["next_offset"] > 0

    paused = training_client.post("/api/training/queue/pause")
    assert paused.json()["state"] == "paused"


def test_job_lists_and_downloads_all_lora_artifacts(training_client, paths):
    dataset_id = make_dataset(training_client)
    job = make_job(training_client, dataset_id, name="artifact job")

    pending = training_client.get(f"/api/training/jobs/{job['id']}/artifacts")
    assert pending.status_code == 200
    assert pending.json() == {"files": [], "total_size_bytes": 0}

    training_client.post("/api/training/queue/start")
    wait_for(lambda: get_job(training_client, job["id"])["status"] == "completed")

    output_dir = paths["workspace"] / "lora_training" / "outputs"
    output_dir.mkdir(parents=True)
    expected = {
        "char_v3-000001.safetensors": b"epoch-one",
        "char_v3-000002.safetensors": b"epoch-two",
        "char_v3.safetensors": b"final",
    }
    for name, content in expected.items():
        (output_dir / name).write_bytes(content)
    (output_dir / "char_v30-000001.safetensors").write_bytes(b"another job")
    (output_dir / "char_v3-000002.json").write_text("{}")

    listed = training_client.get(f"/api/training/jobs/{job['id']}/artifacts")
    assert listed.status_code == 200
    assert listed.json() == {
        "files": [
            {"name": name, "size_bytes": len(content)}
            for name, content in sorted(expected.items())
        ],
        "total_size_bytes": sum(map(len, expected.values())),
    }

    downloaded = training_client.get(
        f"/api/training/jobs/{job['id']}/artifacts/download"
    )
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/zip"
    assert (
        f"char_v3-{job['id'][:8]}-loras.zip"
        in downloaded.headers["content-disposition"]
    )
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert archive.namelist() == sorted(expected)
        assert {name: archive.read(name) for name in archive.namelist()} == expected


def test_running_job_exposes_completed_epoch_artifacts(training_client, paths):
    dataset_id = make_dataset(training_client)
    job = make_job(
        training_client,
        dataset_id,
        name="running artifacts",
        stub_mode="slow",
    )
    training_client.post("/api/training/queue/start")
    wait_for(
        lambda: (
            (data := get_job(training_client, job["id"]))["status"] == "running"
            and data["current_stage"] == "train"
        )
    )

    output_dir = paths["workspace"] / "lora_training" / "outputs"
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "char_v3-000001.safetensors"
    checkpoint.write_bytes(b"epoch-one")

    listed = training_client.get(f"/api/training/jobs/{job['id']}/artifacts")
    assert listed.status_code == 200
    assert listed.json() == {
        "files": [{"name": checkpoint.name, "size_bytes": len(b"epoch-one")}],
        "total_size_bytes": len(b"epoch-one"),
    }

    downloaded = training_client.get(
        f"/api/training/jobs/{job['id']}/artifacts/download"
    )
    assert downloaded.status_code == 200
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        assert archive.namelist() == [checkpoint.name]
        assert archive.read(checkpoint.name) == b"epoch-one"

    assert (
        training_client.post(f"/api/training/jobs/{job['id']}/cancel").status_code
        == 200
    )


def test_completed_job_with_no_lora_artifacts_has_no_download(training_client):
    dataset_id = make_dataset(training_client)
    job = make_job(training_client, dataset_id)
    training_client.post("/api/training/queue/start")
    wait_for(lambda: get_job(training_client, job["id"])["status"] == "completed")

    listed = training_client.get(f"/api/training/jobs/{job['id']}/artifacts")
    assert listed.json() == {"files": [], "total_size_bytes": 0}
    downloaded = training_client.get(
        f"/api/training/jobs/{job['id']}/artifacts/download"
    )
    assert downloaded.status_code == 404
    assert downloaded.json()["detail"] == "No LoRA artifacts found for this job"


def test_skip_cache_stages(training_client):
    dataset_id = make_dataset(training_client)
    job = make_job(training_client, dataset_id, skip_cache=True)
    assert [stage["status"] for stage in job["stages"]] == [
        "skipped",
        "skipped",
        "pending",
    ]

    training_client.post("/api/training/queue/start")
    done = wait_for(
        lambda: (
            (data := get_job(training_client, job["id"]))["status"] == "completed"
            and data
        )
    )
    assert [stage["status"] for stage in done["stages"]] == [
        "skipped",
        "skipped",
        "completed",
    ]


def test_failed_stage_marks_job_failed_and_retry_clones(training_client, paths):
    dataset_id = make_dataset(training_client)
    job = make_job(training_client, dataset_id, stub_mode="fail")
    training_client.post("/api/training/queue/start")

    failed = wait_for(
        lambda: (
            (data := get_job(training_client, job["id"]))["status"] == "failed" and data
        )
    )
    assert [stage["status"] for stage in failed["stages"]] == [
        "failed",
        "skipped",
        "skipped",
    ]
    assert "boom failure" in failed["error"]

    retried = training_client.post(f"/api/training/jobs/{job['id']}/retry")
    assert retried.status_code == 202
    clone = retried.json()
    assert clone["id"] != job["id"]
    wait_for(lambda: get_job(training_client, clone["id"])["status"] == "failed")

    # Terminal jobs can be deleted; their log file goes too.
    log_path = paths["data"] / "jobs" / f"{job['id']}.log"
    assert log_path.is_file()
    assert training_client.delete(f"/api/training/jobs/{job['id']}").status_code == 204
    assert not log_path.exists()
    assert training_client.get(f"/api/training/jobs/{job['id']}").status_code == 404


def test_cancel_running_job(training_client):
    dataset_id = make_dataset(training_client)
    slow = make_job(training_client, dataset_id, name="slow", stub_mode="slow")
    training_client.post("/api/training/queue/start")

    wait_for(
        lambda: (
            (data := get_job(training_client, slow["id"]))["status"] == "running"
            and data["current_stage"] == "train"
        )
    )
    cancelled = training_client.post(f"/api/training/jobs/{slow['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    stages = {stage["key"]: stage["status"] for stage in cancelled.json()["stages"]}
    assert stages["train"] == "cancelled"

    # Idempotent for an already-cancelled job.
    assert (
        training_client.post(f"/api/training/jobs/{slow['id']}/cancel").status_code
        == 200
    )

    # The runner moves on to the next job afterwards.
    follow_up = make_job(training_client, dataset_id, name="after-cancel")
    wait_for(lambda: get_job(training_client, follow_up["id"])["status"] == "completed")


def test_cancel_queued_job_and_completed_conflict(training_client):
    dataset_id = make_dataset(training_client)
    job = make_job(training_client, dataset_id)
    cancelled = training_client.post(f"/api/training/jobs/{job['id']}/cancel")
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["queue_position"] is None

    done = make_job(training_client, dataset_id, name="to-complete")
    training_client.post("/api/training/queue/start")
    wait_for(lambda: get_job(training_client, done["id"])["status"] == "completed")
    assert (
        training_client.post(f"/api/training/jobs/{done['id']}/cancel").status_code
        == 409
    )
    assert (
        training_client.post(f"/api/training/jobs/{done['id']}/retry").status_code
        == 409
    )


def test_reorder_and_delete_rules(training_client):
    dataset_id = make_dataset(training_client)
    jobs = [
        make_job(training_client, dataset_id, name=f"job-{index}") for index in range(3)
    ]

    reordered = training_client.patch(
        f"/api/training/jobs/{jobs[2]['id']}", json={"queue_position": 0}
    )
    assert reordered.status_code == 200
    assert reordered.json()["queue_position"] == 0
    listed = training_client.get(
        "/api/training/jobs", params={"status": "queued"}
    ).json()
    positions = {job["name"]: job["queue_position"] for job in listed}
    assert positions == {"job-2": 0, "job-0": 1, "job-1": 2}

    # Queued jobs cannot be deleted.
    assert (
        training_client.delete(f"/api/training/jobs/{jobs[0]['id']}").status_code == 409
    )

    cancelled = training_client.post(
        f"/api/training/jobs/{jobs[1]['id']}/cancel"
    ).json()
    assert cancelled["status"] == "cancelled"
    # Cancelling renumbers the remaining queue.
    listed = training_client.get(
        "/api/training/jobs", params={"status": "queued"}
    ).json()
    positions = {job["name"]: job["queue_position"] for job in listed}
    assert positions == {"job-2": 0, "job-0": 1}
    # Terminal jobs cannot be reordered.
    assert (
        training_client.patch(
            f"/api/training/jobs/{jobs[1]['id']}", json={"queue_position": 0}
        ).status_code
        == 409
    )


def test_huggingface_token_never_echoed(training_client):
    dataset_id = make_dataset(training_client)
    job = make_job(
        training_client, dataset_id, values={"huggingfaceToken": "hf_supersecret"}
    )
    assert "hf_supersecret" not in json.dumps(job)
    fetched = training_client.get(f"/api/training/jobs/{job['id']}")
    assert "hf_supersecret" not in fetched.text
    listed = training_client.get("/api/training/jobs")
    assert "hf_supersecret" not in listed.text


def test_huggingface_token_is_redacted_from_logs_and_errors(training_client):
    dataset_id = make_dataset(training_client)
    job = make_job(
        training_client,
        dataset_id,
        stub_mode="leak_fail",
        values={"huggingfaceToken": "hf_supersecret"},
    )
    training_client.post("/api/training/queue/start")
    failed = wait_for(
        lambda: (
            (data := get_job(training_client, job["id"]))["status"] == "failed" and data
        )
    )
    logs = training_client.get(f"/api/training/jobs/{job['id']}/logs").json()["content"]
    assert "hf_supersecret" not in failed["error"]
    assert "hf_supersecret" not in logs
    assert "token was ***" in logs


def test_queue_state_persists_across_restart(make_client):
    with make_client() as first:
        assert first.get("/api/training/queue").json()["state"] == "paused"
        first.post("/api/training/queue/start")
    with make_client() as second:
        assert second.get("/api/training/queue").json()["state"] == "running"


async def test_recover_interrupted_jobs(tmp_path):
    from app.db import Database, utc_now
    from app.queue_runner import recover_interrupted_jobs

    db = await Database.open(tmp_path / "recover.db")
    stages = [
        {"key": "cache_latents", "status": "completed"},
        {"key": "cache_text_encoder", "status": "running"},
        {"key": "train", "status": "pending"},
    ]
    await db.execute(
        "INSERT INTO training_jobs (id, name, profile_id, dataset_config_toml, values_json, "
        "status, stages_json, created_at) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
        ("j1", "interrupted", "wan-22", "", "{}", json.dumps(stages), utc_now()),
    )
    await recover_interrupted_jobs(db)
    row = await db.fetch_one("SELECT * FROM training_jobs WHERE id = 'j1'")
    assert row["status"] == "failed"
    assert row["error"] == "Interrupted by server restart"
    assert [stage["status"] for stage in json.loads(row["stages_json"])] == [
        "completed",
        "failed",
        "skipped",
    ]
    await db.close()


async def test_runner_does_not_claim_a_cancelled_or_paused_job(tmp_path):
    from app.config import Settings
    from app.db import Database, utc_now
    from app.queue_runner import QueueRunner

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db = await Database.open(tmp_path / "claim.db")
    stages = [
        {"key": key, "status": "pending"}
        for key in ("cache_latents", "cache_text_encoder", "train")
    ]
    await db.execute(
        "INSERT INTO training_jobs (id, name, profile_id, dataset_config_toml, values_json, "
        "status, queue_position, stages_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'cancelled', NULL, ?, ?)",
        ("cancelled", "cancelled", "wan-22", "", "{}", json.dumps(stages), utc_now()),
    )
    await db.execute(
        "INSERT INTO training_jobs (id, name, profile_id, dataset_config_toml, values_json, "
        "status, queue_position, stages_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?)",
        ("paused", "paused", "wan-22", "", "{}", json.dumps(stages), utc_now()),
    )
    runner = QueueRunner(
        db,
        Settings(
            data_root=tmp_path,
            scripts_dir=tmp_path,
            workspace_root=workspace,
            web_dir=tmp_path / "web",
            cors_origins=(),
        ),
    )

    runner.queue_running.set()
    assert not await runner._claim_job("cancelled", tmp_path / "cancelled.log")
    runner.queue_running.clear()
    assert not await runner._claim_job("paused", tmp_path / "paused.log")
    statuses = await db.fetch_all("SELECT id, status FROM training_jobs ORDER BY id")
    assert {row["id"]: row["status"] for row in statuses} == {
        "cancelled": "cancelled",
        "paused": "queued",
    }
    await db.close()
