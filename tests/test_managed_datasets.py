import io
import json
import tomllib
from pathlib import Path

import pytest


def _managed_request(
    client,
    *,
    name="uploaded dataset",
    media_type="image",
    resolution=(512, 768),
    captions=("first caption",),
    uploads=(("sample.png", b"image bytes", "image/png"),),
    caption_uploads=(),
    control_uploads=(),
    target_frames=None,
    headers=None,
):
    data = {
        "name": name,
        "description": "created from uploads",
        "media_type": media_type,
        "resolution": json.dumps(resolution),
    }
    if captions is not None:
        data["captions"] = json.dumps(captions)
    if target_frames is not None:
        data["target_frames"] = json.dumps(target_frames)
    files = [
        ("files", (filename, io.BytesIO(content), content_type))
        for filename, content, content_type in uploads
    ]
    files.extend(
        ("caption_files", (filename, io.BytesIO(content), content_type))
        for filename, content, content_type in caption_uploads
    )
    files.extend(
        ("control_files", (filename, io.BytesIO(content), content_type))
        for filename, content, content_type in control_uploads
    )
    return client.post("/api/datasets/managed", data=data, files=files, headers=headers)


def _managed_batch_request(client, *, name="multi dataset", specs=None, files=None):
    specs = specs or [
        {
            "media_type": "image",
            "resolution": [512, 512],
            "num_repeats": 3,
            "captions": ["still image"],
            "file_count": 1,
            "caption_file_count": 0,
            "control_file_count": 1,
        },
        {
            "media_type": "video",
            "resolution": [960, 544],
            "num_repeats": 2,
            "target_frames": [1, 25, 49],
            "captions": ["moving subject"],
            "file_count": 1,
            "caption_file_count": 0,
            "control_file_count": 0,
        },
    ]
    files = files or [
        ("files", ("still.png", io.BytesIO(b"image"), "image/png")),
        ("files", ("motion.mp4", io.BytesIO(b"video"), "video/mp4")),
        ("control_files", ("still.jpg", io.BytesIO(b"control"), "image/jpeg")),
    ]
    return client.post(
        "/api/datasets/managed/batch",
        data={
            "name": name,
            "description": "multiple TOML entries",
            "dataset_specs": json.dumps(specs),
        },
        files=files,
    )


def test_create_managed_batch_writes_multiple_datasets_and_repeats(client, paths):
    response = _managed_batch_request(client)

    assert response.status_code == 201, response.text
    created = response.json()
    assert created["general"] == {"caption_extension": ".txt"}
    assert len(created["datasets"]) == 2

    managed_dir = paths["data"] / "managed_datasets" / created["id"]
    image_dataset, video_dataset = created["datasets"]
    assert image_dataset["resolution"] == [512, 512]
    assert image_dataset["num_repeats"] == 3
    assert image_dataset["image_directory"] == str(
        (managed_dir / "dataset-1" / "media").resolve()
    )
    assert image_dataset["control_directory"] == str(
        (managed_dir / "dataset-1" / "control").resolve()
    )
    assert video_dataset["resolution"] == [960, 544]
    assert video_dataset["num_repeats"] == 2
    assert video_dataset["target_frames"] == [1, 25, 49]
    assert video_dataset["video_directory"] == str(
        (managed_dir / "dataset-2" / "media").resolve()
    )
    assert (managed_dir / "dataset-1" / "media" / "still.txt").read_text(
        encoding="utf-8"
    ) == "still image"
    assert (managed_dir / "dataset-2" / "media" / "motion.txt").read_text(
        encoding="utf-8"
    ) == "moving subject"

    document = tomllib.loads(
        (managed_dir / "dataset_config.toml").read_text(encoding="utf-8")
    )
    assert len(document["datasets"]) == 2
    assert [dataset["num_repeats"] for dataset in document["datasets"]] == [3, 2]

    assert client.delete(f"/api/datasets/{created['id']}").status_code == 204
    assert not managed_dir.exists()


@pytest.mark.parametrize(
    ("spec_override", "detail"),
    [
        ({"num_repeats": 0}, "num_repeats must be a positive integer"),
        ({"file_count": 2}, "file counts do not match"),
    ],
)
def test_managed_batch_rejects_invalid_specs(client, spec_override, detail):
    spec = {
        "media_type": "image",
        "resolution": [512, 512],
        "num_repeats": 1,
        "captions": ["caption"],
        "file_count": 1,
        "caption_file_count": 0,
        "control_file_count": 0,
        **spec_override,
    }
    response = _managed_batch_request(
        client,
        specs=[spec],
        files=[("files", ("sample.png", io.BytesIO(b"image"), "image/png"))],
    )

    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_create_managed_image_dataset_writes_media_captions_and_export(client, paths):
    response = _managed_request(
        client,
        captions=("close portrait", "full body"),
        uploads=(
            ("../portrait.JPG", b"jpeg bytes", "image/jpeg"),
            ("portrait.png", b"png bytes", "image/png"),
        ),
    )

    assert response.status_code == 201, response.text
    created = response.json()
    assert created["warnings"] == []
    assert created["general"] == {
        "resolution": [512, 768],
        "caption_extension": ".txt",
    }

    managed_dir = paths["data"] / "managed_datasets" / created["id"]
    media_dir = managed_dir / "media"
    cache_dir = managed_dir / "cache"
    assert cache_dir.is_dir()
    assert (media_dir / "portrait.jpg").read_bytes() == b"jpeg bytes"
    assert (media_dir / "portrait.txt").read_text(encoding="utf-8") == "close portrait"
    assert (media_dir / "portrait-2.png").read_bytes() == b"png bytes"
    assert (media_dir / "portrait-2.txt").read_text(encoding="utf-8") == "full body"

    saved_document = tomllib.loads(
        (managed_dir / "dataset_config.toml").read_text(encoding="utf-8")
    )
    assert saved_document["general"]["resolution"] == [512, 768]
    assert saved_document["datasets"][0]["image_directory"] == str(media_dir.resolve())

    dataset = created["datasets"][0]
    assert dataset == {
        "image_directory": str(media_dir.resolve()),
        "cache_directory": str(cache_dir.resolve()),
    }
    assert Path(dataset["image_directory"]).is_absolute()

    exported = client.get(f"/api/datasets/{created['id']}/export")
    assert exported.status_code == 200
    document = tomllib.loads(exported.text)
    assert document["general"]["caption_extension"] == ".txt"
    assert document["datasets"][0]["image_directory"] == str(media_dir.resolve())


def test_caption_sidecar_is_matched_by_stem_and_saved_beside_image(client, paths):
    response = _managed_request(
        client,
        name="sidecar captions",
        captions=None,
        uploads=(("a.PNG", b"target", "image/png"),),
        caption_uploads=(("a.txt", b"caption from sidecar\n", "text/plain"),),
    )

    assert response.status_code == 201, response.text
    managed_dir = paths["data"] / "managed_datasets" / response.json()["id"]
    assert (managed_dir / "media" / "a.png").read_bytes() == b"target"
    assert (managed_dir / "media" / "a.txt").read_text(encoding="utf-8") == (
        "caption from sidecar"
    )


def test_caption_sidecar_overrides_matching_manual_caption(client, paths):
    response = _managed_request(
        client,
        captions=("manual caption",),
        caption_uploads=(("sample.TXT", b"sidecar caption", "text/plain"),),
    )

    assert response.status_code == 201, response.text
    managed_dir = paths["data"] / "managed_datasets" / response.json()["id"]
    assert (managed_dir / "media" / "sample.txt").read_text(encoding="utf-8") == (
        "sidecar caption"
    )


@pytest.mark.parametrize(
    ("caption_uploads", "uploads", "detail"),
    [
        (
            (("missing.txt", b"caption", "text/plain"),),
            (("sample.png", b"image", "image/png"),),
            "has no matching media file",
        ),
        (
            (
                ("sample.txt", b"one", "text/plain"),
                ("SAMPLE.TXT", b"two", "text/plain"),
            ),
            (("sample.png", b"image", "image/png"),),
            "More than one caption sidecar",
        ),
        (
            (("sample.txt", b"caption", "text/plain"),),
            (
                ("sample.png", b"one", "image/png"),
                ("sample.jpg", b"two", "image/jpeg"),
            ),
            "matches more than one media file",
        ),
    ],
)
def test_caption_sidecar_rejects_unmatched_duplicate_or_ambiguous_stems(
    client, caption_uploads, uploads, detail
):
    response = _managed_request(
        client,
        captions=None,
        uploads=uploads,
        caption_uploads=caption_uploads,
    )

    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_control_image_upload_adds_control_directory_and_preserves_pairing(
    client, paths
):
    response = _managed_request(
        client,
        name="controlled image",
        uploads=(("a.jpg", b"target", "image/jpeg"),),
        caption_uploads=(("a.txt", b"turn it blue", "text/plain"),),
        control_uploads=(("a.png", b"control", "image/png"),),
    )

    assert response.status_code == 201, response.text
    created = response.json()
    managed_dir = paths["data"] / "managed_datasets" / created["id"]
    control_dir = managed_dir / "control"
    assert (control_dir / "a.png").read_bytes() == b"control"
    assert created["datasets"][0]["control_directory"] == str(control_dir.resolve())
    document = tomllib.loads(
        (managed_dir / "dataset_config.toml").read_text(encoding="utf-8")
    )
    assert document["datasets"][0]["control_directory"] == str(control_dir.resolve())


def test_multiple_numbered_control_images_follow_musubi_naming(client, paths):
    response = _managed_request(
        client,
        name="multiple controls",
        control_uploads=(
            ("sample_0.png", b"first", "image/png"),
            ("sample_0001.jpg", b"second", "image/jpeg"),
        ),
    )

    assert response.status_code == 201, response.text
    managed_dir = paths["data"] / "managed_datasets" / response.json()["id"]
    assert (managed_dir / "control" / "sample_0.png").read_bytes() == b"first"
    assert (managed_dir / "control" / "sample_0001.jpg").read_bytes() == b"second"


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        (
            {"control_uploads": (("other.png", b"control", "image/png"),)},
            "has no matching target image",
        ),
        (
            {
                "uploads": (
                    ("a.png", b"a", "image/png"),
                    ("b.png", b"b", "image/png"),
                ),
                "captions": ("a", "b"),
                "control_uploads": (("a.png", b"control", "image/png"),),
            },
            "Every target image needs a control image",
        ),
        (
            {
                "media_type": "video",
                "uploads": (("sample.mp4", b"video", "video/mp4"),),
                "control_uploads": (("sample.png", b"control", "image/png"),),
            },
            "only for image datasets",
        ),
    ],
)
def test_control_images_reject_unmatched_incomplete_or_video_uploads(
    client, overrides, detail
):
    response = _managed_request(client, **overrides)

    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_create_managed_video_dataset_uses_sampling_defaults(client, paths):
    response = _managed_request(
        client,
        name="video upload",
        media_type="video",
        resolution=(960, 544),
        captions=("walking",),
        uploads=(("walk.MP4", b"video bytes", "video/mp4"),),
    )

    assert response.status_code == 201, response.text
    created = response.json()
    managed_dir = paths["data"] / "managed_datasets" / created["id"]
    dataset = created["datasets"][0]
    assert dataset == {
        "video_directory": str((managed_dir / "media").resolve()),
        "cache_directory": str((managed_dir / "cache").resolve()),
        "target_frames": [1],
        "frame_extraction": "head",
    }
    assert (managed_dir / "media" / "walk.mp4").read_bytes() == b"video bytes"
    assert (managed_dir / "media" / "walk.txt").read_text(encoding="utf-8") == "walking"


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"name": "   "}, "name must not be empty"),
        ({"media_type": "audio"}, "media_type must be"),
        ({"resolution": (512,)}, "resolution must be"),
        ({"resolution": (512, 0)}, "resolution must be"),
        ({"captions": ()}, "exactly one entry"),
        ({"captions": ("   ",)}, "whitespace-only"),
        (
            {"uploads": (("sample.mp4", b"video", "video/mp4"),)},
            "Unsupported image file extension",
        ),
        (
            {
                "media_type": "video",
                "uploads": (("sample.png", b"image", "image/png"),),
            },
            "Unsupported video file extension",
        ),
        (
            {
                "media_type": "video",
                "uploads": (("sample.mp4", b"video", "video/mp4"),),
                "target_frames": [],
            },
            "target_frames must be",
        ),
        ({"target_frames": [1]}, "target_frames only applies"),
        (
            {"uploads": (("no-extension", b"data", "application/octet-stream"),)},
            "Unsupported image file extension",
        ),
    ],
)
def test_managed_dataset_rejects_invalid_inputs(client, overrides, detail):
    response = _managed_request(client, **overrides)
    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_managed_dataset_rejects_invalid_json_and_missing_files(client):
    invalid_json = client.post(
        "/api/datasets/managed",
        data={
            "name": "bad json",
            "media_type": "image",
            "resolution": "not json",
            "captions": "[]",
        },
        files={"files": ("sample.png", io.BytesIO(b"image"), "image/png")},
    )
    assert invalid_json.status_code == 422
    assert "resolution must be valid JSON" in invalid_json.json()["detail"]

    missing_files = client.post(
        "/api/datasets/managed",
        data={
            "name": "no files",
            "media_type": "image",
            "resolution": "[512, 512]",
            "captions": "[]",
        },
    )
    assert missing_files.status_code == 422


def test_managed_dataset_upload_quotas_return_413_and_clean_up(
    make_client, paths, monkeypatch
):
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_FILES", "1")
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_FILE_BYTES", "4")
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_TOTAL_BYTES", "8")
    with make_client() as quota_client:
        too_many = _managed_request(
            quota_client,
            captions=("one", "two"),
            uploads=(
                ("one.png", b"1", "image/png"),
                ("two.png", b"2", "image/png"),
            ),
        )
        assert too_many.status_code == 413

        too_large = _managed_request(
            quota_client,
            uploads=(("large.png", b"12345", "image/png"),),
        )
        assert too_large.status_code == 413
        assert "4-byte limit" in too_large.json()["detail"]
        assert list((paths["data"] / "managed_datasets").iterdir()) == []

    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_FILES", "3")
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_FILE_BYTES", "10")
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_TOTAL_BYTES", "6")
    with make_client() as quota_client:
        too_large_total = _managed_request(
            quota_client,
            name="total quota",
            captions=("one", "two"),
            uploads=(
                ("one.png", b"1234", "image/png"),
                ("two.png", b"5678", "image/png"),
            ),
        )
        assert too_large_total.status_code == 413
        assert "6-byte total limit" in too_large_total.json()["detail"]
        assert list((paths["data"] / "managed_datasets").iterdir()) == []


def test_managed_request_content_length_is_rejected_before_parsing(
    make_client, monkeypatch
):
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_REQUEST_BYTES", "10")
    with make_client() as limited_client:
        response = _managed_request(
            limited_client,
            headers={"Origin": "http://localhost:5173"},
        )
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "request body" in response.json()["detail"]


@pytest.mark.parametrize(
    "request_path", ["/api/datasets/managed", "/api/datasets/managed/batch"]
)
async def test_managed_request_chunked_body_is_counted_before_app_consumes_it(
    request_path,
):
    from app.main import ManagedUploadBodyLimitMiddleware, _ManagedBodyTooLarge

    assert issubclass(_ManagedBodyTooLarge, OSError)

    sent = []
    incoming = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ]
    )

    async def receive():
        return next(incoming)

    async def send(message):
        sent.append(message)

    async def consuming_app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass

    middleware = ManagedUploadBodyLimitMiddleware(consuming_app, max_bytes=6)
    await middleware(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": request_path,
            "raw_path": request_path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["detail"].startswith(
        "Managed dataset request body"
    )


def test_managed_aggregate_storage_quota_counts_existing_datasets(
    make_client, paths, monkeypatch
):
    with make_client() as first_client:
        first = _managed_request(first_client, name="first aggregate")
        assert first.status_code == 201

    managed_root = paths["data"] / "managed_datasets"
    existing_bytes = sum(
        path.stat().st_size for path in managed_root.rglob("*") if path.is_file()
    )
    monkeypatch.setenv("MUSUBI_GUI_MANAGED_MAX_STORAGE_BYTES", str(existing_bytes + 1))
    with make_client() as limited_client:
        second = _managed_request(limited_client, name="second aggregate")
        assert second.status_code == 413
        assert "storage quota" in second.json()["detail"]
    assert (
        len([path for path in managed_root.iterdir() if not path.name.startswith(".")])
        == 1
    )


def test_managed_dataset_cleans_directory_when_database_insert_fails(
    client, paths, monkeypatch
):
    from app import datasets

    async def no_name_collision(*args, **kwargs):
        return False

    async def insert_collision(*args, **kwargs):
        return False

    monkeypatch.setattr(datasets, "_name_taken", no_name_collision)
    monkeypatch.setattr(datasets, "_try_dataset_write", insert_collision)

    response = _managed_request(client)

    assert response.status_code == 409
    assert list((paths["data"] / "managed_datasets").iterdir()) == []


def test_delete_managed_dataset_removes_only_its_owned_directory(client, paths):
    managed = _managed_request(client).json()
    managed_dir = paths["data"] / "managed_datasets" / managed["id"]
    assert managed_dir.is_dir()

    assert client.delete(f"/api/datasets/{managed['id']}").status_code == 204
    assert not managed_dir.exists()

    external_dir = paths["workspace"] / "external-images"
    external_dir.mkdir()
    external_file = external_dir / "keep.png"
    external_file.write_bytes(b"keep me")
    external = client.post(
        "/api/datasets",
        json={
            "name": "external dataset",
            "general": {"resolution": [512, 512], "caption_extension": ".txt"},
            "datasets": [{"image_directory": str(external_dir)}],
        },
    ).json()

    assert client.delete(f"/api/datasets/{external['id']}").status_code == 204
    assert external_file.read_bytes() == b"keep me"


def test_delete_managed_dataset_leaves_retriable_tombstone_when_cleanup_fails(
    client, paths, monkeypatch
):
    from app import datasets

    managed = _managed_request(client).json()
    real_rmtree = datasets.shutil.rmtree

    def fail_cleanup(path):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(datasets.shutil, "rmtree", fail_cleanup)
    response = client.delete(f"/api/datasets/{managed['id']}")
    assert response.status_code == 500
    assert "tombstone cleanup is pending" in response.json()["detail"]
    assert client.get(f"/api/datasets/{managed['id']}").status_code == 404
    tombstones = list((paths["data"] / "managed_datasets").glob(".tombstone-*"))
    assert len(tombstones) == 1
    assert (tombstones[0] / "media" / "sample.png").is_file()

    monkeypatch.setattr(datasets.shutil, "rmtree", real_rmtree)
    created = _managed_request(client, name="cleanup retry")
    assert created.status_code == 201
    assert not tombstones[0].exists()


def test_managed_creation_cleanup_failure_leaves_orphan_for_retry(
    client, paths, monkeypatch
):
    from app import datasets

    real_insert = datasets._try_dataset_write
    real_rmtree = datasets.shutil.rmtree

    async def insertion_failure(*args, **kwargs):
        return False

    def cleanup_failure(path):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(datasets, "_try_dataset_write", insertion_failure)
    monkeypatch.setattr(datasets.shutil, "rmtree", cleanup_failure)
    failed = _managed_request(client, name="orphaned create")
    assert failed.status_code == 500
    assert "cleanup is pending" in failed.json()["detail"]
    orphans = list((paths["data"] / "managed_datasets").glob(".orphan-*"))
    assert len(orphans) == 1

    monkeypatch.setattr(datasets, "_try_dataset_write", real_insert)
    monkeypatch.setattr(datasets.shutil, "rmtree", real_rmtree)
    retried = _managed_request(client, name="after orphan cleanup")
    assert retried.status_code == 201
    assert not orphans[0].exists()


def test_managed_delete_commit_failure_restores_directory_and_config(
    client, paths, monkeypatch
):
    from app import datasets

    managed = _managed_request(client).json()
    managed_dir = paths["data"] / "managed_datasets" / managed["id"]
    real_commit = datasets._commit_managed_delete

    async def fail_commit(db):
        raise OSError("simulated commit failure")

    monkeypatch.setattr(datasets, "_commit_managed_delete", fail_commit)
    failed = client.delete(f"/api/datasets/{managed['id']}")
    assert failed.status_code == 500
    assert "restored" in failed.json()["detail"]
    assert client.get(f"/api/datasets/{managed['id']}").status_code == 200
    assert (managed_dir / "media" / "sample.png").is_file()
    assert list((paths["data"] / "managed_datasets").glob(".tombstone-*")) == []

    monkeypatch.setattr(datasets, "_commit_managed_delete", real_commit)
    assert client.delete(f"/api/datasets/{managed['id']}").status_code == 204


def test_startup_cleans_stale_managed_tombstones(make_client, paths):
    stale = (
        paths["data"]
        / "managed_datasets"
        / ".tombstone-00000000-0000-0000-0000-000000000001-cleanup"
    )
    stale.mkdir(parents=True)
    (stale / "leftover.bin").write_bytes(b"leftover")
    with make_client():
        assert not stale.exists()


def test_startup_restores_pending_delete_when_config_rollback_survived(
    make_client, paths, monkeypatch
):
    from app import datasets

    real_commit = datasets._commit_managed_delete
    real_rename = datasets._rename_managed_directory

    async def fail_commit(db):
        raise OSError("simulated commit failure")

    def fail_restore(source, destination):
        if source.name.startswith(".pending-delete-"):
            raise OSError("simulated restore failure")
        return real_rename(source, destination)

    with make_client() as first_client:
        managed = _managed_request(first_client, name="pending restore").json()
        monkeypatch.setattr(datasets, "_commit_managed_delete", fail_commit)
        monkeypatch.setattr(datasets, "_rename_managed_directory", fail_restore)
        failed = first_client.delete(f"/api/datasets/{managed['id']}")
        assert failed.status_code == 500
        assert first_client.get(f"/api/datasets/{managed['id']}").status_code == 200
        pending = list((paths["data"] / "managed_datasets").glob(".pending-delete-*"))
        assert len(pending) == 1
        assert (pending[0] / "media" / "sample.png").is_file()

    monkeypatch.setattr(datasets, "_commit_managed_delete", real_commit)
    monkeypatch.setattr(datasets, "_rename_managed_directory", real_rename)
    with make_client() as restarted:
        assert restarted.get(f"/api/datasets/{managed['id']}").status_code == 200
        restored = paths["data"] / "managed_datasets" / managed["id"]
        assert (restored / "media" / "sample.png").is_file()
        assert (
            list((paths["data"] / "managed_datasets").glob(".pending-delete-*")) == []
        )


def test_update_managed_dataset_keeps_owned_toml_in_sync(client, paths):
    managed = _managed_request(client).json()
    managed_dir = paths["data"] / "managed_datasets" / managed["id"]
    dataset = managed["datasets"][0]
    payload = {
        "name": "updated managed dataset",
        "description": "updated description",
        "general": {"resolution": [1024, 576], "caption_extension": ".txt"},
        "datasets": [{**dataset, "num_repeats": 2}],
    }

    updated = client.put(f"/api/datasets/{managed['id']}", json=payload)
    assert updated.status_code == 200, updated.text
    saved = tomllib.loads(
        (managed_dir / "dataset_config.toml").read_text(encoding="utf-8")
    )
    assert saved["general"]["resolution"] == [1024, 576]
    assert saved["datasets"][0]["num_repeats"] == 2
    assert (
        tomllib.loads(client.get(f"/api/datasets/{managed['id']}/export").text) == saved
    )

    changed_source = {
        **payload,
        "datasets": [{**dataset, "image_directory": str(paths["workspace"] / "other")}],
    }
    rejected = client.put(f"/api/datasets/{managed['id']}", json=changed_source)
    assert rejected.status_code == 422
    assert "source path" in rejected.json()["detail"]
    assert (
        tomllib.loads((managed_dir / "dataset_config.toml").read_text(encoding="utf-8"))
        == saved
    )


def test_delete_managed_dataset_is_blocked_while_training_job_references_it(
    client, paths
):
    managed = _managed_request(client).json()
    managed_dir = paths["data"] / "managed_datasets" / managed["id"]

    async def insert_referencing_job():
        await client.app.state.db.execute(
            "INSERT INTO training_jobs "
            "(id, name, profile_id, dataset_config_id, dataset_config_toml, values_json, "
            "status, stages_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "managed-reference-job",
                "managed reference",
                "wan-22",
                managed["id"],
                (managed_dir / "dataset_config.toml").read_text(encoding="utf-8"),
                "{}",
                "cancelled",
                "[]",
                "2026-07-13T00:00:00.000000Z",
            ),
        )

    client.portal.call(insert_referencing_job)

    blocked = client.delete(f"/api/datasets/{managed['id']}")
    assert blocked.status_code == 409
    assert "referencing training jobs" in blocked.json()["detail"]
    assert client.get(f"/api/datasets/{managed['id']}").status_code == 200
    assert (managed_dir / "media" / "sample.png").is_file()

    assert client.delete("/api/training/jobs/managed-reference-job").status_code == 204
    assert client.delete(f"/api/datasets/{managed['id']}").status_code == 204
    assert not managed_dir.exists()
