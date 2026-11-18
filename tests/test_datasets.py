import io
import tomllib
from pathlib import Path

import pytest

VALID_PAYLOAD = {
    "name": "my-character-video",
    "description": "WAN 2.2 character dataset",
    "general": {
        "resolution": [960, 544],
        "caption_extension": ".txt",
        "batch_size": 1,
        "enable_bucket": True,
        "bucket_no_upscale": False,
    },
    "datasets": [
        {
            "image_directory": "/workspace/data/char_images",
            "cache_directory": "/workspace/cache/char_images",
            "num_repeats": 2,
        },
        {
            "video_directory": "/workspace/data/char_videos",
            "cache_directory": "/workspace/cache/char_videos",
            "target_frames": [1, 25, 45],
            "frame_extraction": "head",
            "source_fps": 30.0,
        },
    ],
}


def create(client, payload=None):
    response = client.post("/api/datasets", json=payload or VALID_PAYLOAD)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_read_update_delete(client):
    created = create(client)
    assert created["warnings"] == []
    assert created["general"]["resolution"] == [960, 544]

    listed = client.get("/api/datasets").json()
    assert [item["name"] for item in listed] == ["my-character-video"]

    read = client.get(f"/api/datasets/{created['id']}").json()
    assert read["datasets"][1]["target_frames"] == [1, 25, 45]

    updated_payload = {**VALID_PAYLOAD, "name": "renamed", "description": "v2"}
    updated = client.put(f"/api/datasets/{created['id']}", json=updated_payload)
    assert updated.status_code == 200
    assert updated.json()["name"] == "renamed"

    assert client.delete(f"/api/datasets/{created['id']}").status_code == 204
    assert client.get(f"/api/datasets/{created['id']}").status_code == 404


def test_duplicate_name_returns_409(client, monkeypatch):
    create(client)

    # The database constraint must still become a 409 if a concurrent request
    # slips past the advisory name check.
    async def stale_name_check(*args, **kwargs):
        return False

    from app import datasets

    monkeypatch.setattr(datasets, "_name_taken", stale_name_check)
    response = client.post("/api/datasets", json=VALID_PAYLOAD)
    assert response.status_code == 409


def test_validation_errors_return_422(client):
    cases = [
        {**VALID_PAYLOAD, "datasets": []},
        {**VALID_PAYLOAD, "datasets": [{"num_repeats": 1}]},  # no source key
        {
            **VALID_PAYLOAD,
            "datasets": [{"image_directory": "/a", "image_jsonl_file": "/b.jsonl"}],
        },  # two source keys
        {
            **VALID_PAYLOAD,
            "datasets": [{"image_jsonl_file": "/a.jsonl"}],
        },  # jsonl without cache_directory
        {
            **VALID_PAYLOAD,
            "datasets": [{"image_jsonl_file": "", "cache_directory": ""}],
        },  # required paths must be non-empty strings
        {
            **VALID_PAYLOAD,
            "datasets": [{"image_directory": "/a", "future_option": None}],
        },  # JSON values must be representable in TOML
        {
            **VALID_PAYLOAD,
            "datasets": [
                {
                    "image_directory": "/a",
                    "cache_directory": "/c",
                    "num_repeats": 0,
                }
            ],
        },  # repeats must be positive
        {
            **VALID_PAYLOAD,
            "general": {},
            "datasets": [{"image_directory": "/a", "caption_extension": ".txt"}],
        },  # resolution nowhere
        {
            **VALID_PAYLOAD,
            "datasets": [
                {
                    "video_directory": "/v",
                    "frame_extraction": "sideways",
                    "cache_directory": "/c",
                }
            ],
        },  # bad frame_extraction
        {
            **VALID_PAYLOAD,
            "datasets": [{"video_directory": "/v", "cache_directory": "/c"}],
        },  # target_frames required unless full
        {
            **VALID_PAYLOAD,
            "datasets": [
                {
                    "video_directory": "/v",
                    "cache_directory": "/c",
                    "target_frames": [25],
                    "source_fps": "thirty",
                }
            ],
        },  # source_fps not a number
        {
            **VALID_PAYLOAD,
            "datasets": [
                {"image_directory": "/a", "cache_directory": "/same"},
                {"image_directory": "/b", "cache_directory": "/same"},
            ],
        },  # duplicate cache_directory
    ]
    for payload in cases:
        response = client.post("/api/datasets", json=payload)
        assert response.status_code == 422, payload


def test_unknown_keys_become_warnings(client):
    payload = {
        **VALID_PAYLOAD,
        "datasets": [
            {
                "video_directory": "/v",
                "cache_directory": "/c",
                "target_frames": [24],
                "frame_extractoin": "head",
            }
        ],
    }
    created = create(client, payload)
    warnings = created["warnings"]
    assert any("unknown key 'frame_extractoin'" in warning for warning in warnings)
    assert any("not N*4+1" in warning for warning in warnings)
    # Preserved, not dropped.
    assert created["datasets"][0]["frame_extractoin"] == "head"


def test_import_export_round_trip(client):
    toml_body = """
future_format_version = 2

[general]
resolution = [960, 544]
caption_extension = ".txt"

[[datasets]]
video_directory = "/workspace/data/char_videos"
cache_directory = "/workspace/cache/char_videos"
target_frames = [1, 25]
source_fps = 30
"""
    response = client.post(
        "/api/datasets/import",
        files={
            "file": (
                "char videos.toml",
                io.BytesIO(toml_body.encode()),
                "application/toml",
            )
        },
    )
    assert response.status_code == 201, response.text
    imported = response.json()
    assert imported["name"] == "char videos"
    assert any("future_format_version" in warning for warning in imported["warnings"])

    # Same filename again: deduplicated with a numeric suffix.
    response = client.post(
        "/api/datasets/import",
        files={
            "file": (
                "char videos.toml",
                io.BytesIO(toml_body.encode()),
                "application/toml",
            )
        },
    )
    assert response.json()["name"] == "char videos-2"

    export = client.get(f"/api/datasets/{imported['id']}/export")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/toml")
    assert 'filename="char-videos.toml"' in export.headers["content-disposition"]
    # source_fps must render as a TOML float.
    assert "source_fps = 30.0" in export.text
    round_tripped = tomllib.loads(export.text)
    assert round_tripped["future_format_version"] == 2
    assert round_tripped["datasets"][0]["target_frames"] == [1, 25]


def test_import_invalid_toml_returns_400(client):
    response = client.post(
        "/api/datasets/import",
        files={"file": ("bad.toml", io.BytesIO(b"not = [valid"), "application/toml")},
    )
    assert response.status_code == 400


def test_import_rule_violation_returns_422(client):
    response = client.post(
        "/api/datasets/import",
        files={"file": ("empty.toml", io.BytesIO(b"[general]\n"), "application/toml")},
    )
    assert response.status_code == 422


def test_validate_checks_filesystem(client, paths):
    image_dir = paths["workspace"] / "images"
    image_dir.mkdir()
    (image_dir / "a.png").write_bytes(b"png")
    (image_dir / "b.jpg").write_bytes(b"jpg")
    (image_dir / "a.txt").write_text("caption")

    payload = {
        "name": "fs-check",
        "general": {"resolution": [512, 512], "caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(image_dir),
                "cache_directory": str(paths["workspace"] / "cache" / "images"),
            },
            {
                "video_directory": str(paths["workspace"] / "missing"),
                "cache_directory": str(paths["workspace"] / "cache" / "videos"),
                "target_frames": [1],
            },
        ],
    }
    created = create(client, payload)
    result = client.post(f"/api/datasets/{created['id']}/validate").json()

    first, second = result["datasets"]
    assert first["exists"] is True
    assert first["image_count"] == 2
    assert first["caption_count"] == 1
    assert first["cache_directory_creatable"] is True
    assert second["exists"] is False


def test_validate_resolves_relative_paths_against_the_workspace(
    client, paths, monkeypatch
):
    # The process cwd must not be the workspace, or a relative path would pass
    # even without being resolved against the workspace root.
    monkeypatch.chdir(paths["workspace"].parent)
    image_dir = paths["workspace"] / "images"
    image_dir.mkdir()
    (image_dir / "a.png").write_bytes(b"png")
    (image_dir / "a.txt").write_text("caption")

    payload = {
        "name": "relative-check",
        "general": {"resolution": [512, 512], "caption_extension": ".txt"},
        "datasets": [
            {"image_directory": "images", "cache_directory": "cache/images"},
            {
                "video_directory": "missing",
                "cache_directory": "cache/videos",
                "target_frames": [1],
            },
        ],
    }
    created = create(client, payload)
    result = client.post(f"/api/datasets/{created['id']}/validate").json()

    first, second = result["datasets"]
    assert first["exists"] is True
    assert first["image_count"] == 1
    assert first["caption_count"] == 1
    # The raw values are echoed back, not the resolved ones.
    assert first["source_path"] == "images"
    assert first["cache_directory"] == "cache/images"
    assert first["cache_directory_creatable"] is True
    assert second["exists"] is False
    assert second["cache_directory"] == "cache/videos"


def test_resolve_path_is_unconfined_while_resolve_inside_workspace_confines(tmp_path):
    from app.config import Settings

    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    outside = (tmp_path / "outside").resolve()
    settings = Settings(
        data_root=tmp_path / "data",
        workspace_root=workspace,
        web_dir=tmp_path / "web",
    )

    assert settings.resolve_path("a/../b") == workspace / "b"
    assert settings.resolve_path(str(outside)) == outside
    assert settings.resolve_path("../outside") == outside
    assert settings.resolve_inside_workspace("a/../b") == workspace / "b"
    assert settings.resolve_inside_workspace(".") == workspace
    for raw in (str(outside), "../outside"):
        with pytest.raises(ValueError) as error:
            settings.resolve_inside_workspace(raw)
        assert str(error.value) == (
            f"Path is outside the configured workspace root: {raw}"
        )


def _media_folder(parent, name="photos", stems=("a", "b"), suffix=".png", caption=True):
    folder = parent / name
    folder.mkdir()
    for stem in stems:
        (folder / f"{stem}{suffix}").write_bytes(b"media")
        if caption:
            (folder / f"{stem}.txt").write_text(f"{stem} caption", encoding="utf-8")
    return folder


def _from_folder(client, path, **overrides):
    payload = {
        "name": "from folder",
        "path": str(path),
        "resolution": [512, 768],
        **overrides,
    }
    return client.post("/api/datasets/from-folder", json=payload)


def test_from_folder_creates_a_dataset_pointing_at_the_folder(client, paths):
    folder = _media_folder(paths["workspace"])

    response = _from_folder(
        client, folder, description="local photos", num_repeats=3, batch_size=2
    )

    assert response.status_code == 201, response.text
    created = response.json()
    assert created["media_count"] == 2
    assert created["warnings"] == []
    assert created["description"] == "local photos"
    assert created["general"] == {
        "caption_extension": ".txt",
        "batch_size": 2,
        "enable_bucket": True,
        "bucket_no_upscale": False,
    }
    assert created["datasets"] == [
        {
            "image_directory": str(folder.resolve()),
            "cache_directory": str(folder.resolve() / "cache"),
            "resolution": [512, 768],
            "num_repeats": 3,
        }
    ]
    stored = client.get(f"/api/datasets/{created['id']}").json()
    assert stored["datasets"] == created["datasets"]
    listed = client.get("/api/datasets").json()
    assert [item["id"] for item in listed] == [created["id"]]
    # The folder is only referenced; nothing is created inside it.
    assert not (folder / "cache").exists()


def test_from_folder_resolves_a_relative_path_under_the_workspace(
    client, paths, monkeypatch
):
    monkeypatch.chdir(paths["workspace"].parent)
    folder = _media_folder(paths["workspace"], name="relative-photos")

    response = _from_folder(client, "relative-photos")

    assert response.status_code == 201, response.text
    dataset = response.json()["datasets"][0]
    assert dataset["image_directory"] == str(folder.resolve())
    assert dataset["cache_directory"] == str(folder.resolve() / "cache")


def test_from_folder_rejects_a_missing_folder(client, paths):
    not_a_folder = paths["workspace"] / "file.txt"
    not_a_folder.write_text("not a folder")

    for path in (paths["workspace"] / "nowhere", not_a_folder):
        response = _from_folder(client, path)
        assert response.status_code == 422
        assert response.json()["detail"].startswith("Folder not found on the server: ")

    assert client.get("/api/datasets").json() == []


def test_from_folder_rejects_a_folder_without_top_level_media(client, paths):
    folder = paths["workspace"] / "empty"
    folder.mkdir()
    (folder / "notes.txt").write_text("not media")
    nested = folder / "nested"
    nested.mkdir()
    (nested / "deep.png").write_bytes(b"png")
    photos = _media_folder(paths["workspace"])

    response = _from_folder(client, folder)
    assert response.status_code == 422
    assert response.json()["detail"] == f"No image files found in {folder.resolve()}"

    # Image files do not count towards a video dataset.
    response = _from_folder(client, photos, media_type="video")
    assert response.status_code == 422
    assert response.json()["detail"] == f"No video files found in {photos.resolve()}"
    assert client.get("/api/datasets").json() == []


def test_from_folder_requires_a_caption_for_every_media_file(client, paths):
    folder = _media_folder(paths["workspace"], stems=("a", "b", "c"))
    (folder / "b.txt").unlink()
    (folder / "c.txt").unlink()

    response = _from_folder(client, folder)

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "2 image file(s) have no .txt caption: b.png, c.png"
    )
    assert client.get("/api/datasets").json() == []


def test_from_folder_names_at_most_twenty_uncaptioned_files(client, paths):
    stems = [f"image{index:02d}" for index in range(25)]
    folder = _media_folder(paths["workspace"], stems=stems, caption=False)

    response = _from_folder(client, folder)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail.startswith("25 image file(s) have no .txt caption: image00.png, ")
    assert "image19.png" in detail
    assert "image20.png" not in detail


def test_from_folder_video_defaults_to_single_frame_head_extraction(client, paths):
    folder = _media_folder(
        paths["workspace"], name="clips", stems=("clip",), suffix=".MP4"
    )

    response = _from_folder(client, folder, media_type="video", resolution=[960, 544])

    assert response.status_code == 201, response.text
    created = response.json()
    assert created["media_count"] == 1
    assert created["datasets"] == [
        {
            "video_directory": str(folder.resolve()),
            "cache_directory": str(folder.resolve() / "cache"),
            "resolution": [960, 544],
            "num_repeats": 1,
            "target_frames": [1],
            "frame_extraction": "head",
        }
    ]

    explicit = _from_folder(
        client,
        folder,
        name="explicit frames",
        media_type="video",
        resolution=[960, 544],
        target_frames=[1, 25],
    )
    assert explicit.status_code == 201, explicit.text
    assert explicit.json()["datasets"][0]["target_frames"] == [1, 25]


def test_from_folder_rejects_target_frames_for_an_image_dataset(client, paths):
    folder = _media_folder(paths["workspace"])

    response = _from_folder(client, folder, target_frames=[1])

    assert response.status_code == 422
    assert response.json()["detail"] == "target_frames only applies to video datasets"
    assert client.get("/api/datasets").json() == []


def test_from_folder_name_must_be_unique_and_non_empty(client, paths):
    first = _media_folder(paths["workspace"], name="first")
    second = _media_folder(paths["workspace"], name="second")
    assert _from_folder(client, first).status_code == 201

    taken = _from_folder(client, second)
    assert taken.status_code == 409
    assert (
        taken.json()["detail"] == "A dataset config named 'from folder' already exists"
    )

    blank = _from_folder(client, second, name="   ")
    assert blank.status_code == 422
    assert blank.json()["detail"] == "name must not be empty"
    assert len(client.get("/api/datasets").json()) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": [512]},
        {"resolution": [512, 512, 512]},
        {"resolution": [0, 512]},
        {"num_repeats": 0},
        {"batch_size": 0},
        {"media_type": "audio"},
        {"media_type": "video", "target_frames": []},
        {"media_type": "video", "target_frames": [0]},
    ],
)
def test_from_folder_rejects_invalid_options(client, paths, overrides):
    folder = _media_folder(paths["workspace"])

    response = _from_folder(client, folder, **overrides)

    assert response.status_code == 422
    assert client.get("/api/datasets").json() == []


def test_from_folder_rejects_an_empty_path(client):
    response = _from_folder(client, "")

    assert response.status_code == 422
    assert client.get("/api/datasets").json() == []


def test_from_folder_reports_an_unreadable_folder(client, paths, monkeypatch):
    folder = _media_folder(paths["workspace"])

    def unreadable(self):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "iterdir", unreadable)
    response = _from_folder(client, folder)

    assert response.status_code == 422
    assert response.json()["detail"] == (
        f"Folder cannot be read on the server: {folder.resolve()}"
    )
