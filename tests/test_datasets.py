import io
import tomllib

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
