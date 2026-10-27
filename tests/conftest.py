import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def paths(tmp_path):
    layout = {
        "data": tmp_path / "data",
        "workspace": tmp_path / "workspace",
    }
    layout["workspace"].mkdir()
    return layout


@pytest.fixture
def make_client(paths, monkeypatch):
    monkeypatch.setenv("MUSUBI_GUI_DATA_ROOT", str(paths["data"]))
    monkeypatch.setenv("MUSUBI_GUI_WORKSPACE_ROOT", str(paths["workspace"]))
    monkeypatch.setenv("MUSUBI_GUI_WEB_DIR", str(paths["workspace"] / "no-web"))

    from app.main import create_app

    return lambda: TestClient(create_app())


@pytest.fixture
def client(make_client):
    with make_client() as test_client:
        yield test_client


def wait_for(predicate, timeout=30.0, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError("Timed out waiting for condition")
