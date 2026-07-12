import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GIT_BASH_DIR = Path(r"C:\Program Files\Git\bin")


@pytest.fixture
def paths(tmp_path):
    layout = {
        "data": tmp_path / "data",
        "scripts": tmp_path / "scripts",
        "workspace": tmp_path / "workspace",
    }
    layout["scripts"].mkdir()
    layout["workspace"].mkdir()
    return layout


@pytest.fixture
def make_client(paths, monkeypatch):
    monkeypatch.setenv("MUSUBI_GUI_DATA_ROOT", str(paths["data"]))
    monkeypatch.setenv("MUSUBI_GUI_SCRIPTS_DIR", str(paths["scripts"]))
    monkeypatch.setenv("MUSUBI_GUI_WORKSPACE_ROOT", str(paths["workspace"]))
    monkeypatch.setenv("MUSUBI_GUI_WEB_DIR", str(paths["workspace"] / "no-web"))
    if os.name == "nt" and (GIT_BASH_DIR / "bash.exe").is_file():
        # The WSL bash stub in system32 has no distro; prefer Git Bash.
        monkeypatch.setenv("PATH", f"{GIT_BASH_DIR}{os.pathsep}{os.environ['PATH']}")

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
