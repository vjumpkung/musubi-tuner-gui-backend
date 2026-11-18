import sys

import pytest

from app.utils import process


@pytest.fixture
def captured(monkeypatch):
    calls: dict = {}

    async def fake_exec(*argv, **kwargs):
        calls["argv"] = list(argv)
        calls["kwargs"] = kwargs
        return "process"

    monkeypatch.setattr(process.asyncio, "create_subprocess_exec", fake_exec)
    return calls


async def test_spawn_resolves_bare_name_via_path(monkeypatch, tmp_path, captured):
    resolved = str(tmp_path / "venv" / "python")
    monkeypatch.setattr(
        process.shutil,
        "which",
        lambda name, *args, **kwargs: resolved if name == "python" else None,
    )

    result = await process.spawn(["python", "script.py", "--flag"], tmp_path)

    assert result == "process"
    assert captured["argv"] == [resolved, "script.py", "--flag"]
    assert captured["kwargs"]["cwd"] == str(tmp_path)


async def test_spawn_raises_when_bare_name_not_found(monkeypatch, tmp_path, captured):
    monkeypatch.setattr(process.shutil, "which", lambda name, *args, **kwargs: None)

    with pytest.raises(OSError, match="accelerate was not found on PATH"):
        await process.spawn(["accelerate", "launch"], tmp_path)

    assert "argv" not in captured


async def test_spawn_leaves_absolute_path_untouched(monkeypatch, tmp_path, captured):
    def fail_which(name, *args, **kwargs):
        raise AssertionError("shutil.which must not be called for absolute paths")

    monkeypatch.setattr(process.shutil, "which", fail_which)

    await process.spawn([sys.executable, "-c", "pass"], tmp_path)

    assert captured["argv"] == [sys.executable, "-c", "pass"]
