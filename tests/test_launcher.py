import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from app import launcher


class FakeServer:
    """Stands in for uvicorn.Server: run() blocks until should_exit is set."""

    def __init__(self, *, starts: bool = True):
        self.starts = starts
        self.started = False
        self.should_exit = False
        self.stopped = False
        self.run_thread: str | None = None

    def run(self) -> None:
        self.run_thread = threading.current_thread().name
        if self.starts:
            self.started = True
            deadline = time.monotonic() + 5
            while not self.should_exit and time.monotonic() < deadline:
                time.sleep(0.01)
        self.stopped = True


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setenv("MUSUBI_GUI_DATA_ROOT", str(data_root))
    monkeypatch.setenv("MUSUBI_GUI_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.delenv("MUSUBI_GUI_PASSWORD", raising=False)
    monkeypatch.delenv("MUSUBI_GUI_PORT", raising=False)
    return SimpleNamespace(data_root=data_root)


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(launcher, "_make_server", lambda host, port: fake)
    return fake


@pytest.fixture
def opened(monkeypatch, server):
    """Records URLs passed to the browser opener and then stops the fake server."""
    urls: list[str] = []

    def fake_open(url: str) -> None:
        urls.append(url)
        server.should_exit = True

    monkeypatch.setattr(launcher, "_open_browser", fake_open)
    return urls


def test_port_in_use_returns_1(env, monkeypatch, capsys):
    def unexpected(host, port):
        raise AssertionError("server must not be created when the port is busy")

    monkeypatch.setattr(launcher, "_make_server", unexpected)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]

        assert launcher.main(["--browser", "--port", str(port)]) == 1

    err = capsys.readouterr().err
    assert f"Port {port} is already in use" in err
    assert "--port" in err


def test_browser_mode_serves_in_main_thread_and_opens_browser(env, server, opened):
    port = free_port()

    assert launcher.main(["--browser", "--port", str(port)]) == 0

    assert opened == [f"http://127.0.0.1:{port}/"]
    assert server.run_thread == threading.main_thread().name
    assert server.stopped


def test_browser_mode_swallows_keyboard_interrupt(env, monkeypatch, server):
    def interrupted() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "run", interrupted)
    monkeypatch.setattr(launcher, "_open_browser", lambda url: None)

    assert launcher.main(["--browser", "--port", str(free_port())]) == 0


def test_window_mode_runs_window_with_storage_under_data_root(
    env, monkeypatch, server, capsys
):
    port = free_port()
    calls: list[tuple[str, str]] = []

    def fake_window(url: str, storage_path: str) -> bool:
        calls.append((url, storage_path))
        return True

    def no_browser(url: str) -> None:
        raise AssertionError("browser must not open when the window is shown")

    monkeypatch.setattr(launcher, "_run_window", fake_window)
    monkeypatch.setattr(launcher, "_open_browser", no_browser)

    assert launcher.main(["--port", str(port)]) == 0

    assert calls == [(f"http://127.0.0.1:{port}/", str(env.data_root / "webview"))]
    assert server.run_thread != threading.main_thread().name
    assert server.should_exit
    assert server.stopped


def test_window_mode_falls_back_to_browser_without_pywebview(
    env, monkeypatch, server, opened, capsys
):
    port = free_port()
    monkeypatch.setitem(sys.modules, "webview", None)

    assert launcher.main(["--port", str(port)]) == 0

    assert opened == [f"http://127.0.0.1:{port}/"]
    assert server.stopped
    assert "pywebview is not installed" in capsys.readouterr().out


def test_window_mode_reports_server_that_fails_to_start(env, monkeypatch, capsys):
    failing = FakeServer(starts=False)
    monkeypatch.setattr(launcher, "_make_server", lambda host, port: failing)

    def unexpected(*args):
        raise AssertionError("no window or browser when the server is down")

    monkeypatch.setattr(launcher, "_run_window", unexpected)
    monkeypatch.setattr(launcher, "_open_browser", unexpected)

    assert launcher.main(["--port", str(free_port())]) == 1

    assert "failed to start" in capsys.readouterr().out


def test_non_loopback_host_without_password_warns(env, server, opened, capsys):
    port = free_port()

    assert launcher.main(["--browser", "--host", "0.0.0.0", "--port", str(port)]) == 0

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "MUSUBI_GUI_PASSWORD" in out
    # The wildcard address is not browsable, so the banner shows loopback.
    assert f"http://127.0.0.1:{port}/" in out
    assert opened == [f"http://127.0.0.1:{port}/"]


def test_non_loopback_host_with_password_does_not_warn(
    env, monkeypatch, server, opened, capsys
):
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", "secret")

    assert (
        launcher.main(["--browser", "--host", "0.0.0.0", "--port", str(free_port())])
        == 0
    )

    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "secret" not in out


def test_loopback_host_without_password_does_not_warn(env, server, opened, capsys):
    assert launcher.main(["--browser", "--port", str(free_port())]) == 0

    assert "WARNING" not in capsys.readouterr().out


def test_port_defaults_to_environment(env, monkeypatch, server, opened):
    port = free_port()
    monkeypatch.setenv("MUSUBI_GUI_PORT", str(port))

    assert launcher.main(["--browser"]) == 0

    assert opened == [f"http://127.0.0.1:{port}/"]


def test_port_option_overrides_environment(env, monkeypatch, server, opened):
    monkeypatch.setenv("MUSUBI_GUI_PORT", "1")
    port = free_port()

    assert launcher.main(["--browser", "--port", str(port)]) == 0

    assert opened == [f"http://127.0.0.1:{port}/"]


@pytest.mark.parametrize("raw", ["abc", "0", "65536"])
def test_invalid_environment_port_is_rejected(env, monkeypatch, raw):
    monkeypatch.setenv("MUSUBI_GUI_PORT", raw)

    with pytest.raises(SystemExit) as excinfo:
        launcher.main(["--browser"])

    assert excinfo.value.code == 2


def test_banner_shows_urls_and_resolved_tools(env, monkeypatch, server, opened, capsys):
    tools = {"python": r"C:\trainer\.venv\Scripts\python.exe", "accelerate": None}
    monkeypatch.setattr(launcher.shutil, "which", lambda name: tools[name])
    port = free_port()

    assert launcher.main(["--browser", "--port", str(port)]) == 0

    out = capsys.readouterr().out
    assert f"http://127.0.0.1:{port}/\n" in out
    assert f"http://127.0.0.1:{port}/mcp" in out
    assert r"python:     C:\trainer\.venv\Scripts\python.exe" in out
    assert "accelerate: not found" in out


def test_make_server_uses_a_single_worker_without_reload(env):
    server = launcher._make_server("127.0.0.1", 8123)

    assert server.config.workers == 1
    assert server.config.reload is False
    assert (server.config.host, server.config.port) == ("127.0.0.1", 8123)
    assert not server.started


def fake_webview(*, start=None) -> SimpleNamespace:
    calls: dict[str, object] = {}

    def create_window(*args, **kwargs):
        calls["create_window"] = (args, kwargs)

    def default_start(**kwargs):
        calls["start"] = kwargs

    return SimpleNamespace(
        settings={"ALLOW_DOWNLOADS": False},
        create_window=create_window,
        start=start or default_start,
        calls=calls,
    )


def test_run_window_configures_and_starts_webview(env, monkeypatch, tmp_path):
    webview = fake_webview()
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(launcher, "_webview2_available", lambda: True)
    storage = tmp_path / "data" / "webview"

    assert launcher._run_window("http://127.0.0.1:8000/", str(storage)) is True

    assert webview.settings["ALLOW_DOWNLOADS"] is True
    assert webview.calls["create_window"] == (
        ("Musubi Tuner GUI", "http://127.0.0.1:8000/"),
        {"width": 1440, "height": 900, "confirm_close": True},
    )
    assert webview.calls["start"] == {
        "private_mode": False,
        "storage_path": str(storage),
    }
    assert storage.is_dir()


@pytest.mark.skipif(sys.platform != "win32", reason="WebView2 is a Windows runtime")
def test_run_window_falls_back_without_webview2(env, monkeypatch, tmp_path, capsys):
    webview = fake_webview()
    monkeypatch.setitem(sys.modules, "webview", webview)
    monkeypatch.setattr(launcher, "_webview2_available", lambda: False)

    assert launcher._run_window("http://127.0.0.1:8000/", str(tmp_path / "w")) is False

    assert "start" not in webview.calls
    assert launcher.WEBVIEW2_URL in capsys.readouterr().out


def test_run_window_falls_back_when_webview_raises(env, monkeypatch, tmp_path, capsys):
    def broken_start(**kwargs):
        raise RuntimeError("no display")

    monkeypatch.setitem(sys.modules, "webview", fake_webview(start=broken_start))
    monkeypatch.setattr(launcher, "_webview2_available", lambda: True)

    assert launcher._run_window("http://127.0.0.1:8000/", str(tmp_path / "w")) is False

    assert "no display" in capsys.readouterr().out
