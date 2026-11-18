"""Local launcher: run the GUI server and show it in a desktop window.

    python -m app.launcher [--browser] [--host HOST] [--port PORT]

By default the server runs in a background thread and the UI opens in a native
window (pywebview, installed with the ``desktop`` extra). ``--browser`` opens
the default web browser instead. Either way exactly one uvicorn worker runs in
this process, because the in-process queue runner must not execute jobs twice.
"""

from __future__ import annotations

import argparse
import errno
import logging
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from .config import load_settings

DEFAULT_PORT = 8000
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
START_TIMEOUT_SECONDS = 60.0
STOP_TIMEOUT_SECONDS = 15.0
WEBVIEW2_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"


def _port_number(raw: str) -> int:
    try:
        port = int(raw)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be an integer between 1 and 65535")
    return port


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.launcher",
        description="Run Musubi Tuner GUI locally in a desktop window or browser.",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="open the default web browser instead of a desktop window",
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on")
    parser.add_argument(
        "--port",
        type=_port_number,
        # A string default is converted with ``type`` too, so a bad
        # MUSUBI_GUI_PORT is reported like a bad --port.
        default=os.environ.get("MUSUBI_GUI_PORT", "").strip() or str(DEFAULT_PORT),
        help="port to listen on (default: $MUSUBI_GUI_PORT or %(default)s)",
    )
    return parser


def _bind_error(host: str, port: int) -> OSError | None:
    """Return the error from trying to bind (host, port), or None when it is free."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            # Matches asyncio: a lingering TIME_WAIT must not block a restart.
            # Never set on Windows, where it would allow stealing a busy port.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            return error
    return None


def _display_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    if ":" in display_host:
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}/"


def _print_banner(host: str, url: str) -> None:
    print("Musubi Tuner GUI", flush=True)
    print(f"  app:        {url}", flush=True)
    print(f"  mcp:        {url}mcp", flush=True)
    print(f"  python:     {shutil.which('python') or 'not found'}", flush=True)
    print(f"  accelerate: {shutil.which('accelerate') or 'not found'}", flush=True)
    if load_settings().password is None and host not in LOOPBACK_HOSTS:
        print(
            "WARNING: the GUI is reachable from the network without a password. "
            "Set MUSUBI_GUI_PASSWORD to require one.",
            flush=True,
        )


def _make_server(host: str, port: int) -> uvicorn.Server:
    # Imported here so the app (and the environment it reads) loads at launch.
    from .main import app

    # No reload or extra workers: on Windows they switch uvicorn to a
    # SelectorEventLoop, which cannot run the subprocesses training needs.
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        workers=1,
        timeout_graceful_shutdown=5,
        log_level="info",
    )
    return uvicorn.Server(config)


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def _webview2_available() -> bool:
    """Whether pywebview will render with Edge WebView2 (Windows only)."""
    # webview.start() picks the renderer while it creates the window, which is
    # too late to fall back. initialize() resolves it without opening anything
    # (and is repeated, harmlessly, by start()).
    from webview.guilib import initialize

    return initialize().renderer == "edgechromium"


def _run_window(url: str, storage_path: str) -> bool:
    """Show the GUI in a desktop window and block until it is closed.

    Returns False, after saying why, when no window could be shown so the caller
    can fall back to the browser.
    """
    try:
        import webview
    except ImportError:
        print(
            "pywebview is not installed; opening in your browser instead. "
            "Install with: uv sync --extra desktop",
            flush=True,
        )
        return False

    try:
        if sys.platform == "win32" and not _webview2_available():
            print(
                "The Microsoft Edge WebView2 runtime was not found; opening in "
                f"your browser instead. Install it from {WEBVIEW2_URL}",
                flush=True,
            )
            return False
        webview.settings["ALLOW_DOWNLOADS"] = True
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        webview.create_window(
            "Musubi Tuner GUI", url, width=1440, height=900, confirm_close=True
        )
        # Keep cookies and localStorage between runs (the login session, theme).
        webview.start(private_mode=False, storage_path=storage_path)
    except Exception as error:
        print(
            f"Could not open the desktop window ({error}); opening in your "
            "browser instead.",
            flush=True,
        )
        return False
    return True


def _open_when_started(server: uvicorn.Server, url: str) -> None:
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.started:
            _open_browser(url)
            return
        time.sleep(0.05)


def _serve_in_browser(server: uvicorn.Server, url: str) -> int:
    """Run uvicorn in this thread and open the browser once it is listening."""
    threading.Thread(
        target=_open_when_started, args=(server, url), name="open-browser", daemon=True
    ).start()
    try:
        server.run()
    except KeyboardInterrupt:
        # uvicorn re-raises Ctrl+C after its graceful shutdown has finished.
        pass
    return 0


def _wait_until_stopped(thread: threading.Thread) -> None:
    # Short joins keep Ctrl+C responsive on Windows.
    while thread.is_alive():
        thread.join(timeout=0.5)


def _serve_in_window(server: uvicorn.Server, url: str, storage_path: str) -> int:
    """Run uvicorn in a thread and the desktop window in this (main) thread."""
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while not server.started:
        if not thread.is_alive():
            print("The server failed to start; see the log above.", flush=True)
            return 1
        if time.monotonic() >= deadline:
            server.should_exit = True
            print(
                f"The server did not start within {START_TIMEOUT_SECONDS:.0f} seconds.",
                flush=True,
            )
            return 1
        time.sleep(0.05)

    try:
        if not _run_window(url, storage_path):
            _open_browser(url)
            print("Press Ctrl+C to stop the server.", flush=True)
            _wait_until_stopped(thread)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        thread.join(timeout=STOP_TIMEOUT_SECONDS)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    error = _bind_error(args.host, args.port)
    if error is not None:
        if error.errno == errno.EADDRINUSE:
            print(
                f"Port {args.port} is already in use. Is Musubi Tuner GUI already "
                "running? Use --port to pick another port.",
                file=sys.stderr,
            )
        else:
            print(f"Cannot listen on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1

    url = _display_url(args.host, args.port)
    _print_banner(args.host, url)
    server = _make_server(args.host, args.port)
    if args.browser:
        return _serve_in_browser(server, url)
    storage_path = str(load_settings().data_root / "webview")
    return _serve_in_window(server, url, storage_path)


if __name__ == "__main__":
    raise SystemExit(main())
