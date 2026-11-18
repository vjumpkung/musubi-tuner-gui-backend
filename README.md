# musubi-tuner-gui-backend

FastAPI backend for the musubi-tuner GUI. Implements the contract in
`musubi-tuner-gui-frontend/docs/download-api.md`:

- **Downloads** — fetches fixed catalog model bundles as async jobs.
- **Dataset manager** — stores musubi-tuner dataset configs, managed captioned media, control
  images, and TOML import/export.
- **Training job queue** — executes training jobs one at a time through a FIFO queue.
- **System monitor** — reports host CPU/RAM and NVIDIA GPU/VRAM utilization.

## Architecture

- `app/routes/` contains the FastAPI endpoint modules.
- `app/schemas/` contains Pydantic request models.
- `app/services/` contains background and stateful application services.
- `app/utils/` contains stateless validation, command, and process helpers.
- `app/main.py` wires the routers and service lifecycles together.
- `app/launcher.py` runs the server and shows the GUI in a desktop window or browser for local
  use (see [Run locally on Windows](#run-locally-on-windows)).

The original flat module paths remain as compatibility aliases; new imports should use the
responsibility-based packages above.

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Always run with a **single worker process**: the training queue runner is an asyncio task
inside the app, and multiple workers would double-execute jobs.

## Run locally on Windows

`install.bat` and `start.bat` set up and run the GUI on a Windows PC with an NVIDIA GPU. The GUI
opens in its own desktop window, so there is no uvicorn command to type.

Requirements:

- An NVIDIA GPU with a recent driver.
- [Git](https://git-scm.com/download/win).
- [Node.js](https://nodejs.org/) 20.19 or newer. It is only used to build the web UI, so it is
  not needed once `web\index.html` exists.
- Several GB of free disk space for PyTorch and the other trainer dependencies.

[uv](https://docs.astral.sh/uv/) is installed automatically when it is missing, and it fetches
Python 3.12 for the backend and trainer environments.

The parent folder of this repository becomes the workspace root (`MUSUBI_GUI_WORKSPACE_ROOT`), so
use a short ASCII path such as `C:\musubi`:

```text
C:\musubi\
  musubi-tuner-gui-backend\    this repository
  musubi-tuner-gui-frontend\   cloned by install.bat to build the web UI
  musubi-tuner\                the trainer, with its own .venv
```

```bat
mkdir C:\musubi
cd C:\musubi
git clone https://github.com/vjumpkung/musubi-tuner-gui-backend.git
cd musubi-tuner-gui-backend
install.bat
```

`install.bat` sets up the backend environment (including the `desktop` extra), builds the web UI
into `web\`, clones and installs musubi-tuner with PyTorch (cu128 wheels) in
`..\musubi-tuner\.venv`, and then starts the GUI. Steps that are already done are skipped, so it
is safe to run again. Later, start the GUI with `start.bat`.

| `install.bat` option | Effect                                                          |
| -------------------- | --------------------------------------------------------------- |
| `--skip-trainer`     | Do not set up musubi-tuner and PyTorch (training will not work) |
| `--rebuild-frontend` | Rebuild the web UI even if `web\index.html` already exists      |
| `--no-start`         | Do not run `start.bat` when the installation finishes           |

Set `TORCH_INDEX_URL` before running `install.bat` to use another PyTorch wheel index (default
`https://download.pytorch.org/whl/cu128`).

| `start.bat` option | Effect                                                                       |
| ------------------ | ---------------------------------------------------------------------------- |
| `--browser`        | Open the default web browser instead of a desktop window                     |
| `--host HOST`      | Interface to listen on (default `127.0.0.1`; use `0.0.0.0` for the network)  |
| `--port PORT`      | Port to listen on (default `MUSUBI_GUI_PORT`, else `8000`)                   |

`start.bat` passes its arguments to `python -m app.launcher`. It also sets
`MUSUBI_GUI_WORKSPACE_ROOT` to the workspace root (unless already set) and puts
`..\musubi-tuner\.venv\Scripts` first on `PATH`, so training jobs run with the trainer's `python`
and `accelerate`. For that reason it starts the backend with its own `.venv\Scripts\python.exe`
directly instead of `uv run`, and the backend resolves bare `python`/`accelerate` against `PATH`
itself, because Windows would otherwise look in the backend interpreter's folder first.

The launcher prints the app URL, the MCP URL, and the `python`/`accelerate` it found. It exits
with a message suggesting `--port` when the port is already in use, and it never picks a random
port because the login cookie and browser storage belong to one origin. By default it only
listens on `127.0.0.1`; with `--host` on a non-loopback address and no `MUSUBI_GUI_PASSWORD`, it
warns that the GUI is reachable without a password. To require a password locally, set it before
starting:

```bat
set MUSUBI_GUI_PASSWORD=choose-a-long-random-password
start.bat
```

The window uses [pywebview](https://pywebview.flowrl.com/) (the `desktop` extra, installed by
`install.bat`) on the Microsoft Edge WebView2 runtime. If pywebview or the WebView2 runtime is
missing, or the window cannot open, the launcher says why and opens the browser instead. The
window keeps its cookies and browser storage under `data\webview`, so the login session
survives restarts. Closing the window asks for confirmation, then stops the server **and any
running training job**. With `--browser`, press Ctrl+C in the console to stop it.

On other platforms, or without the batch files:

```bash
uv sync --extra desktop
# Put the trainer environment's python/accelerate first on PATH, then run the
# backend's own interpreter (not `uv run`, which would put .venv first):
.venv/bin/python -m app.launcher --browser
```

## Development vs production

- **Development** — run this server, then `pnpm dev` in `musubi-tuner-gui-frontend`. The Vite
  dev server proxies `/api` requests here (target configurable with `VITE_API_PROXY_TARGET`,
  default `http://127.0.0.1:8000`).
- **Production** — `pnpm build` in the frontend writes the built app into `web/` in this
  directory, and FastAPI serves it (SPA fallback included). API routes always win over
  static files.

## Configuration (environment variables)

| Variable                    | Default                  | Purpose                                             |
| --------------------------- | ------------------------ | --------------------------------------------------- |
| `MUSUBI_GUI_DATA_ROOT`      | `./data`                 | SQLite database, job logs, dataset TOML snapshots   |
| `HF_TOKEN`                  | unset                    | Optional download token when the request omits one  |
| `MUSUBI_GUI_WORKSPACE_ROOT` | current directory        | Root confining download destinations, `musubiPath`, and output/logging dirs |
| `MUSUBI_GUI_WEB_DIR`        | `./web`                  | Built Vite frontend served in production            |
| `MUSUBI_GUI_CORS_ORIGINS`   | Vite dev origins on 5173 | Comma-separated allowed CORS origins                |
| `MUSUBI_GUI_PASSWORD`       | unset (login disabled)   | Shared password for the GUI, API, and MCP endpoint (see [Authentication](#authentication)). Set a long random value whenever the server is reachable by others, for example on RunPod |
| `MUSUBI_GUI_PORT`           | `8000`                   | Default port for `python -m app.launcher` and `start.bat` (`--port` overrides it) |
| `MUSUBI_GUI_MANAGED_MAX_FILES` | unset (no limit)       | Optional maximum files in one managed dataset upload |
| `MUSUBI_GUI_MANAGED_MAX_FILE_BYTES` | `107374182400` (100 GiB) | Maximum bytes for one uploaded media or control file |
| `MUSUBI_GUI_MANAGED_MAX_TOTAL_BYTES` | `1099511627776` (1 TiB) | Maximum combined media bytes in one upload    |
| `MUSUBI_GUI_MANAGED_MAX_REQUEST_BYTES` | `1181116006400` (1.07 TiB) | Maximum full multipart request-body bytes before parsing |
| `MUSUBI_GUI_MANAGED_MAX_STORAGE_BYTES` | `10995116277760` (10 TiB) | Maximum aggregate bytes under managed dataset storage |

The browser UI stages managed uploads one file per request, so a dataset may contain any number
of files within the byte quotas. Direct multipart calls to `POST /api/datasets/managed`,
`POST /api/datasets/managed/batch`, and `PUT /api/datasets/{id}/managed` remain bounded by
Starlette's form parser at 1000 file parts per request (HTTP 400).

Managed upload admission is serialized in the required single-worker server so concurrent
requests cannot overcommit the aggregate storage quota. Failed creations and deletions use
identifiable `.orphan-*`, `.pending-delete-*`, and `.tombstone-*` directories. Reconciliation
checks SQLite ownership before restoring or removing them at startup and before the next managed
storage operation; all retained state remains included in quota accounting.

Managed uploads accept captions entered in the UI or UTF-8 `.txt` sidecars with the same stem as
their media (`a.png` + `a.txt`). Image datasets may include a separate control-image upload whose
files use the target stem (`a.jpg` + control `a.png`) or numbered stems such as `a_0.png` and
`a_0001.png`; the generated TOML includes `control_directory`. One managed config can contain
multiple image and video `[[datasets]]` entries, each with its own resolution, cache directory,
sampling settings, and `num_repeats` value.

Downloads fetch catalog URLs from `huggingface.co` directly with HTTPX; no bash or hf CLI
is needed. Supply a token in the request or use the optional `HF_TOKEN` environment
fallback. Cached `hf auth login` credentials are not read. Files are re-downloaded and
replaced on every run; downloads do not reuse the Hugging Face cache or resume transfers.
Training stages require `python` and `accelerate` on `PATH`.

## Authentication

Password protection is optional and off by default. Set `MUSUBI_GUI_PASSWORD` to turn it on;
an unset or blank value leaves the GUI and API open, as before. The server logs one line at
startup saying whether it is enabled, and never logs the password.

- **Browser sessions** — the web UI shows a login page. `POST /api/auth/login` with
  `{"password": "..."}` returns `200 {"enabled": true, "authenticated": true}` and sets the
  `musubi_gui_session` cookie (`HttpOnly`, `SameSite=Lax`, valid for 7 days, `Secure` when the
  request arrived over HTTPS directly or through a proxy that sends `X-Forwarded-Proto: https`).
  A wrong password returns `401 {"detail": "Incorrect password"}`. The session is a signed token,
  and changing the password invalidates existing sessions.
- **Scripts and MCP clients** — send `Authorization: Bearer <MUSUBI_GUI_PASSWORD>` with each
  request instead of using a cookie.
- **Endpoints** — all public, under `/api/auth`:

  | Method | Route                | Purpose                                                     |
  | ------ | -------------------- | ----------------------------------------------------------- |
  | `GET`  | `/api/auth/status`   | `{"enabled", "authenticated"}` (`authenticated` is `true` when auth is disabled) |
  | `POST` | `/api/auth/login`    | Exchange the password for a session cookie                  |
  | `POST` | `/api/auth/logout`   | Clear the cookie (`204`)                                    |

- **Protected paths** — everything under `/api/` except `/api/auth/` (including the Socket.IO
  endpoint), `/mcp`, `/openapi.json`, `/docs`, and `/redoc`. A request without valid credentials
  gets `401 {"detail": "Authentication required"}`. The built web UI and its assets stay public so
  the login page can load.
- **CSRF header** — a cookie-authenticated `POST`, `PUT`, `PATCH`, or `DELETE` must carry an
  `X-Requested-With` header (any value), otherwise it gets
  `403 {"detail": "Missing X-Requested-With header"}`. The web UI sends it automatically.
  Bearer-authenticated requests and the Socket.IO transport are exempt.
- **Throttling** — a wrong password, whether from the login form or a Bearer header, is delayed
  by one second, and attempts made during that delay get `429` with `Retry-After: 1`. Guessing is
  therefore limited to about one attempt per second.

The password travels in plain text over plain HTTP, so use HTTPS (the RunPod proxy provides it)
whenever the GUI is reachable beyond your own machine.

## Realtime updates

The server pushes state changes to the web UI over Socket.IO, so progress, queue changes, logs,
and system usage update without polling. The Socket.IO endpoint is `/api/socket.io` on the same
port, uses the same authentication as the API, and accepts same-origin pages (checked against the
`Host`/`X-Forwarded-Host` header) plus the origins in `MUSUBI_GUI_CORS_ORIGINS`. Events are sent
right after the matching database write commits and are delivered in order.

| Event              | Payload                                                         | Sent to                  |
| ------------------ | --------------------------------------------------------------- | ------------------------ |
| `queue:update`     | Queue object, as in `GET /api/training/queue`                   | Everyone                 |
| `job:update`       | Training job object, as in `GET /api/training/jobs/{id}`        | Everyone                 |
| `job:removed`      | `{"id": "<job id>"}`                                            | Everyone                 |
| `jobs:changed`     | `{}` — the job list changed (new, reordered, or unlinked from a deleted dataset config); refetch it | Everyone |
| `download:update`  | Download job object, as in `GET /api/downloads/{id}`            | Everyone                 |
| `job:log`          | `{"job_id", "offset", "next_offset", "content"}`                | `job-log` subscribers of that job |
| `system:resources` | Snapshot, as in `GET /api/system/resources`, every 5 seconds    | `system` subscribers     |

`job:log` and `system:resources` are only sent to clients that subscribed. A client sends
`subscribe` or `unsubscribe` with `{"topic": "system"}` or
`{"topic": "job-log", "job_id": "<job id>"}` (`job_id` is 1 to 64 characters); any other payload is
ignored. Subscriptions belong to the connection, so a client must subscribe again after
reconnecting. `job:log` chunks use the same byte offsets as `GET /api/training/jobs/{id}/logs`:
output is merged for up to a quarter of a second into chunks of roughly 64 KiB at most, so a
client appends a chunk when its `offset` equals the offset it already has and reloads from the
logs endpoint if there is a gap.

The REST endpoints stay the source of truth. While the socket is disconnected, the web UI falls
back to its previous HTTP polling and refetches everything when the socket reconnects.

## MCP server

The backend exposes an [MCP](https://modelcontextprotocol.io/) server over Streamable HTTP at
`/mcp` on the same port as the GUI, so an AI client such as Claude Code can inspect and drive
the GUI: check the system, start downloads, create dataset configs, and queue and monitor
training jobs.

With `MUSUBI_GUI_PASSWORD` set, requests must carry `Authorization: Bearer <MUSUBI_GUI_PASSWORD>`;
without a password no header is needed. For example, with Claude Code:

```bash
claude mcp add --transport http musubi-tuner-gui http://127.0.0.1:8000/mcp --header "Authorization: Bearer <password>"
```

| Area      | Tools                                                                                      |
| --------- | ------------------------------------------------------------------------------------------ |
| System    | `get_system_resources`                                                                     |
| Downloads | `list_download_models`, `list_downloads`, `get_download`, `start_download`, `cancel_download` |
| Datasets  | `list_dataset_configs`, `get_dataset_config`, `validate_dataset_config`, `create_dataset_config` |
| Training  | `list_training_profiles`, `get_training_queue`, `start_training_queue`, `pause_training_queue`, `list_training_jobs`, `get_training_job`, `create_training_job`, `cancel_training_job`, `retry_training_job`, `reorder_training_job`, `get_training_job_logs`, `list_training_job_artifacts` |

Most tools call the matching REST endpoint inside the server, so validation, errors, and
side effects are the same as for the web UI. `list_download_models` returns the catalog bundle
ids that `start_download` takes as `script_id`, and `list_training_profiles` returns the profile
ids that `create_training_job` takes as `profile_id` with each profile's required `values` keys
(`outputName`, `outputDir`, and `musubiPath` are always required as well). By design there are
no tools that delete data or upload files, and TOML import/export and checkpoint downloads are
not exposed.

## Tests

```bash
uv run pytest
```
