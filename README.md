# musubi-tuner-gui-backend

FastAPI backend for the musubi-tuner GUI. Implements the contract in
`musubi-tuner-gui-frontend/docs/download-api.md`:

- **Downloads** — runs allowlisted model download scripts as async jobs.
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
| `MUSUBI_GUI_SCRIPTS_DIR`    | `./scripts`              | Allowlisted `download_*.sh` scripts                 |
| `MUSUBI_GUI_WORKSPACE_ROOT` | current directory        | Root confining download destinations, `musubiPath`, and output/logging dirs |
| `MUSUBI_GUI_WEB_DIR`        | `./web`                  | Built Vite frontend served in production            |
| `MUSUBI_GUI_CORS_ORIGINS`   | Vite dev origins on 5173 | Comma-separated allowed CORS origins                |
| `MUSUBI_GUI_MANAGED_MAX_FILES` | `1000`                 | Maximum files in one managed dataset upload         |
| `MUSUBI_GUI_MANAGED_MAX_FILE_BYTES` | `107374182400` (100 GiB) | Maximum bytes for one uploaded media or control file |
| `MUSUBI_GUI_MANAGED_MAX_TOTAL_BYTES` | `1099511627776` (1 TiB) | Maximum combined media bytes in one upload    |
| `MUSUBI_GUI_MANAGED_MAX_REQUEST_BYTES` | `1181116006400` (1.07 TiB) | Maximum full multipart request-body bytes before parsing |
| `MUSUBI_GUI_MANAGED_MAX_STORAGE_BYTES` | `10995116277760` (10 TiB) | Maximum aggregate bytes under managed dataset storage |

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

The download scripts are not part of this repository — place the `download_*.sh` files
listed in the API contract into `scripts/`. Server tools required at runtime: `bash` and
the Hugging Face `hf` CLI (downloads), plus `python` and `accelerate` on `PATH`
(training stages).

## Tests

```bash
uv run pytest
```
