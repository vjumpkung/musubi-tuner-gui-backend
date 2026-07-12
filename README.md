# musubi-tuner-gui-backend

FastAPI backend for the musubi-tuner GUI. Implements the contract in
`musubi-tuner-gui-frontend/docs/download-api.md`:

- **Downloads** — runs allowlisted model download scripts as async jobs.
- **Dataset manager** — stores musubi-tuner dataset configs with TOML import/export.
- **Training job queue** — executes training jobs one at a time through a FIFO queue.

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

The download scripts are not part of this repository — place the `download_*.sh` files
listed in the API contract into `scripts/`. Server tools required at runtime: `bash` and
the Hugging Face `hf` CLI (downloads), plus `python` and `accelerate` on `PATH`
(training stages).

## Tests

```bash
uv run pytest
```
