# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 FastAPI backend managed with `uv`. Application code lives in `app/`.
`app/main.py` creates the FastAPI application and owns its lifespan. API routers are grouped by
feature in `downloads.py`, `datasets.py`, and `training.py`. Queue execution lives in
`queue_runner.py`; subprocess construction and lifecycle handling live in `command_builder.py` and
`process.py`. SQLite access is centralized in `db.py`, while environment-based configuration and
workspace path confinement live in `config.py`. Tests live in `tests/` and mirror the main backend
features.

The built frontend may be written to `web/` and served by FastAPI in production. Treat `web/`,
`data/`, local databases, job logs, and dataset snapshots as generated runtime content rather than
backend source code.

## Setup, Run, and Test Commands

- `uv sync`: create or update the virtual environment from `pyproject.toml` and `uv.lock`.
- `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`: run the API locally.
- `uv run pytest`: run the complete backend test suite.
- `uv run pytest tests/test_training.py`: run one feature test module.
- `uv run pytest tests/test_training.py::test_name`: run one specific test.

Always run Uvicorn with exactly one worker. The in-process queue runner executes training jobs, so
multiple workers can claim and execute the same work independently.

## Coding Style & API Conventions

Use Python 3.12 syntax, four-space indentation, type annotations, and focused async functions.
Prefer FastAPI dependency/request state patterns already used in the routers. Keep database writes
serialized through the existing write locks and commit or roll back atomically. Return plain,
JSON-serializable response objects and use `HTTPException` with clear `detail` messages for expected
client errors.

Keep API routes under `/api`. Preserve existing status-code semantics: `201` for created dataset
configs, `202` for accepted background work, and `204` for successful deletions without a body.
When changing a response or request schema, update the matching frontend API types and tests.

## Testing Guidelines

Every behavioral change should include or update tests. Use the shared fixtures in
`tests/conftest.py`, keep filesystem operations inside temporary directories, and replace external
commands with deterministic stubs. Cover successful behavior plus validation, conflict, missing
resource, cancellation, and cleanup paths as applicable. Run `uv run pytest` before handing off a
change.

Tests involving the queue must remain deterministic and must not launch real training or download
processes. Use the existing polling helpers instead of fixed long sleeps.

## Queue and Process Safety

The queue state and job transitions are concurrency-sensitive. Use the runner's claim lock and the
database write lock consistently when changing job ownership, status, or queue positions. Preserve
FIFO ordering and renumber queued jobs after cancellation or reordering. Ensure application
shutdown terminates child process trees and closes downloads, runner tasks, and the database.

Build subprocess commands as argument lists and execute them without a shell. Keep download scripts
strictly allowlisted. Do not weaken `extraArgs` validation or pass untrusted strings into shell
commands. Secrets such as Hugging Face tokens must never appear in API responses, errors, or logs;
maintain redaction whenever command logging changes.

## Filesystem and Configuration Safety

All client-supplied workspace paths must be resolved through `Settings.resolve_inside_workspace`.
Do not permit download destinations, `musubiPath`, output directories, or logging directories to
escape `MUSUBI_GUI_WORKSPACE_ROOT`. Avoid committing `.env` files, databases, generated logs,
downloaded models, dataset snapshots, or secrets. Document new environment variables in
`README.md` with safe defaults.

## Commit and Pull Request Guidelines

Keep commits focused and use short imperative messages. Pull requests should summarize API or
behavior changes, mention configuration or migration requirements, identify any frontend contract
changes, and list the verification commands run.
