"""MCP (Model Context Protocol) endpoint exposing a curated slice of the API.

The tools are generated from the FastAPI app's own OpenAPI schema (so they reuse
the existing validation and route logic) through an explicit allowlist; every
other operation is excluded. Tool calls run in-process against the same app, so
they pass through ``AuthMiddleware``. FastMCP strips the incoming client's
``Authorization`` header, therefore the in-process client always sends the
per-process internal token instead.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.http import StarletteWithLifespan
from fastmcp.server.providers.openapi import MCPType, RouteMap
from fastmcp.utilities.openapi import HttpMethod

from ..download_catalog import DOWNLOAD_MODELS
from ..profiles import TRAINING_PROFILES
from ..services.auth import AuthManager
from ..utils.command_builder import describe_values

# (HTTP method, OpenAPI path, MCP tool name). Everything not listed is excluded:
# deletes, dataset updates, the multipart file upload and token-list finalize
# routes (files are uploaded with curl, never through MCP), the other managed
# dataset routes, import/export, file downloads, /api/auth/*, and socket.io.
TOOL_ALLOWLIST: tuple[tuple[HttpMethod, str, str], ...] = (
    ("GET", "/api/system/resources", "get_system_resources"),
    ("GET", "/api/downloads", "list_downloads"),
    ("GET", "/api/downloads/{job_id}", "get_download"),
    ("POST", "/api/downloads", "start_download"),
    ("POST", "/api/downloads/{job_id}/cancel", "cancel_download"),
    ("GET", "/api/datasets", "list_dataset_configs"),
    ("GET", "/api/datasets/{config_id}", "get_dataset_config"),
    ("POST", "/api/datasets/{config_id}/validate", "validate_dataset_config"),
    ("POST", "/api/datasets", "create_dataset_config"),
    ("POST", "/api/datasets/from-folder", "create_dataset_from_folder"),
    ("POST", "/api/datasets/managed/upload-sessions", "create_dataset_upload_session"),
    (
        "POST",
        "/api/datasets/managed/upload-sessions/{session_id}/finalize-simple",
        "finalize_dataset_upload",
    ),
    (
        "DELETE",
        "/api/datasets/managed/upload-sessions/{session_id}",
        "cancel_dataset_upload_session",
    ),
    ("GET", "/api/training/queue", "get_training_queue"),
    ("POST", "/api/training/queue/start", "start_training_queue"),
    ("POST", "/api/training/queue/pause", "pause_training_queue"),
    ("GET", "/api/training/jobs", "list_training_jobs"),
    ("GET", "/api/training/jobs/{job_id}", "get_training_job"),
    ("POST", "/api/training/jobs", "create_training_job"),
    ("POST", "/api/training/jobs/{job_id}/cancel", "cancel_training_job"),
    ("POST", "/api/training/jobs/{job_id}/retry", "retry_training_job"),
    ("PATCH", "/api/training/jobs/{job_id}", "reorder_training_job"),
    ("GET", "/api/training/jobs/{job_id}/logs", "get_training_job_logs"),
    ("GET", "/api/training/jobs/{job_id}/artifacts", "list_training_job_artifacts"),
)

MCP_INSTRUCTIONS = """\
Train a LoRA with musubi-tuner in this order.
1. Models: call list_download_models, then start_download (the id is script_id) and \
poll get_download until it ends. Its files field lists the absolute paths downloaded.
2. Profile: call list_training_profiles(profile_id=...) for every field the profile \
accepts, with its type, options and CLI flags.
3. Dataset on the server: call create_dataset_from_folder first. Use an absolute path \
or one relative to the workspace root. Every image needs a .txt caption of the same name.
4. Dataset on your machine: if step 3 says "Folder not found on the server", call \
create_dataset_upload_session, then get_dataset_upload_info, upload each file with the \
curl command it returns (kind=target for media, kind=caption for .txt files), then call \
finalize_dataset_upload. If finalize reports a problem the session is kept, so fix it and \
call it again. Call cancel_dataset_upload_session to abandon the upload.
5. Never send file bytes through MCP. curl authenticates with $MUSUBI_GUI_PASSWORD from \
your shell environment; never print, echo or log its value.
6. create_training_job: the dataset id is dataset_config_id. Send booleans as JSON \
true/false, never as strings. Relative paths resolve under the workspace root; musubiPath, \
outputDir and loggingDir must stay inside it. Read warnings in the response: they name \
values the profile ignores.
7. The queue is paused after each enqueue and after every finished, failed or cancelled \
job, so call start_training_queue for each job.
8. Poll get_training_job and get_training_job_logs, passing the next_offset from the \
previous response as offset.
9. list_training_job_artifacts returns absolute paths of the results.
"""


def _operation_ids(app: FastAPI) -> dict[tuple[str, str], str]:
    """Map (method, path) to the operation id FastAPI emits in its OpenAPI schema.

    Read from the schema rather than ``app.routes``: current FastAPI keeps
    included routers lazy, so ``app.routes`` no longer lists the API routes.
    """
    operation_ids: dict[tuple[str, str], str] = {}
    for path, operations in app.openapi()["paths"].items():
        for method, operation in operations.items():
            if isinstance(operation, dict) and "operationId" in operation:
                operation_ids[(method.upper(), path)] = operation["operationId"]
    return operation_ids


def build_mcp_app(app: FastAPI, auth: AuthManager) -> StarletteWithLifespan:
    """Build the MCP ASGI app. Call after every API router has been included."""
    operation_ids = _operation_ids(app)
    mcp_names: dict[str, str] = {}
    route_maps: list[RouteMap] = []
    for method, path, tool_name in TOOL_ALLOWLIST:
        operation_id = operation_ids.get((method, path))
        if operation_id is None:
            raise RuntimeError(f"MCP allowlist route not found: {method} {path}")
        mcp_names[operation_id] = tool_name
        route_maps.append(
            RouteMap(
                methods=[method],
                pattern=f"^{re.escape(path)}$",
                mcp_type=MCPType.TOOL,
            )
        )
    route_maps.append(RouteMap(mcp_type=MCPType.EXCLUDE))

    mcp = FastMCP.from_fastapi(
        app,
        name="musubi-tuner-gui",
        route_maps=route_maps,
        mcp_names=mcp_names,
        instructions=MCP_INSTRUCTIONS,
        httpx_client_kwargs={
            "headers": {"Authorization": f"Bearer {auth.internal_token}"},
            "timeout": 120,
        },
    )

    @mcp.tool(annotations={"readOnlyHint": True})
    def list_download_models() -> list[dict[str, Any]]:
        """List the model bundles that start_download can fetch (use the id as
        script_id)."""
        return [
            {"id": model.id, "name": model.name, "file_count": len(model.files)}
            for model in DOWNLOAD_MODELS.values()
        ]

    @mcp.tool(annotations={"readOnlyHint": True})
    def list_training_profiles(profile_id: str | None = None) -> list[dict[str, Any]]:
        """List the training profiles that create_training_job accepts (use the id
        as profile_id) and the model values each requires. outputName, outputDir
        and musubiPath are always required as well. Pass profile_id to get just
        that profile with every value it accepts (fields: type, options, CLI
        flags and conditions), which create_training_job needs."""
        if profile_id is not None:
            profile = TRAINING_PROFILES.get(profile_id)
            if profile is None:
                raise ToolError(
                    f"Unknown profile_id '{profile_id}'. "
                    f"Available: {', '.join(TRAINING_PROFILES)}"
                )
            return [
                {
                    "id": profile.id,
                    "name": profile.name,
                    "required_fields": [
                        field.key for field in profile.model_fields if field.required
                    ],
                    "fields": describe_values(profile),
                }
            ]
        return [
            {
                "id": profile.id,
                "name": profile.name,
                "required_fields": [
                    field.key for field in profile.model_fields if field.required
                ],
            }
            for profile in TRAINING_PROFILES.values()
        ]

    @mcp.tool(annotations={"readOnlyHint": True})
    def get_dataset_upload_info() -> dict[str, Any]:
        """Get the URL and curl command for uploading dataset files to an upload
        session (create_dataset_upload_session). Files are uploaded with curl, not
        through MCP; replace {session_id} with the session id."""
        request = get_http_request()
        # Read the forwarded headers raw: uvicorn only trusts them from 127.0.0.1,
        # but behind a platform proxy they carry the public address.
        proto = (
            request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
            or request.url.scheme
        )
        host = (
            request.headers.get("x-forwarded-host", "").split(",")[0].strip()
            or request.headers.get("host", "").strip()
            or request.url.netloc
        )
        upload_url = f"{proto}://{host}/api/datasets/managed/upload-sessions/{{session_id}}/files"
        auth_header = (
            ' -H "Authorization: Bearer $MUSUBI_GUI_PASSWORD"' if auth.enabled else ""
        )
        return {
            "upload_url": upload_url,
            "password_required": auth.enabled,
            "curl": (
                f'curl -fsS{auth_header} -F kind=target -F "file=@image.png" {upload_url}'
            ),
            "note": (
                "Replace {session_id} with the id from create_dataset_upload_session. "
                'Use -F kind=caption -F "file=@image.txt" for the .txt caption files, '
                "and upload every file before calling finalize_dataset_upload. If "
                "upload_url is unreachable, use the origin of the MCP server URL "
                "instead. On Windows PowerShell use curl.exe and "
                "$env:MUSUBI_GUI_PASSWORD."
            ),
        }

    return mcp.http_app(
        path="/",
        stateless_http=True,
        json_response=True,
        host_origin_protection="auto",
    )
