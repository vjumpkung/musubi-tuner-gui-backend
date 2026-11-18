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
from fastmcp.server.http import StarletteWithLifespan
from fastmcp.server.providers.openapi import MCPType, RouteMap
from fastmcp.utilities.openapi import HttpMethod

from ..download_catalog import DOWNLOAD_MODELS
from ..profiles import TRAINING_PROFILES
from ..services.auth import AuthManager

# (HTTP method, OpenAPI path, MCP tool name). Everything not listed is excluded:
# deletes, dataset updates, managed uploads/sessions, import/export, file
# downloads, /api/auth/*, and socket.io.
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
    def list_training_profiles() -> list[dict[str, Any]]:
        """List the training profiles that create_training_job accepts (use the id
        as profile_id) and the model values each requires. outputName, outputDir
        and musubiPath are always required as well."""
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

    return mcp.http_app(
        path="/",
        stateless_http=True,
        json_response=True,
        host_origin_protection="auto",
    )
