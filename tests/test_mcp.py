import pytest

from app.download_catalog import DOWNLOAD_MODELS
from app.profiles import TRAINING_PROFILES
from app.services import auth as auth_service

PASSWORD = "mcp test password"
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

ALLOWLISTED_TOOLS = {
    "get_system_resources",
    "list_downloads",
    "get_download",
    "start_download",
    "cancel_download",
    "list_dataset_configs",
    "get_dataset_config",
    "validate_dataset_config",
    "create_dataset_config",
    "get_training_queue",
    "start_training_queue",
    "pause_training_queue",
    "list_training_jobs",
    "get_training_job",
    "create_training_job",
    "cancel_training_job",
    "retry_training_job",
    "reorder_training_job",
    "get_training_job_logs",
    "list_training_job_artifacts",
}
CATALOG_TOOLS = {"list_download_models", "list_training_profiles"}
# Names FastMCP would auto-generate for operations that must stay excluded.
EXCLUDED_TOOLS = {
    "delete_dataset",
    "delete_job",
    "update_dataset",
    "update_managed_dataset_files",
    "export_dataset",
    "import_dataset",
    "create_managed_dataset",
    "create_managed_upload_session",
    "download_job_artifacts",
    "download_job_artifact",
    "download_job_config",
    "read_status",
    "login",
    "logout",
}


def rpc(client, method, params=None, *, path="/mcp", headers=None, request_id=1):
    body = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return client.post(
        path,
        json=body,
        headers={**MCP_HEADERS, **(headers or {})},
        follow_redirects=False,
    )


def call_tool(client, name, arguments=None, **kwargs):
    response = rpc(
        client,
        "tools/call",
        {"name": name, "arguments": arguments or {}},
        **kwargs,
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def list_tool_names(client, **kwargs) -> set[str]:
    response = rpc(client, "tools/list", {}, **kwargs)
    assert response.status_code == 200, response.text
    return {tool["name"] for tool in response.json()["result"]["tools"]}


@pytest.fixture
def auth_client(make_client, monkeypatch):
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", PASSWORD)
    monkeypatch.setattr(auth_service, "FAILURE_DELAY_SECONDS", 0)
    with make_client() as test_client:
        yield test_client


def test_tools_list_is_exactly_the_allowlist_plus_catalog_tools(client):
    names = list_tool_names(client)

    assert names == ALLOWLISTED_TOOLS | CATALOG_TOOLS
    assert not names & EXCLUDED_TOOLS


def test_excluded_operation_cannot_be_called(client):
    result = call_tool(client, "delete_job", {"job_id": "anything"})

    assert result["isError"] is True


def test_get_training_queue_returns_queue_state(client):
    result = call_tool(client, "get_training_queue")

    assert result["isError"] is False
    assert set(result["structuredContent"]) == {"state", "queued", "running_job_id"}


def test_list_training_jobs_returns_jobs_list(client):
    result = call_tool(client, "list_training_jobs")

    assert result["isError"] is False
    assert result["structuredContent"] == {"result": []}


def test_tool_error_is_reported_as_tool_result(client):
    result = call_tool(client, "get_training_job", {"job_id": "missing"})

    assert result["isError"] is True
    assert "404" in result["content"][0]["text"]


def test_list_download_models_returns_catalog(client):
    result = call_tool(client, "list_download_models")

    models = result["structuredContent"]["result"]
    assert models
    assert {model["id"] for model in models} == set(DOWNLOAD_MODELS)
    assert all(set(model) == {"id", "name", "file_count"} for model in models)
    assert all(model["file_count"] >= 1 for model in models)


def test_list_training_profiles_returns_profiles(client):
    result = call_tool(client, "list_training_profiles")

    profiles = result["structuredContent"]["result"]
    assert profiles
    assert {profile["id"] for profile in profiles} == set(TRAINING_PROFILES)
    assert all(
        set(profile) == {"id", "name", "required_fields"} for profile in profiles
    )
    assert all(profile["required_fields"] for profile in profiles)


def test_write_tools_change_queue_state(client):
    started = call_tool(client, "start_training_queue")
    assert started["structuredContent"]["state"] == "running"
    assert client.get("/api/training/queue").json()["state"] == "running"

    paused = call_tool(client, "pause_training_queue")
    assert paused["structuredContent"]["state"] == "paused"
    assert client.get("/api/training/queue").json()["state"] == "paused"


@pytest.mark.parametrize("path", ["/mcp", "/mcp/", "/mcp?client=test"])
def test_mcp_endpoint_accepts_bare_and_slash_paths(client, path):
    result = call_tool(client, "get_training_queue", path=path)

    assert result["isError"] is False


def test_mcp_paths_work_with_spa_catch_all(make_client, paths, monkeypatch):
    web_dir = paths["workspace"] / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    monkeypatch.setenv("MUSUBI_GUI_WEB_DIR", str(web_dir))

    with make_client() as client:
        assert "spa" in client.get("/").text
        for path in ("/mcp", "/mcp/"):
            response = rpc(client, "tools/list", {}, path=path)
            assert response.status_code == 200, (path, response.text)
            assert "get_training_queue" in response.text


def test_mcp_requires_credentials_when_auth_enabled(auth_client):
    for path in ("/mcp", "/mcp/"):
        assert rpc(auth_client, "tools/list", {}, path=path).status_code == 401

    wrong = {"Authorization": "Bearer not-the-password"}
    assert rpc(auth_client, "tools/list", {}, headers=wrong).status_code == 401


def test_mcp_with_password_bearer_runs_tools_in_process(auth_client):
    headers = {"Authorization": f"Bearer {PASSWORD}"}

    assert list_tool_names(auth_client, headers=headers) == (
        ALLOWLISTED_TOOLS | CATALOG_TOOLS
    )
    # The in-process API call behind the tool carries the internal token.
    result = call_tool(auth_client, "get_training_queue", headers=headers)
    assert result["isError"] is False
    assert result["structuredContent"]["state"] in {"paused", "running"}
    # The same API route is still protected from outside.
    assert auth_client.get("/api/training/queue").status_code == 401


async def test_mcp_path_middleware_rewrites_only_the_bare_path():
    from app.main import McpPathMiddleware

    seen = []

    async def downstream(scope, receive, send):
        seen.append(scope)

    middleware = McpPathMiddleware(downstream)

    async def run(scope):
        await middleware(scope, None, None)

    bare = {
        "type": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"a=1",
    }
    await run(bare)
    await run({"type": "http", "path": "/mcp/", "raw_path": b"/mcp/"})
    await run({"type": "http", "path": "/mcpx", "raw_path": b"/mcpx"})
    await run({"type": "websocket", "path": "/mcp", "raw_path": b"/mcp"})

    assert seen[0]["path"] == "/mcp/"
    assert seen[0]["raw_path"] == b"/mcp/"
    assert seen[0]["query_string"] == b"a=1"
    assert (seen[1]["path"], seen[2]["path"], seen[3]["path"]) == (
        "/mcp/",
        "/mcpx",
        "/mcp",
    )
    assert bare["path"] == "/mcp"
