import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.websockets import WebSocketDisconnect

from app.services import auth as auth_service

PASSWORD = "correct horse battery"
XHR = {"X-Requested-With": "XMLHttpRequest"}
QUEUE_URL = "/api/training/queue"
PAUSE_URL = "/api/training/queue/pause"
BEARER = {"Authorization": f"Bearer {PASSWORD}"}


@pytest.fixture
def auth_client(make_client, monkeypatch):
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", PASSWORD)
    monkeypatch.setattr(auth_service, "FAILURE_DELAY_SECONDS", 0)
    with make_client() as test_client:
        yield test_client


def login(client, password=PASSWORD, **kwargs):
    return client.post("/api/auth/login", json={"password": password}, **kwargs)


def cookie_attributes(response) -> list[str]:
    return [part.strip().lower() for part in response.headers["set-cookie"].split(";")]


def cookie_header(token: str) -> dict[str, str]:
    return {"Cookie": f"{auth_service.SESSION_COOKIE}={token}"}


def test_auth_disabled_leaves_api_open(client):
    assert client.get(QUEUE_URL).status_code == 200
    assert client.get("/api/auth/status").json() == {
        "enabled": False,
        "authenticated": True,
    }
    response = login(client, "anything")
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "authenticated": True}
    assert "set-cookie" not in response.headers
    assert client.post(PAUSE_URL).status_code == 200


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_password_disables_auth(make_client, monkeypatch, value):
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", value)
    with make_client() as client:
        assert client.get(QUEUE_URL).status_code == 200


def test_settings_strip_password_and_hide_it_from_repr(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", "  super-secret  ")
    settings = Settings()
    assert settings.password == "super-secret"
    assert "super-secret" not in repr(settings)


def test_startup_logs_auth_mode_without_secrets(make_client, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="app.main"):
        with make_client():
            pass
        monkeypatch.setenv("MUSUBI_GUI_PASSWORD", PASSWORD)
        with make_client():
            pass

    assert "Password authentication disabled" in caplog.text
    assert "Password authentication enabled" in caplog.text
    assert PASSWORD not in caplog.text


def test_enabled_requires_authentication(auth_client):
    response = auth_client.get(QUEUE_URL)
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
    status = auth_client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json() == {"enabled": True, "authenticated": False}


def test_login_wrong_then_right_password(auth_client):
    wrong = login(auth_client, "nope")
    assert wrong.status_code == 401
    assert wrong.json() == {"detail": "Incorrect password"}
    assert "set-cookie" not in wrong.headers

    response = login(auth_client)
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "authenticated": True}
    attributes = cookie_attributes(response)
    assert attributes[0].startswith(f"{auth_service.SESSION_COOKIE}=")
    assert "httponly" in attributes
    assert "samesite=lax" in attributes
    assert "path=/" in attributes
    assert f"max-age={auth_service.SESSION_TTL_SECONDS}" in attributes
    assert "secure" not in attributes
    assert PASSWORD not in response.headers["set-cookie"]

    assert auth_client.get(QUEUE_URL).status_code == 200
    assert auth_client.get("/api/auth/status").json() == {
        "enabled": True,
        "authenticated": True,
    }


def test_login_rejects_empty_password(auth_client):
    assert login(auth_client, "").status_code == 422


def test_bearer_password_grants_access(auth_client):
    assert auth_client.get(QUEUE_URL, headers=BEARER).status_code == 200
    # Bearer requests are exempt from the CSRF header requirement.
    assert auth_client.post(PAUSE_URL, headers=BEARER).status_code == 200

    wrong = auth_client.get(QUEUE_URL, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert wrong.json() == {"detail": "Authentication required"}


def test_bearer_internal_token_grants_access(auth_client):
    token = auth_client.app.state.auth.internal_token
    headers = {"Authorization": f"Bearer {token}"}
    assert auth_client.get(QUEUE_URL, headers=headers).status_code == 200
    assert auth_client.post(PAUSE_URL, headers=headers).status_code == 200


def test_invalid_bearer_is_not_rescued_by_cookie(auth_client):
    assert login(auth_client).status_code == 200
    response = auth_client.get(QUEUE_URL, headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def _concurrent_attempts(slow, fast):
    """Run `slow` in a thread, then `fast` while its failure delay is pending."""
    started = threading.Event()

    def run_slow():
        started.set()
        return slow()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_slow)
        started.wait()
        time.sleep(0.15)
        second = fast()
        first = future.result(timeout=10)
    return first, second


def test_failed_login_throttles_concurrent_attempts(auth_client, monkeypatch):
    monkeypatch.setattr(auth_service, "FAILURE_DELAY_SECONDS", 0.6)

    first, second = _concurrent_attempts(
        slow=lambda: login(auth_client, "wrong-one"),
        fast=lambda: login(auth_client, "wrong-two"),
    )

    assert first.status_code == 401
    assert second.status_code == 429
    assert second.headers["retry-after"] == "1"
    assert second.json() == {"detail": "Too many attempts, try again shortly"}

    # Once the delay elapsed the next attempt is evaluated normally again.
    assert login(auth_client).status_code == 200


def test_bearer_attempts_share_the_throttle(auth_client, monkeypatch):
    monkeypatch.setattr(auth_service, "FAILURE_DELAY_SECONDS", 0.6)

    def wrong_bearer(value):
        return lambda: auth_client.get(
            QUEUE_URL, headers={"Authorization": f"Bearer {value}"}
        )

    first, second = _concurrent_attempts(
        slow=wrong_bearer("wrong-one"), fast=wrong_bearer("wrong-two")
    )
    assert first.status_code == 401
    assert second.status_code == 429
    assert second.headers["retry-after"] == "1"

    # The login endpoint is throttled by the same lock.
    first, second = _concurrent_attempts(
        slow=wrong_bearer("wrong-one"), fast=lambda: login(auth_client, "wrong-two")
    )
    assert first.status_code == 401
    assert second.status_code == 429


def test_invalid_session_tokens_are_rejected(auth_client):
    manager = auth_client.app.state.auth
    now = int(time.time())
    ttl = auth_service.SESSION_TTL_SECONDS
    good = manager.issue_token()
    expires, signature = good.split(".")
    bad_tokens = {
        "expired": manager.issue_token(now=now - ttl - 60),
        "too far in the future": manager.issue_token(now=now + ttl * 2),
        "tampered signature": f"{expires}.{'A' * len(signature)}",
        "tampered expiry": f"{int(expires) + 1}.{signature}",
        "no signature": f"{expires}.",
        "no expiry": f".{signature}",
        "not a number": f"abc.{signature}",
        "no separator": "garbage",
    }
    for label, token in bad_tokens.items():
        response = auth_client.get(QUEUE_URL, headers=cookie_header(token))
        assert response.status_code == 401, label
        status = auth_client.get("/api/auth/status", headers=cookie_header(token))
        assert status.json()["authenticated"] is False, label

    assert auth_client.get(QUEUE_URL, headers=cookie_header(good)).status_code == 200


def test_malformed_cookie_header_is_rejected(auth_client):
    for raw in ("garbage", "=;=;", auth_service.SESSION_COOKIE, "a=b; ;;; c"):
        response = auth_client.get(QUEUE_URL, headers={"Cookie": raw})
        assert response.status_code == 401, raw


def test_logout_clears_cookie(auth_client):
    assert login(auth_client).status_code == 200
    assert auth_client.get(QUEUE_URL).status_code == 200

    response = auth_client.post("/api/auth/logout", headers=XHR)
    assert response.status_code == 204
    assert response.content == b""
    attributes = cookie_attributes(response)
    assert attributes[0].startswith(f"{auth_service.SESSION_COOKIE}=")
    assert "max-age=0" in attributes
    assert "path=/" in attributes

    assert auth_client.get(QUEUE_URL).status_code == 401
    assert auth_client.get("/api/auth/status").json()["authenticated"] is False


def test_password_change_invalidates_sessions(make_client, monkeypatch):
    monkeypatch.setattr(auth_service, "FAILURE_DELAY_SECONDS", 0)
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", "password-one")
    with make_client() as first:
        assert login(first, "password-one").status_code == 200
        token = first.cookies.get(auth_service.SESSION_COOKIE)
    assert token
    cookie = cookie_header(token)

    # Same password, new process, same data root: the session secret persisted.
    with make_client() as restarted:
        assert restarted.get(QUEUE_URL, headers=cookie).status_code == 200

    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", "password-two")
    with make_client() as changed:
        assert changed.get(QUEUE_URL, headers=cookie).status_code == 401
        assert login(changed, "password-one").status_code == 401
        assert login(changed, "password-two").status_code == 200


def test_spa_is_public_but_docs_and_mcp_are_gated(make_client, monkeypatch, paths):
    web = paths["workspace"] / "web"
    web.mkdir()
    (web / "index.html").write_text("<html><body>spa</body></html>")
    monkeypatch.setenv("MUSUBI_GUI_WEB_DIR", str(web))
    monkeypatch.setenv("MUSUBI_GUI_PASSWORD", PASSWORD)
    with make_client() as client:
        assert client.get("/").status_code == 200
        assert "spa" in client.get("/some/route").text
        assert client.get("/api/auth/status").status_code == 200
        for url in ("/api/training/queue", "/docs", "/redoc", "/openapi.json"):
            assert client.get(url).status_code == 401, url
        for url in ("/mcp", "/mcp/", "/mcp/anything"):
            assert client.get(url).status_code == 401, url

        assert client.get("/docs", headers=BEARER).status_code == 200


def test_cookie_authenticated_mutations_require_xhr_header(auth_client):
    assert login(auth_client).status_code == 200

    blocked = auth_client.post(PAUSE_URL)
    assert blocked.status_code == 403
    assert blocked.json() == {"detail": "Missing X-Requested-With header"}
    # Safe methods are not subject to the CSRF check.
    assert auth_client.get(QUEUE_URL).status_code == 200

    assert auth_client.post(PAUSE_URL, headers=XHR).status_code == 200

    auth_client.cookies.clear()
    assert auth_client.post(PAUSE_URL, headers=BEARER).status_code == 200


@pytest.mark.parametrize(
    ("forwarded", "secure"),
    [
        (None, False),
        ("http", False),
        ("https", True),
        ("https, http", True),
        ("http, https", False),
    ],
)
def test_cookie_secure_follows_forwarded_proto(auth_client, forwarded, secure):
    headers = {"X-Forwarded-Proto": forwarded} if forwarded else {}
    response = login(auth_client, headers=headers)
    assert response.status_code == 200
    assert ("secure" in cookie_attributes(response)) is secure


def test_websocket_is_rejected_without_credentials(auth_client):
    with pytest.raises(WebSocketDisconnect) as rejected:
        with auth_client.websocket_connect("/api/socket.io/"):
            pass
    assert rejected.value.code == 1008

    # With credentials the middleware lets it through (no route exists here).
    with pytest.raises(WebSocketDisconnect) as passed:
        with auth_client.websocket_connect("/api/socket.io/", headers=BEARER):
            pass
    assert passed.value.code != 1008
