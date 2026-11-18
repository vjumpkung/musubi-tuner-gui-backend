"""Password authentication API routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from starlette.responses import JSONResponse

from ..schemas.auth import LoginRequest
from ..services.auth import (
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    AuthManager,
    cookie_secure,
    session_token_from_cookie_header,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth(request: Request) -> AuthManager:
    return request.app.state.auth


@router.get("/status")
async def read_status(request: Request) -> dict:
    auth = _auth(request)
    if not auth.enabled:
        return {"enabled": False, "authenticated": True}
    token = session_token_from_cookie_header(request.headers.get("cookie"))
    return {
        "enabled": True,
        "authenticated": token is not None and auth.verify_token(token),
    }


@router.post("/login")
async def login(body: LoginRequest, request: Request) -> Response:
    auth = _auth(request)
    if not auth.enabled:
        return JSONResponse({"enabled": False, "authenticated": True})
    result = await auth.check_secret(body.password.get_secret_value())
    if result == "throttled":
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many attempts, try again shortly"},
            headers={"Retry-After": "1"},
        )
    if result != "ok":
        return JSONResponse(status_code=401, content={"detail": "Incorrect password"})
    response = JSONResponse({"enabled": True, "authenticated": True})
    response.set_cookie(
        SESSION_COOKIE,
        auth.issue_token(),
        max_age=SESSION_TTL_SECONDS,
        path="/",
        httponly=True,
        samesite="lax",
        secure=cookie_secure(request),
    )
    return response


@router.post("/logout", status_code=204)
async def logout(request: Request) -> Response:
    response = Response(status_code=204)
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=cookie_secure(request),
    )
    return response
