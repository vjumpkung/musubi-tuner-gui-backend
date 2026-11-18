"""Optional shared-password authentication.

Authentication is enabled only when ``MUSUBI_GUI_PASSWORD`` is set. Browser
sessions use a stateless signed cookie; scripts and the MCP endpoint may send the
password as a Bearer token. The password, tokens, and session secret must never be
logged or included in responses.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import time
from http.cookies import CookieError, SimpleCookie
from typing import Literal

from ..db import Database

SESSION_COOKIE = "musubi_gui_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600
FAILURE_DELAY_SECONDS = 1.0
SECRET_SETTING_KEY = "auth_session_secret"

CheckResult = Literal["ok", "bad", "throttled"]


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def cookie_secure(scope_or_request) -> bool:
    """True when the client reached us over HTTPS (directly or via a proxy)."""
    if hasattr(scope_or_request, "scope"):
        scope = scope_or_request.scope
    else:
        scope = scope_or_request
    if scope.get("scheme") in {"https", "wss"}:
        return True
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-proto":
            first = value.decode("latin-1").split(",")[0].strip().lower()
            return first == "https"
    return False


def session_token_from_cookie_header(raw: str | None) -> str | None:
    """Extract the session cookie value from a Cookie header; tolerate junk."""
    if not raw:
        return None
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(raw)
    except CookieError:
        return None
    morsel = jar.get(SESSION_COOKIE)
    return morsel.value if morsel is not None else None


class AuthManager:
    def __init__(self, password: str | None):
        self._password = password
        self.internal_token = secrets.token_urlsafe(32)
        self._secret: bytes | None = None
        self._lock: asyncio.Lock | None = None

    @property
    def enabled(self) -> bool:
        return self._password is not None

    async def load_secret(self, db: Database) -> None:
        """Load (or create) the persisted session-signing secret."""
        stored = await db.get_setting(SECRET_SETTING_KEY, "")
        if not stored:
            stored = secrets.token_hex(32)
            await db.set_setting(SECRET_SETTING_KEY, stored)
        self._secret = stored.encode("ascii")
        self._lock = asyncio.Lock()

    def _signing_key(self) -> bytes | None:
        if self._password is None or self._secret is None:
            return None
        return hmac.new(self._secret, self._password.encode(), hashlib.sha256).digest()

    def _sign(self, key: bytes, expires: int) -> str:
        return _b64url(hmac.new(key, str(expires).encode(), hashlib.sha256).digest())

    def issue_token(self, now: int | None = None) -> str:
        key = self._signing_key()
        if key is None:
            raise RuntimeError("Authentication is not ready")
        current = int(time.time()) if now is None else now
        expires = current + SESSION_TTL_SECONDS
        return f"{expires}.{self._sign(key, expires)}"

    def verify_token(self, token: str) -> bool:
        key = self._signing_key()
        if key is None:
            return False
        expires_text, separator, signature = token.partition(".")
        if not separator or not signature:
            return False
        try:
            expires = int(expires_text)
        except ValueError:
            return False
        now = int(time.time())
        if expires < now or expires > now + SESSION_TTL_SECONDS:
            return False
        return hmac.compare_digest(
            signature.encode("utf-8"), self._sign(key, expires).encode("ascii")
        )

    def is_internal_token(self, candidate: str) -> bool:
        return hmac.compare_digest(
            candidate.encode("utf-8"), self.internal_token.encode("utf-8")
        )

    async def check_secret(self, candidate: str) -> CheckResult:
        """Check a password attempt; failures serialize behind a delay.

        While one failed attempt is sleeping, concurrent attempts are throttled
        instead of queued, so guessing is limited to ~1 attempt per delay.
        """
        if self._password is None or self._lock is None:
            return "bad"
        lock = self._lock
        if lock.locked():
            return "throttled"
        if hmac.compare_digest(
            hashlib.sha256(candidate.encode("utf-8")).digest(),
            hashlib.sha256(self._password.encode("utf-8")).digest(),
        ):
            return "ok"
        async with lock:
            await asyncio.sleep(FAILURE_DELAY_SECONDS)
        return "bad"
