"""Origin validation for browser-initiated connections."""

from __future__ import annotations

from urllib.parse import urlsplit


def origin_allowed(
    origin: str | None, environ: dict, cors_origins: tuple[str, ...]
) -> bool:
    """Allow requests without an Origin, same-origin requests, or configured origins.

    Same-origin compares the Origin host with the Host header and the first
    ``X-Forwarded-Host`` value, so reverse proxies (e.g. RunPod) keep working.
    """
    if not origin:
        return True
    netloc = urlsplit(origin).netloc
    if netloc:
        hosts = [environ.get("HTTP_HOST")]
        forwarded = environ.get("HTTP_X_FORWARDED_HOST")
        if forwarded:
            hosts.append(forwarded.split(",")[0].strip())
        if netloc in hosts:
            return True
    return origin in cors_origins
