"""Compatibility alias for :mod:`app.services.downloads`."""

import sys

from .routes import downloads as _routes
from .services import downloads as _implementation

_implementation.DownloadRequest = _routes.DownloadRequest
_implementation.router = _routes.router
_implementation.start_download = _routes.start_download
_implementation.read_download = _routes.read_download
_implementation.cancel_download = _routes.cancel_download

sys.modules[__name__] = _implementation
