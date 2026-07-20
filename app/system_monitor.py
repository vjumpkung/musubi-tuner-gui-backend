"""Compatibility alias for :mod:`app.services.system_monitor`."""

import sys

from .routes import system_monitor as _routes
from .services import system_monitor as _implementation

_implementation.router = _routes.router
_implementation.read_system_resources = _routes.read_system_resources

sys.modules[__name__] = _implementation
