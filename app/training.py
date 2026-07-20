"""Compatibility alias for :mod:`app.routes.training`."""

import sys

from .routes import training as _implementation

sys.modules[__name__] = _implementation
