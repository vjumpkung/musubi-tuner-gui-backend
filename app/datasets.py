"""Compatibility alias for :mod:`app.routes.datasets`."""

import sys

from .routes import datasets as _implementation

sys.modules[__name__] = _implementation
