"""Compatibility alias for :mod:`app.utils.process`."""

import sys

from .utils import process as _implementation

sys.modules[__name__] = _implementation
