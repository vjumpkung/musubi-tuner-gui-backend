"""Compatibility alias for :mod:`app.services.queue_runner`."""

import sys

from .services import queue_runner as _implementation

sys.modules[__name__] = _implementation
