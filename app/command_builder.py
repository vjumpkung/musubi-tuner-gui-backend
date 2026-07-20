"""Compatibility alias for :mod:`app.utils.command_builder`."""

import sys

from .utils import command_builder as _implementation

sys.modules[__name__] = _implementation
