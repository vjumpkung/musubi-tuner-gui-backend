"""Compatibility alias for :mod:`app.utils.dataset_rules`."""

import sys

from .utils import dataset_rules as _implementation

sys.modules[__name__] = _implementation
