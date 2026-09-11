"""Compatibility import for :mod:`sashimi_si._physics`."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module('sashimi_si._physics')
_sys.modules[__name__] = _module
