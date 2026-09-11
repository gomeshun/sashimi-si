"""Compatibility import for :mod:`sashimi_si._itamae_migration`."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module('sashimi_si._itamae_migration')
_sys.modules[__name__] = _module
