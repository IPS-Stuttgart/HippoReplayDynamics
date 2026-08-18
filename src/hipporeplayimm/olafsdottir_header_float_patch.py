"""Robust Axona header numeric parsing for Olafsdottir2016 loaders."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

_FLOAT_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.])"
)
_PATCH_FLAG = "_olafsdottir_header_float_patch_applied"
_STRICTLY_POSITIVE_KEYS = frozenset({"timebase", "sample_rate", "pixels_per_metre"})


def _validated_header_float(key: str, value: float) -> float:
    if key in _STRICTLY_POSITIVE_KEYS and value <= 0.0:
        raise ValueError(f"Axona header {key} must be positive")
    return value


def _header_float(header: Mapping[str, str], key: str, default: float) -> float:
    raw = header.get(key)
    if raw is None:
        return _validated_header_float(key, float(default))
    match = _FLOAT_TOKEN.search(str(raw))
    if match is None:
        return _validated_header_float(key, float(default))
    value = float(match.group(0))
    if not math.isfinite(value):
        value = float(default)
    return _validated_header_float(key, value)


def apply_olafsdottir_header_float_patch() -> None:
    """Install scientific-notation-aware Axona header parsing."""

    from . import olafsdottir2016

    if (
        getattr(olafsdottir2016, _PATCH_FLAG, False)
        and getattr(olafsdottir2016, "_header_float", None) is _header_float
    ):
        return
    _header_float.__name__ = olafsdottir2016._header_float.__name__
    _header_float.__doc__ = olafsdottir2016._header_float.__doc__
    olafsdottir2016._header_float = _header_float
    setattr(olafsdottir2016, _PATCH_FLAG, True)


__all__ = ["apply_olafsdottir_header_float_patch"]
