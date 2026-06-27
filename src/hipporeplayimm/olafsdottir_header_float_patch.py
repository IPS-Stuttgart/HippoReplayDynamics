"""Robust Axona header numeric parsing for Olafsdottir2016 loaders."""

from __future__ import annotations

import re
from collections.abc import Mapping

_FLOAT_TOKEN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_PATCH_FLAG = "_olafsdottir_header_float_patch_applied"


def _header_float(header: Mapping[str, str], key: str, default: float) -> float:
    raw = header.get(key)
    if raw is None:
        return float(default)
    match = _FLOAT_TOKEN.search(str(raw))
    return float(match.group(0)) if match else float(default)


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
