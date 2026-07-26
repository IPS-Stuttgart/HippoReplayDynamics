"""Validate boolean flags in the standard place-field encoding configuration."""

from __future__ import annotations

from functools import wraps
import sys

import numpy as np

_PATCH_MARKER = "_encoding_config_boolean_validation_patch"
_PATCH_VERSION = 1
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_BOOLEAN_FIELDS = ("use_excitatory", "exclude_ripple_intervals")


def _validate_boolean_config_value(config: object, name: str) -> bool:
    """Return a strict Python/NumPy boolean scalar without truthiness coercion."""

    message = f"{name} must be a boolean scalar"
    value = getattr(config, name, None)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(message)
    try:
        item = array.item()
    except ValueError as exc:
        raise ValueError(message) from exc
    if not isinstance(item, (bool, np.bool_)):
        raise ValueError(message)
    return bool(item)


def _synchronize_aliases(previous: object, patched: object) -> None:
    """Refresh package-local aliases imported before the validator was patched."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_validate_encoding_config", None) is previous:
            module._validate_encoding_config = patched


def apply_encoding_config_boolean_validation_patch() -> None:
    """Install strict validation for ``EncodingConfig`` boolean options."""

    from . import encoding

    current = encoding._validate_encoding_config
    if getattr(current, _PATCH_MARKER, None) == _PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_aliases(previous, current)
        return

    previous = current

    @wraps(previous)
    def _validate_encoding_config(config):
        for name in _BOOLEAN_FIELDS:
            _validate_boolean_config_value(config, name)
        return previous(config)

    setattr(_validate_encoding_config, _PATCH_MARKER, _PATCH_VERSION)
    setattr(_validate_encoding_config, _ORIGINAL_ATTR, previous)
    encoding._validate_encoding_config = _validate_encoding_config
    _synchronize_aliases(previous, _validate_encoding_config)


__all__ = ["apply_encoding_config_boolean_validation_patch"]
