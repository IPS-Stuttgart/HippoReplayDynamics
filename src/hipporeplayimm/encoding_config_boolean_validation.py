"""Validate boolean flags in the standard place-field encoding configuration."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCH_MARKER = "_encoding_config_boolean_validation_patch"
_PATCH_VERSION = 2
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_FIT_PATCH_MARKER = "_encoding_config_boolean_fit_guard"
_FIT_ORIGINAL_ATTR = "__hipporeplayimm_boolean_fit_original__"
_BOOLEAN_FIELDS = ("use_excitatory", "exclude_ripple_intervals")


def _validate_boolean_config_value(config: object, name: str) -> bool:
    """Return a strict Python/NumPy boolean scalar without truthiness coercion."""

    message = f"{name} must be a boolean scalar"
    value = getattr(config, name, None)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(message) from exc
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.bool_):
        raise TypeError(message)
    try:
        item = array.item()
    except ValueError as exc:
        raise TypeError(message) from exc
    if not isinstance(item, (bool, np.bool_)):
        raise TypeError(message)
    return bool(item)


def _synchronize_aliases(
    attribute_name: str,
    previous: object,
    patched: object,
) -> None:
    """Refresh package-local aliases imported before a callable was patched."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, attribute_name, None) is previous:
            setattr(module, attribute_name, patched)


def _install_place_field_fit_guard(encoding: Any) -> None:
    """Validate config before outer wrappers can access session attributes."""

    current = encoding.fit_place_field_encoding
    if getattr(current, _FIT_PATCH_MARKER, None) == _PATCH_VERSION:
        original = getattr(current, _FIT_ORIGINAL_ATTR, None)
        if original is not None:
            _synchronize_aliases("fit_place_field_encoding", original, current)
        return

    previous = current

    @wraps(previous)
    def fit_place_field_encoding(session, *args, **kwargs):
        config = args[0] if args else kwargs.get("config")
        if config is not None:
            for name in _BOOLEAN_FIELDS:
                _validate_boolean_config_value(config, name)
        return previous(session, *args, **kwargs)

    setattr(fit_place_field_encoding, _FIT_PATCH_MARKER, _PATCH_VERSION)
    setattr(fit_place_field_encoding, _FIT_ORIGINAL_ATTR, previous)
    encoding.fit_place_field_encoding = fit_place_field_encoding
    _synchronize_aliases(
        "fit_place_field_encoding",
        previous,
        fit_place_field_encoding,
    )


def apply_encoding_config_boolean_validation_patch() -> None:
    """Install strict validation for ``EncodingConfig`` boolean options."""

    from . import encoding

    current = encoding._validate_encoding_config
    if getattr(current, _PATCH_MARKER, None) == _PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_aliases("_validate_encoding_config", previous, current)
    else:
        previous = current

        @wraps(previous)
        def _validate_encoding_config(config):
            for name in _BOOLEAN_FIELDS:
                _validate_boolean_config_value(config, name)
            return previous(config)

        setattr(_validate_encoding_config, _PATCH_MARKER, _PATCH_VERSION)
        setattr(_validate_encoding_config, _ORIGINAL_ATTR, previous)
        encoding._validate_encoding_config = _validate_encoding_config
        _synchronize_aliases(
            "_validate_encoding_config",
            previous,
            _validate_encoding_config,
        )

    _install_place_field_fit_guard(encoding)


__all__ = ["apply_encoding_config_boolean_validation_patch"]
