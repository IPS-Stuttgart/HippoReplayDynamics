"""Validate boolean flags in the standard place-field encoding configuration."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCH_MARKER = "_encoding_config_boolean_validation_patch"
_PATCH_VERSION = 3
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_DISPATCH_PATCH_MARKER = "_encoding_config_boolean_kinematics_guard"
_DISPATCH_ORIGINAL_ATTR = "__hipporeplayimm_boolean_kinematics_original__"
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


def _synchronize_validator_aliases(previous: object, patched: object) -> None:
    """Refresh package-local validator aliases imported before patching."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_validate_encoding_config", None) is previous:
            module._validate_encoding_config = patched


def _encoding_config_for_kinematics_call(
    original: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> object | None:
    """Return the nested encoding config for patched encoder entry points."""

    function_name = getattr(original, "__name__", "")
    config = args[0] if args else kwargs.get("config")
    if config is None:
        return None
    if function_name == "fit_place_field_encoding":
        return config
    if function_name == "fit_clusterless_mark_encoding":
        return getattr(config, "encoding", None)
    return None


def _install_kinematics_validation_guard() -> None:
    """Validate flags before run-local wrappers access session attributes."""

    from . import place_field_run_local_kinematics as kinematics

    current = kinematics._call_with_run_local_kinematics
    if getattr(current, _DISPATCH_PATCH_MARKER, None) == _PATCH_VERSION:
        return

    previous = current

    @wraps(previous)
    def _call_with_run_local_kinematics(
        original: Any,
        session: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        config = _encoding_config_for_kinematics_call(original, args, kwargs)
        if config is not None:
            for name in _BOOLEAN_FIELDS:
                _validate_boolean_config_value(config, name)
        return previous(original, session, *args, **kwargs)

    setattr(
        _call_with_run_local_kinematics,
        _DISPATCH_PATCH_MARKER,
        _PATCH_VERSION,
    )
    setattr(
        _call_with_run_local_kinematics,
        _DISPATCH_ORIGINAL_ATTR,
        previous,
    )
    kinematics._call_with_run_local_kinematics = _call_with_run_local_kinematics


def apply_encoding_config_boolean_validation_patch() -> None:
    """Install strict validation for ``EncodingConfig`` boolean options."""

    from . import encoding

    current = encoding._validate_encoding_config
    if getattr(current, _PATCH_MARKER, None) == _PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_validator_aliases(previous, current)
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
        _synchronize_validator_aliases(previous, _validate_encoding_config)

    _install_kinematics_validation_guard()


__all__ = ["apply_encoding_config_boolean_validation_patch"]
