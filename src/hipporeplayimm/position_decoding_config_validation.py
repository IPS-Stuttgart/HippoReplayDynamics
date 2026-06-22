"""Strict runtime validation for position-decoding configuration values.

The position-decoding validation helper uses several public config fields in
array slicing, fold construction, and spike-count filtering.  Invalid integer
knobs should be rejected before those operations so callers do not get silent
window truncation or unrelated NumPy/type errors.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_position_decoding_config_validation_patch_applied"


def apply_position_decoding_config_validation_patch() -> None:
    """Install strict validation on ``validate_session_position_decoding``."""

    from . import position_validation as validation

    if getattr(validation, _PATCHED_FLAG, False):
        return

    original_validate_session_position_decoding = validation.validate_session_position_decoding

    @wraps(original_validate_session_position_decoding)
    def validate_session_position_decoding_with_config_validation(session: Any, config: Any = None) -> Any:
        config = validation.PositionDecodingConfig() if config is None else config
        _validate_position_decoding_config(config)
        return original_validate_session_position_decoding(session, config)

    validation.validate_session_position_decoding = validate_session_position_decoding_with_config_validation
    setattr(validation, _PATCHED_FLAG, True)


def _validate_position_decoding_config(config: Any) -> None:
    _positive_finite_scalar("decode_bin_s", getattr(config, "decode_bin_s"))
    _positive_integer("n_folds", getattr(config, "n_folds"))
    _nonnegative_integer("min_spikes_per_window", getattr(config, "min_spikes_per_window"))
    max_windows = getattr(config, "max_windows_per_session", None)
    if max_windows is not None:
        _positive_integer("max_windows_per_session", max_windows)


def _positive_finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _positive_integer(name: str, value: Any) -> int:
    integer = _integer_value(name, value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _nonnegative_integer(name: str, value: Any) -> int:
    integer = _integer_value(name, value)
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _integer_value(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, str):
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        return integer
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite integer")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} must be an integer")
    return integer


__all__ = ["apply_position_decoding_config_validation_patch"]
