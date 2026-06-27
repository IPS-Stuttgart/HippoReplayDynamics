"""Strict runtime validation for position-decoding configuration values.

The position-decoding validation helpers use public configuration fields in
array slicing, fold construction, spike-count filtering, and explicit training
frame masks.  Invalid integer knobs or non-boolean masks should be rejected
before those operations so callers do not get silent window truncation,
accidental truthiness, or unrelated NumPy/type errors.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_position_decoding_config_validation_patch_applied"
_VALIDATE_WRAPPER_FLAG = "_position_decoding_config_validation_validate_wrapper"
_MASK_WRAPPER_FLAG = "_position_decoding_config_validation_mask_wrapper"


def apply_position_decoding_config_validation_patch() -> None:
    """Install strict validation on position-decoding runtime entry points."""

    from . import position_validation as validation

    current_validate = validation.validate_session_position_decoding
    current_mask_encoder = validation.fit_place_field_encoding_for_position_mask
    validate_is_current = bool(getattr(current_validate, _VALIDATE_WRAPPER_FLAG, False))
    mask_is_current = bool(getattr(current_mask_encoder, _MASK_WRAPPER_FLAG, False))
    if getattr(validation, _PATCHED_FLAG, False) and validate_is_current and mask_is_current:
        return

    if not validate_is_current:
        original_validate_session_position_decoding = current_validate

        @wraps(original_validate_session_position_decoding)
        def validate_session_position_decoding_with_config_validation(session: Any, config: Any = None) -> Any:
            config = validation.PositionDecodingConfig() if config is None else config
            config = _validated_position_decoding_config(config)
            return original_validate_session_position_decoding(session, config)

        setattr(validate_session_position_decoding_with_config_validation, _VALIDATE_WRAPPER_FLAG, True)
        validation.validate_session_position_decoding = validate_session_position_decoding_with_config_validation

    if not mask_is_current:
        original_fit_place_field_encoding_for_position_mask = current_mask_encoder

        @wraps(original_fit_place_field_encoding_for_position_mask)
        def fit_place_field_encoding_for_position_mask_with_mask_validation(session: Any, train_frame_mask: Any, config: Any = None) -> Any:
            position = validation._clean_position(session.position)
            mask = _validated_train_frame_mask(train_frame_mask, position.shape[0])
            return original_fit_place_field_encoding_for_position_mask(session, mask, config)

        setattr(fit_place_field_encoding_for_position_mask_with_mask_validation, _MASK_WRAPPER_FLAG, True)
        validation.fit_place_field_encoding_for_position_mask = fit_place_field_encoding_for_position_mask_with_mask_validation

    setattr(validation, _PATCHED_FLAG, True)


def _validated_position_decoding_config(config: Any) -> Any:
    updates = {
        "decode_bin_s": _positive_finite_scalar("decode_bin_s", getattr(config, "decode_bin_s")),
        "n_folds": _positive_integer("n_folds", getattr(config, "n_folds")),
        "random_seed": _nonnegative_integer("random_seed", getattr(config, "random_seed")),
        "min_spikes_per_window": _nonnegative_integer(
            "min_spikes_per_window",
            getattr(config, "min_spikes_per_window"),
        ),
    }
    max_windows = getattr(config, "max_windows_per_session", None)
    if max_windows is not None:
        updates["max_windows_per_session"] = _positive_integer("max_windows_per_session", max_windows)
    return replace(config, **updates)


def _validate_position_decoding_config(config: Any) -> None:
    _validated_position_decoding_config(config)


def _validated_train_frame_mask(train_frame_mask: Any, expected_length: int) -> np.ndarray:
    raw = np.asarray(train_frame_mask)
    if raw.shape != (int(expected_length),):
        raise ValueError("train_frame_mask must have one value per cleaned position frame")
    if np.issubdtype(raw.dtype, np.bool_):
        return raw.astype(bool, copy=False)
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("train_frame_mask must contain boolean or 0/1 values") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("train_frame_mask must contain finite boolean or 0/1 values")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("train_frame_mask must contain boolean or 0/1 values")
    return numeric.astype(bool)


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