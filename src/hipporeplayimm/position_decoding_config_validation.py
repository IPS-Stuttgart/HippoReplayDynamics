"""Strict runtime validation for position-decoding configuration values.

The position-decoding validation helpers use public configuration fields in
array slicing, fold construction, spike-count filtering, and explicit training
frame masks.  Invalid integer knobs or non-boolean masks should be rejected
before those operations so callers do not get silent window truncation,
accidental truthiness, or unrelated NumPy/type errors.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_position_decoding_config_validation_patch_applied"
_VALIDATE_WRAPPER_FLAG = "_position_decoding_config_validation_validate_wrapper"
_MASK_WRAPPER_FLAG = "_position_decoding_config_validation_mask_wrapper"
_COUNTS_WRAPPER_FLAG = "_position_decoding_config_validation_counts_wrapper"
_DISTANCE_WRAPPER_FLAG = "_position_decoding_distance_overflow_wrapper"


def apply_position_decoding_config_validation_patch() -> None:
    """Install strict validation on position-decoding runtime entry points."""

    from . import position_validation as validation

    current_validate = validation.validate_session_position_decoding
    current_mask_encoder = validation.fit_place_field_encoding_for_position_mask
    current_spike_counts_for_window = validation._spike_counts_for_window
    current_distance = validation._distance
    validate_is_current = bool(getattr(current_validate, _VALIDATE_WRAPPER_FLAG, False))
    mask_is_current = bool(getattr(current_mask_encoder, _MASK_WRAPPER_FLAG, False))
    counts_is_current = bool(getattr(current_spike_counts_for_window, _COUNTS_WRAPPER_FLAG, False))
    distance_is_current = bool(getattr(current_distance, _DISTANCE_WRAPPER_FLAG, False))
    if (
        getattr(validation, _PATCHED_FLAG, False)
        and validate_is_current
        and mask_is_current
        and counts_is_current
        and distance_is_current
    ):
        return

    if not validate_is_current:
        original_validate_session_position_decoding = current_validate

        @wraps(original_validate_session_position_decoding)
        def validate_session_position_decoding_with_config_validation(session: Any, config: Any = None) -> Any:
            config = validation.PositionDecodingConfig() if config is None else config
            config = _validated_position_decoding_config(config)
            _validate_position_decoding_cell_ids(session, config.encoding)
            return original_validate_session_position_decoding(session, config)

        setattr(validate_session_position_decoding_with_config_validation, _VALIDATE_WRAPPER_FLAG, True)
        validation.validate_session_position_decoding = validate_session_position_decoding_with_config_validation

    if not mask_is_current:
        original_fit_place_field_encoding_for_position_mask = current_mask_encoder

        @wraps(original_fit_place_field_encoding_for_position_mask)
        def fit_place_field_encoding_for_position_mask_with_mask_validation(session: Any, train_frame_mask: Any, config: Any = None) -> Any:
            position = validation._clean_position(session.position)
            mask = _validated_train_frame_mask(train_frame_mask, position.shape[0])
            encoding_config = validation.EncodingConfig() if config is None else config
            _validate_position_decoding_cell_ids(session, encoding_config)
            return original_fit_place_field_encoding_for_position_mask(session, mask, encoding_config)

        setattr(fit_place_field_encoding_for_position_mask_with_mask_validation, _MASK_WRAPPER_FLAG, True)
        validation.fit_place_field_encoding_for_position_mask = fit_place_field_encoding_for_position_mask_with_mask_validation

    if not counts_is_current:
        original_spike_counts_for_window = current_spike_counts_for_window

        @wraps(original_spike_counts_for_window)
        def spike_counts_for_window_with_cell_id_validation(session: Any, encoding: Any, start: float, end: float) -> Any:
            _validate_position_decoding_cell_ids(session, encoding.config)
            cell_ids = _validated_encoding_cell_ids(encoding)
            return _spike_counts_for_window_by_cell_id(validation, session, encoding, start, end, cell_ids)

        setattr(spike_counts_for_window_with_cell_id_validation, _COUNTS_WRAPPER_FLAG, True)
        validation._spike_counts_for_window = spike_counts_for_window_with_cell_id_validation

    if not distance_is_current:
        original_distance = current_distance

        @wraps(original_distance)
        def distance_with_overflow_safe_norm(left: Any, right: Any) -> float:
            delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
            return float(np.hypot.reduce(delta.reshape(-1), initial=0.0))

        setattr(distance_with_overflow_safe_norm, _DISTANCE_WRAPPER_FLAG, True)
        validation._distance = distance_with_overflow_safe_norm

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
    if _contains_textual_or_complex_mask_values(raw):
        raise ValueError("train_frame_mask must contain boolean or numeric 0/1 values")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("train_frame_mask must contain boolean or numeric 0/1 values") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("train_frame_mask must contain finite boolean or numeric 0/1 values")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("train_frame_mask must contain boolean or numeric 0/1 values")
    return numeric.astype(bool)


def _contains_textual_or_complex_mask_values(values: Any) -> bool:
    raw = np.asarray(values)
    if raw.size == 0:
        return False
    if raw.dtype.kind in {"U", "S", "c"}:
        return True
    if raw.dtype == object:
        return any(isinstance(value, (str, bytes, complex, np.complexfloating)) for value in raw.reshape(-1))
    return False


def _validate_position_decoding_cell_ids(session: Any, encoding_config: Any) -> None:
    """Reject lossy cell IDs before position-decoding helpers cast to integers."""

    _validate_session_spike_cell_ids(session)
    if bool(getattr(encoding_config, "use_excitatory", True)):
        _validate_optional_integral_ids(getattr(session, "excitatory_neurons", np.empty(0)), "excitatory_neurons")
    _validate_optional_integral_ids(getattr(session, "inhibitory_neurons", np.empty(0)), "inhibitory_neurons")


def _validated_encoding_cell_ids(encoding: Any) -> np.ndarray:
    cell_ids = _coerce_integral_ids(getattr(encoding, "cell_ids"), "encoding.cell_ids")
    n_cells = int(getattr(encoding, "n_cells"))
    if cell_ids.shape != (n_cells,):
        raise ValueError("encoding.cell_ids must contain one ID per encoding row")
    if np.unique(cell_ids).shape[0] != cell_ids.shape[0]:
        raise ValueError("encoding.cell_ids must be unique")
    return cell_ids


def _spike_counts_for_window_by_cell_id(validation: Any, session: Any, encoding: Any, start: float, end: float, cell_ids: np.ndarray) -> np.ndarray:
    counts = np.zeros(int(getattr(encoding, "n_cells")), dtype=int)
    spikes, _ = validation._spikes_and_cell_ids_for_encoding(session, encoding.config)
    if not spikes.size or not counts.size:
        return counts
    if spikes.ndim != 2 or spikes.shape[1] < 2:
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")

    spike_times = np.asarray(spikes[:, 0], dtype=float)
    spike_cell_ids = _coerce_integral_ids(spikes[:, 1], "spike cell IDs")
    keep = (spike_times >= float(start)) & (spike_times < float(end)) & np.isin(spike_cell_ids, cell_ids)
    if not np.any(keep):
        return counts

    row_by_cell_id = {int(cell_id): row for row, cell_id in enumerate(cell_ids)}
    rows = np.fromiter(
        (row_by_cell_id[int(cell_id)] for cell_id in spike_cell_ids[keep]),
        dtype=int,
        count=int(np.sum(keep)),
    )
    np.add.at(counts, rows, 1)
    return counts


def _validate_session_spike_cell_ids(session: Any) -> None:
    spikes = np.asarray(getattr(session, "spikes", np.empty((0, 2))))
    if spikes.size == 0:
        return
    if spikes.ndim != 2 or spikes.shape[1] < 2:
        raise ValueError("spikes must be two-dimensional with at least time and cell-id columns")
    _coerce_integral_ids(spikes[:, 1], "spike cell IDs")


def _validate_optional_integral_ids(values: Any, name: str) -> None:
    arr = np.asarray(values)
    if arr.size:
        _coerce_integral_ids(arr.reshape(-1), name)


def _coerce_integral_ids(values: Any, name: str) -> np.ndarray:
    from .emission_cell_id_validation import _coerce_integral_ids as coerce

    return coerce(values, name)


def _positive_finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    value = _reject_array_shaped_scalar(name, value, f"{name} must be finite and positive")
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
    message = f"{name} must be an integer"
    value = _reject_array_shaped_scalar(name, value, message)
    try:
        item = np.asarray(value).item()
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(item, (int, np.integer)):
        return int(item)
    if isinstance(item, (str, np.str_, bytes, np.bytes_)):
        try:
            text = bytes(item).decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        except UnicodeDecodeError as exc:
            raise ValueError(message) from exc
        try:
            numeric = Decimal(text.strip())
        except InvalidOperation as exc:
            raise ValueError(message) from exc
        if not numeric.is_finite():
            raise ValueError(f"{name} must be a finite integer")
        integral = numeric.to_integral_value()
        if numeric != integral:
            raise ValueError(message)
        return int(integral)
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite integer")
    if not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _reject_array_shaped_scalar(name: str, value: Any, message: str) -> Any:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.shape != ():
        raise ValueError(message)
    return value


__all__ = ["apply_position_decoding_config_validation_patch"]
