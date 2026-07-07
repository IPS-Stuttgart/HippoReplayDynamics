"""Validate topology-aware accuracy-grid replay parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_accuracy_grid_parameter_validation_patch_applied"
_TRANSITION_PATCHED_FLAG = "_accuracy_grid_transition_parameter_validation_patch_applied"
_MODEL_INIT_PATCHED_FLAG = "_valid_state_grid_model_parameter_validation_patch_applied"


def apply_accuracy_grid_parameter_validation_patch() -> None:
    """Install strict parameter validation for valid-grid replay helpers."""

    from . import accuracy_upgrades

    if getattr(accuracy_upgrades, _PATCHED_FLAG, False):
        return

    _patch_valid_grid_graph_transition(accuracy_upgrades)
    _patch_valid_state_grid_model_init(accuracy_upgrades)
    setattr(accuracy_upgrades, _PATCHED_FLAG, True)


def _patch_valid_grid_graph_transition(accuracy_upgrades) -> None:
    current = accuracy_upgrades.valid_grid_graph_transition
    if getattr(current, _TRANSITION_PATCHED_FLAG, False):
        return

    @wraps(current)
    def valid_grid_graph_transition(
        grid_shape,
        valid_mask,
        *,
        diagonal_neighbors: bool = True,
        stay_probability: float = 0.0,
    ):
        return current(
            _coerce_grid_shape(grid_shape),
            valid_mask,
            diagonal_neighbors=_coerce_boolean_parameter(
                "diagonal_neighbors",
                diagonal_neighbors,
            ),
            stay_probability=_coerce_stay_probability(stay_probability),
        )

    setattr(valid_grid_graph_transition, _TRANSITION_PATCHED_FLAG, True)
    setattr(valid_grid_graph_transition, "__hipporeplayimm_original__", current)
    accuracy_upgrades.valid_grid_graph_transition = valid_grid_graph_transition


def _patch_valid_state_grid_model_init(accuracy_upgrades) -> None:
    cls = accuracy_upgrades.ValidStateGridReplayModel
    current = cls.__init__
    if getattr(current, _MODEL_INIT_PATCHED_FLAG, False):
        return

    @wraps(current)
    def __init__(self, *args, **kwargs):
        current(self, *args, **kwargs)
        self.grid_shape = _coerce_grid_shape(self.grid_shape)
        self.diagonal_neighbors = _coerce_boolean_parameter(
            "diagonal_neighbors",
            self.diagonal_neighbors,
        )
        self.stay_probability = _coerce_stay_probability(self.stay_probability)

    setattr(__init__, _MODEL_INIT_PATCHED_FLAG, True)
    setattr(__init__, "__hipporeplayimm_original__", current)
    cls.__init__ = __init__


def _coerce_grid_shape(grid_shape) -> tuple[int, int]:
    try:
        dims = tuple(grid_shape)
    except TypeError as exc:
        raise ValueError("grid_shape must contain exactly two positive integer dimensions") from exc
    if len(dims) != 2:
        raise ValueError("grid_shape must contain exactly two positive integer dimensions")
    return (
        _coerce_positive_integer_dimension(dims[0]),
        _coerce_positive_integer_dimension(dims[1]),
    )


def _coerce_positive_integer_dimension(value) -> int:
    item = _scalar_item("grid_shape", value)
    if isinstance(item, (bool, np.bool_, str, bytes, np.str_, np.bytes_)):
        raise ValueError("grid_shape must contain positive integer dimensions")
    try:
        numeric = float(item)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_shape must contain positive integer dimensions") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError("grid_shape must contain positive integer dimensions")
    return int(numeric)


def _coerce_boolean_parameter(name: str, value) -> bool:
    item = _scalar_item(name, value)
    if not isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(item)


def _coerce_stay_probability(value) -> float:
    item = _scalar_item("stay_probability", value)
    if isinstance(item, (bool, np.bool_)):
        raise TypeError("stay_probability must be numeric, not boolean")
    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
        raise ValueError("stay_probability must be numeric, not text")
    try:
        numeric = float(item)
    except (TypeError, ValueError) as exc:
        raise ValueError("stay_probability must lie in [0, 1)") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise ValueError("stay_probability must lie in [0, 1)")
    return numeric


def _scalar_item(name: str, value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    try:
        return array.item()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a scalar") from exc


__all__ = ["apply_accuracy_grid_parameter_validation_patch"]
