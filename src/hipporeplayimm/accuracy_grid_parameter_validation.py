"""Validate topology-aware accuracy-grid and replay-gain parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_accuracy_grid_parameter_validation_patch_applied"
_TRANSITION_PATCHED_FLAG = "_accuracy_grid_transition_parameter_validation_patch_applied"
_MODEL_INIT_PATCHED_FLAG = "_valid_state_grid_model_parameter_validation_patch_applied"
_MODEL_SCORE_PATCHED_FLAG = "_valid_state_grid_model_score_parameter_validation_patch_applied"
_REPLAY_GAIN_CONFIG_PATCHED_FLAG = "_replay_gain_config_parameter_validation_patch_applied"


def apply_accuracy_grid_parameter_validation_patch() -> None:
    """Install strict parameter validation for accuracy-upgrade helpers.

    The public runtime patch hook may be called after tests or interactive
    notebooks have restored helper functions/classes to their original objects.
    Do not trust the module-level sentinel alone: each helper wrapper is
    idempotent, so re-check the actual callable state on every invocation and
    re-wrap any stale replacement.
    """

    from . import accuracy_upgrades

    _patch_valid_grid_graph_transition(accuracy_upgrades)
    _patch_valid_state_grid_model_init(accuracy_upgrades)
    _patch_valid_state_grid_model_score(accuracy_upgrades)
    _patch_replay_gain_config_init(accuracy_upgrades)
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


def _patch_valid_state_grid_model_score(accuracy_upgrades) -> None:
    cls = accuracy_upgrades.ValidStateGridReplayModel
    current = cls.score
    if getattr(current, _MODEL_SCORE_PATCHED_FLAG, False):
        return

    @wraps(current)
    def score(self, emissions, bin_centers):
        self.grid_shape = _coerce_grid_shape(self.grid_shape)
        self.diagonal_neighbors = _coerce_boolean_parameter(
            "diagonal_neighbors",
            self.diagonal_neighbors,
        )
        self.stay_probability = _coerce_stay_probability(self.stay_probability)
        return current(self, emissions, bin_centers)

    setattr(score, _MODEL_SCORE_PATCHED_FLAG, True)
    setattr(score, "__hipporeplayimm_original__", current)
    cls.score = score


def _patch_replay_gain_config_init(accuracy_upgrades) -> None:
    """Reject invalid replay-gain priors and bounds at config construction."""

    cls = accuracy_upgrades.ReplayGainConfig
    current = cls.__init__
    if getattr(current, _REPLAY_GAIN_CONFIG_PATCHED_FLAG, False):
        return

    @wraps(current)
    def __init__(self, *args, **kwargs):
        current(self, *args, **kwargs)
        prior_observed_spikes = _coerce_nonnegative_real(
            "prior_observed_spikes",
            self.prior_observed_spikes,
        )
        prior_expected_spikes = _coerce_positive_real(
            "prior_expected_spikes",
            self.prior_expected_spikes,
        )
        min_gain = _coerce_positive_real("min_gain", self.min_gain)
        max_gain = _coerce_positive_real("max_gain", self.max_gain)
        if min_gain > max_gain:
            raise ValueError("min_gain must be less than or equal to max_gain")

        # ReplayGainConfig is frozen; canonicalize accepted NumPy scalar wrappers
        # with object.__setattr__ after the generated dataclass initializer runs.
        object.__setattr__(self, "prior_observed_spikes", prior_observed_spikes)
        object.__setattr__(self, "prior_expected_spikes", prior_expected_spikes)
        object.__setattr__(self, "min_gain", min_gain)
        object.__setattr__(self, "max_gain", max_gain)

    setattr(__init__, _REPLAY_GAIN_CONFIG_PATCHED_FLAG, True)
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
    except (TypeError, ValueError, OverflowError) as exc:
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
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("stay_probability must lie in [0, 1)") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise ValueError("stay_probability must lie in [0, 1)")
    return numeric


def _coerce_finite_real(name: str, value) -> float:
    item = _scalar_item(name, value)
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean")
    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be numeric, not text")
    if isinstance(item, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be real, not complex")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _coerce_nonnegative_real(name: str, value) -> float:
    numeric = _coerce_finite_real(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _coerce_positive_real(name: str, value) -> float:
    numeric = _coerce_finite_real(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
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
