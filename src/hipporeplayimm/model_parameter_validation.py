"""Validate replay-model numeric parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_model_parameter_validation_patch_applied"
_REPLAY_CALIBRATION_PATCHED_FLAG = "_replay_calibration_max_gain_validation_patch_applied"
_STATE_SPACE_DECAY_HELPERS_PATCHED_FLAG = "_state_space_velocity_decay_validation_patch_applied"
_STATE_SPACE_MODE_TRANSITION_PATCHED_FLAG = "_state_space_mode_transition_validation_patch_applied"
_DURATION_OCCUPANCY_DECAY_PATCHED_FLAG = "_duration_occupancy_velocity_decay_validation_patch_applied"
_SPARSE_MOMENTUM_DECAY_PATCHED_FLAG = "_sparse_momentum_velocity_decay_validation_patch_applied"
_DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG = "_displacement_momentum_velocity_decay_validation_patch_applied"
_TRAJECTORY_IMM_PARAMETERS_PATCHED_FLAG = "_trajectory_imm_parameter_validation_patch_applied"
_PYRECEST_IMM_DECAY_PATCHED_FLAG = "_pyrecest_imm_velocity_decay_validation_patch_applied"


def _is_boolean_scalar(value: object) -> bool:
    """Return True for Python, NumPy, and object-wrapped boolean scalars."""

    if isinstance(value, (bool, np.bool_)):
        return True
    arr = np.asarray(value)
    if arr.ndim != 0:
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        try:
            return isinstance(arr.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _reject_boolean_scalar(name: str, value: object) -> None:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not boolean")


def _validate_unit_interval_parameter(name: str, value: object) -> float:
    _reject_boolean_scalar(name, value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and lie in [0, 1]") from exc
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return numeric


def _validate_finite_nonnegative_parameter(name: str, value: object) -> float:
    _reject_boolean_scalar(name, value)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _validate_momentum_velocity_decay(value: object) -> float:
    return _validate_unit_interval_parameter("momentum_velocity_decay", value)


def _validate_momentum_velocity_decay_tau(value: object) -> float:
    return _validate_finite_nonnegative_parameter("momentum_velocity_decay_tau_s", value)


def _config_momentum_velocity_decay_tau(config: object) -> float:
    return _validate_momentum_velocity_decay_tau(getattr(config, "momentum_velocity_decay_tau_s", 0.0))


def _should_validate_config_momentum_velocity_decay(config: object) -> bool:
    return _config_momentum_velocity_decay_tau(config) == 0.0


def _validate_config_momentum_velocity_decay(config: object) -> None:
    if _should_validate_config_momentum_velocity_decay(config):
        _validate_momentum_velocity_decay(getattr(config, "momentum_velocity_decay", 0.95))


def _validate_config_or_scalar_momentum_velocity_decay(config_or_decay: object) -> None:
    if hasattr(config_or_decay, "momentum_velocity_decay"):
        _validate_config_momentum_velocity_decay(config_or_decay)
        return
    _validate_momentum_velocity_decay(config_or_decay)


def _validate_replay_calibration_max_gain(calibration: object | None) -> None:
    if calibration is None:
        return
    try:
        max_gain = float(getattr(calibration, "max_gain"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("max_gain must be finite and at least 1.0") from exc
    if not np.isfinite(max_gain) or max_gain < 1.0:
        raise ValueError("max_gain must be finite and at least 1.0")


def _apply_replay_calibration_max_gain_validation_patch() -> None:
    from . import result_improvement_extensions as extensions

    current = extensions.build_sorted_emissions_with_replay_calibration
    if getattr(current, _REPLAY_CALIBRATION_PATCHED_FLAG, False):
        return

    @wraps(current)
    def build_sorted_emissions_with_replay_calibration(session, encoding, ripple, config=None, calibration=None):
        _validate_replay_calibration_max_gain(calibration)
        return current(session, encoding, ripple, config, calibration)

    setattr(build_sorted_emissions_with_replay_calibration, _REPLAY_CALIBRATION_PATCHED_FLAG, True)
    setattr(build_sorted_emissions_with_replay_calibration, "__hipporeplayimm_original__", current)
    extensions.build_sorted_emissions_with_replay_calibration = build_sorted_emissions_with_replay_calibration


def _apply_state_space_mode_transition_validation_patch() -> None:
    import sys

    from . import state_space_utils

    current = state_space_utils._mode_transition_matrix
    if getattr(current, _STATE_SPACE_MODE_TRANSITION_PATCHED_FLAG, False):
        return

    @wraps(current)
    def mode_transition_matrix(n_modes, stickiness):
        _validate_unit_interval_parameter("mode_stickiness", stickiness)
        return current(n_modes, stickiness)

    setattr(mode_transition_matrix, _STATE_SPACE_MODE_TRANSITION_PATCHED_FLAG, True)
    setattr(mode_transition_matrix, "__hipporeplayimm_original__", current)
    state_space_utils._mode_transition_matrix = mode_transition_matrix
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_mode_transition_matrix", None) is current:
            module._mode_transition_matrix = mode_transition_matrix
    setattr(state_space_utils, _STATE_SPACE_MODE_TRANSITION_PATCHED_FLAG, True)


def _apply_state_space_velocity_decay_validation_patch() -> None:
    from . import state_space_model

    if getattr(state_space_model, _STATE_SPACE_DECAY_HELPERS_PATCHED_FLAG, False):
        return

    original_decays = state_space_model._momentum_velocity_decays
    original_multipliers = state_space_model._momentum_prediction_multipliers

    @wraps(original_decays)
    def momentum_velocity_decays(config, transition_durations):
        _validate_config_momentum_velocity_decay(config)
        return original_decays(config, transition_durations)

    @wraps(original_multipliers)
    def momentum_prediction_multipliers(config, transition_durations, *, fallback_dt):
        _validate_config_momentum_velocity_decay(config)
        return original_multipliers(config, transition_durations, fallback_dt=fallback_dt)

    state_space_model._momentum_velocity_decays = momentum_velocity_decays
    state_space_model._momentum_prediction_multipliers = momentum_prediction_multipliers
    setattr(state_space_model, _STATE_SPACE_DECAY_HELPERS_PATCHED_FLAG, True)


def _apply_duration_occupancy_velocity_decay_validation_patch() -> None:
    from . import duration_occupancy

    if getattr(duration_occupancy, _DURATION_OCCUPANCY_DECAY_PATCHED_FLAG, False):
        return

    original = duration_occupancy._duration_adjusted_decays

    @wraps(original)
    def duration_adjusted_decays(config_or_decay, durations, reference_dt):
        _validate_config_or_scalar_momentum_velocity_decay(config_or_decay)
        return original(config_or_decay, durations, reference_dt)

    duration_occupancy._duration_adjusted_decays = duration_adjusted_decays
    setattr(duration_occupancy, _DURATION_OCCUPANCY_DECAY_PATCHED_FLAG, True)


def _apply_sparse_momentum_velocity_decay_validation_patch() -> None:
    from . import state_space_sparse_momentum

    if getattr(state_space_sparse_momentum, _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG, False):
        return

    original = state_space_sparse_momentum._duration_adjusted_decays

    @wraps(original)
    def duration_adjusted_decays(config, durations, reference_dt):
        _validate_config_momentum_velocity_decay(config)
        return original(config, durations, reference_dt)

    state_space_sparse_momentum._duration_adjusted_decays = duration_adjusted_decays
    setattr(state_space_sparse_momentum, _SPARSE_MOMENTUM_DECAY_PATCHED_FLAG, True)


def _apply_displacement_momentum_velocity_decay_validation_patch() -> None:
    from . import state_space_displacement_momentum

    if getattr(state_space_displacement_momentum, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, False):
        return

    original = state_space_displacement_momentum._duration_adjusted_decays

    @wraps(original)
    def duration_adjusted_decays(config, durations, reference_dt):
        _validate_config_momentum_velocity_decay(config)
        return original(config, durations, reference_dt)

    state_space_displacement_momentum._duration_adjusted_decays = duration_adjusted_decays
    setattr(state_space_displacement_momentum, _DISPLACEMENT_MOMENTUM_DECAY_PATCHED_FLAG, True)


def _apply_trajectory_imm_parameter_validation_patch() -> None:
    from . import state_space_trajectory_imm

    if getattr(state_space_trajectory_imm, _TRAJECTORY_IMM_PARAMETERS_PATCHED_FLAG, False):
        return

    original_mode_stickiness = state_space_trajectory_imm._trajectory_imm_mode_stickiness
    original_mode_prior = state_space_trajectory_imm._trajectory_imm_mode_prior
    original_mode_transition_matrix = state_space_trajectory_imm._trajectory_imm_mode_transition_matrix
    original_mode_transition_matrices = state_space_trajectory_imm._trajectory_imm_mode_transition_matrices

    @wraps(original_mode_stickiness)
    def trajectory_imm_mode_stickiness(config):
        explicit_value = getattr(config, "trajectory_imm_mode_stickiness", None)
        if explicit_value is None:
            _validate_unit_interval_parameter("imm_mode_stickiness", getattr(config, "imm_mode_stickiness", 0.95))
        else:
            _validate_unit_interval_parameter("trajectory_imm_mode_stickiness", explicit_value)
        return original_mode_stickiness(config)

    @wraps(original_mode_prior)
    def trajectory_imm_mode_prior(config):
        value = getattr(config, "trajectory_imm_momentum_initial_probability", None)
        if value is not None:
            _validate_unit_interval_parameter("trajectory_imm_momentum_initial_probability", value)
        return original_mode_prior(config)

    @wraps(original_mode_transition_matrix)
    def trajectory_imm_mode_transition_matrix(config, stickiness):
        _validate_unit_interval_parameter("trajectory_imm_mode_stickiness", stickiness)
        momentum_switch = getattr(config, "trajectory_imm_momentum_switch_probability", None)
        if momentum_switch is not None:
            _validate_finite_nonnegative_parameter("trajectory_imm_momentum_switch_probability", momentum_switch)
        return original_mode_transition_matrix(config, stickiness)

    @wraps(original_mode_transition_matrices)
    def trajectory_imm_mode_transition_matrices(config, stickiness, durations):
        _validate_unit_interval_parameter("trajectory_imm_mode_stickiness", stickiness)
        _validate_finite_nonnegative_parameter("imm_switch_tau_s", getattr(config, "imm_switch_tau_s", 0.0))
        return original_mode_transition_matrices(config, stickiness, durations)

    state_space_trajectory_imm._trajectory_imm_mode_stickiness = trajectory_imm_mode_stickiness
    state_space_trajectory_imm._trajectory_imm_mode_prior = trajectory_imm_mode_prior
    state_space_trajectory_imm._trajectory_imm_mode_transition_matrix = trajectory_imm_mode_transition_matrix
    state_space_trajectory_imm._trajectory_imm_mode_transition_matrices = trajectory_imm_mode_transition_matrices
    setattr(state_space_trajectory_imm, _TRAJECTORY_IMM_PARAMETERS_PATCHED_FLAG, True)


def _apply_pyrecest_imm_velocity_decay_validation_patch() -> None:
    from . import pyrecest_models

    cls = pyrecest_models.PyRecEstGoalParticleIMMModel
    current = cls.__post_init__
    if getattr(current, _PYRECEST_IMM_DECAY_PATCHED_FLAG, False):
        return

    @wraps(current)
    def post_init(self):
        current(self)
        for name in (
            "stationary_velocity_decay",
            "diffusion_velocity_decay",
            "momentum_velocity_decay",
            "jump_velocity_decay",
        ):
            _validate_unit_interval_parameter(name, getattr(self, name))

    setattr(post_init, _PYRECEST_IMM_DECAY_PATCHED_FLAG, True)
    setattr(post_init, "__hipporeplayimm_original__", current)
    cls.__post_init__ = post_init


def apply_model_parameter_validation_patch() -> None:
    """Install strict numeric validation patches for replay-model parameters."""

    from . import models

    if not getattr(models, _PATCHED_FLAG, False):
        original_positive = models._validate_positive_parameter
        original_nonnegative = models._validate_nonnegative_parameter
        original_probability = models._validate_probability_parameter

        @wraps(original_positive)
        def validate_positive_parameter(name: str, value: float) -> None:
            _reject_boolean_scalar(name, value)
            return original_positive(name, value)

        @wraps(original_nonnegative)
        def validate_nonnegative_parameter(name: str, value: float) -> None:
            _reject_boolean_scalar(name, value)
            if name == "velocity_decay":
                _validate_unit_interval_parameter(name, value)
                return None
            return original_nonnegative(name, value)

        @wraps(original_probability)
        def validate_probability_parameter(name: str, value: float) -> None:
            _reject_boolean_scalar(name, value)
            return original_probability(name, value)

        models._validate_positive_parameter = validate_positive_parameter
        models._validate_nonnegative_parameter = validate_nonnegative_parameter
        models._validate_probability_parameter = validate_probability_parameter
        setattr(models, _PATCHED_FLAG, True)

    _apply_state_space_mode_transition_validation_patch()
    _apply_state_space_velocity_decay_validation_patch()
    _apply_duration_occupancy_velocity_decay_validation_patch()
    _apply_sparse_momentum_velocity_decay_validation_patch()
    _apply_displacement_momentum_velocity_decay_validation_patch()
    _apply_trajectory_imm_parameter_validation_patch()
    _apply_pyrecest_imm_velocity_decay_validation_patch()
    _apply_replay_calibration_max_gain_validation_patch()


__all__ = ["apply_model_parameter_validation_patch"]
