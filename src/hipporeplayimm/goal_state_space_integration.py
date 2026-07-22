'''Benchmark integration for exact goal-conditioned state-space replay.'''

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from functools import wraps
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import goal_state_space as _goal_state_space
from .goal_state_space import GoalStateSpaceReplayModel

DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S = 85.0
DEFAULT_GOAL_DRIFT_SPEED_CM_S = 400.0
DEFAULT_GOAL_MAX_STEP_SIGMA = 4.0
GOAL_STATE_SPACE_MODEL_NAMES = frozenset(
    {'sorted-spike-state-space-goal', 'state-space-goal'}
)
GOAL_EVIDENCE_DIAGNOSTIC_COLUMN = 'diagnostic_goal_state_space_evidence_support'
_PARAMETER_VALIDATION_PATCH_ATTR = '_hipporeplayimm_goal_state_space_parameter_validation_patch'
_FARTHEST_POINT_PATCH_ATTR = '_hipporeplayimm_goal_state_space_farthest_point_patch'
_SMALL_DRIFT_PATCH_ATTR = '_hipporeplayimm_goal_state_space_small_drift_patch'
_ORIGINALS_ATTR = '_hipporeplayimm_goal_state_space_parameter_validation_originals'


def apply_goal_state_space_farthest_point_patch() -> None:
    '''Select default goals without overflowing finite coordinate distances.'''

    current = _goal_state_space._farthest_point_subset
    if getattr(current, _FARTHEST_POINT_PATCH_ATTR, False):
        return

    @wraps(current)
    def farthest_point_subset(points, max_points):
        values = _unique_rows_preserve_order(np.asarray(points, dtype=float))
        if values.shape[0] == 0:
            return current(points, max_points)

        if values.shape[0] <= max_points:
            return values.copy()

        coordinate_scale = float(np.max(np.abs(values)))
        if coordinate_scale > 0.0:
            anchor_scores = np.sum(values / coordinate_scale, axis=1)
        else:
            anchor_scores = np.zeros(values.shape[0], dtype=float)

        selected = [int(np.argmin(anchor_scores))]
        min_log_distances = np.full(values.shape[0], np.inf, dtype=float)
        for _ in range(1, max_points):
            _, log_distances = _goal_state_space._scaled_euclidean_distances(
                values,
                values[selected[-1]],
                1.0,
            )
            min_log_distances = np.minimum(min_log_distances, log_distances)
            min_log_distances[np.asarray(selected, dtype=int)] = -np.inf
            selected.append(int(np.argmax(min_log_distances)))
        return values[np.asarray(selected, dtype=int)]

    setattr(farthest_point_subset, _FARTHEST_POINT_PATCH_ATTR, True)
    _goal_state_space._farthest_point_subset = farthest_point_subset


def apply_goal_state_space_small_drift_patch() -> None:
    '''Preserve representable sub-epsilon goal-directed motion.'''

    current = _goal_state_space._goal_drift_prediction
    if getattr(current, _SMALL_DRIFT_PATCH_ATTR, False):
        return

    @wraps(current)
    def goal_drift_prediction(position, goal, drift_step_cm):
        current_position = np.asarray(position, dtype=float)
        target = np.asarray(goal, dtype=float)
        coordinate_scale = float(
            max(np.max(np.abs(current_position)), np.max(np.abs(target)))
        )
        step = float(drift_step_cm)
        if step <= 0.0 or coordinate_scale == 0.0:
            return current(position, goal, drift_step_cm)

        scaled_vector = (
            target / coordinate_scale - current_position / coordinate_scale
        )
        scaled_distance = float(np.hypot.reduce(np.abs(scaled_vector)))
        if scaled_distance == 0.0:
            return current(position, goal, drift_step_cm)

        with np.errstate(over='ignore', invalid='ignore'):
            distance = coordinate_scale * scaled_distance
        if not np.isfinite(distance) or distance > np.finfo(float).eps:
            return current(position, goal, drift_step_cm)

        if step >= distance:
            return target.copy()
        predicted = current_position + step * (scaled_vector / scaled_distance)
        if not np.all(np.isfinite(predicted)):
            raise ValueError('goal drift prediction exceeds floating-point range')
        return predicted

    setattr(goal_drift_prediction, _SMALL_DRIFT_PATCH_ATTR, True)
    setattr(goal_drift_prediction, '__hipporeplayimm_original__', current)
    _goal_state_space._goal_drift_prediction = goal_drift_prediction


def apply_goal_state_space_parameter_validation_patch() -> None:
    '''Reject bool and array-like goal-state-space numeric parameters.'''

    apply_goal_state_space_farthest_point_patch()
    apply_goal_state_space_small_drift_patch()

    originals = getattr(_goal_state_space, _ORIGINALS_ATTR, None)
    if originals is None:
        originals = {
            'score': GoalStateSpaceReplayModel.score,
            'goal_transition_matrix': _goal_state_space._goal_transition_matrix,
        }
        setattr(_goal_state_space, _ORIGINALS_ATTR, originals)

    if (
        getattr(GoalStateSpaceReplayModel.score, _PARAMETER_VALIDATION_PATCH_ATTR, False)
        and getattr(_goal_state_space._goal_transition_matrix, _PARAMETER_VALIDATION_PATCH_ATTR, False)
    ):
        return

    original_score = originals['score']
    original_goal_transition_matrix = originals['goal_transition_matrix']

    @wraps(original_score)
    def score(self, emissions, bin_centers):
        _positive_parameter('transition_sigma_cm_sqrt_s', self.transition_sigma_cm_sqrt_s)
        _nonnegative_parameter('drift_speed_cm_s', self.drift_speed_cm_s)
        _positive_parameter('max_step_sigma', self.max_step_sigma)
        centers = _coerce_position_matrix(bin_centers)
        candidate_goals = _coerce_candidate_goals(self.candidate_goals, centers)
        if candidate_goals is self.candidate_goals:
            return original_score(self, emissions, centers)
        old_candidate_goals = self.candidate_goals
        try:
            self.candidate_goals = candidate_goals
            return original_score(self, emissions, centers)
        finally:
            self.candidate_goals = old_candidate_goals

    @wraps(original_goal_transition_matrix)
    def _goal_transition_matrix(bin_centers, goal, *, drift_step_cm, sigma_cm, max_step_sigma):
        _positive_parameter('sigma_cm', sigma_cm)
        _nonnegative_parameter('drift_step_cm', drift_step_cm)
        _positive_parameter('max_step_sigma', max_step_sigma)
        centers = _coerce_position_matrix(bin_centers)
        goal_vector = _coerce_goal_vector(goal, centers.shape[1] if centers.ndim == 2 else None)
        return original_goal_transition_matrix(
            centers,
            goal_vector,
            drift_step_cm=drift_step_cm,
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )

    setattr(score, _PARAMETER_VALIDATION_PATCH_ATTR, True)
    setattr(_goal_transition_matrix, _PARAMETER_VALIDATION_PATCH_ATTR, True)
    GoalStateSpaceReplayModel.score = score
    _goal_state_space.GoalStateSpaceReplayModel.score = score
    _goal_state_space._goal_transition_matrix = _goal_transition_matrix


def apply_goal_state_space_patch() -> None:
    '''Register the exact goal-state-space model with benchmark entry points.'''

    from . import benchmarks as bench
    from . import evidence_reporting as evidence
    from . import ground_truth as gt

    if getattr(bench, '_goal_state_space_patch_applied', False):
        return

    base_benchmark_config = bench.BenchmarkConfig
    base_build_models = bench._build_models
    base_metadata = bench._benchmark_config_metadata

    @dataclass(frozen=True)
    class BenchmarkConfig(base_benchmark_config):
        goal_state_space_transition_sigma_cm_sqrt_s: float = DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S
        goal_state_space_drift_speed_cm_s: float = DEFAULT_GOAL_DRIFT_SPEED_CM_S
        goal_state_space_max_step_sigma: float = DEFAULT_GOAL_MAX_STEP_SIGMA

    def build_models(config: object, session=None) -> dict[str, object]:
        model_names = tuple(str(name) for name in getattr(config, 'models'))
        output: dict[str, object] = {}
        non_goal_models = tuple(name for name in model_names if not _is_goal_model_name(name))
        if non_goal_models:
            output.update(
                base_build_models(
                    _copy_config_with_models(config, non_goal_models),
                    session=session,
                )
            )
        goal_candidates = bench._session_goal_candidates(session) if session is not None else None
        for name in model_names:
            if _is_goal_model_name(name):
                output[name] = _goal_state_space_model(
                    config,
                    goal_candidates=goal_candidates,
                    name=name,
                )
        return {name: output[name] for name in model_names}

    def benchmark_config_metadata(config: object) -> dict[str, object]:
        metadata = dict(base_metadata(config))
        metadata.update(goal_state_space_metadata_for_config(config))
        return metadata

    if GOAL_EVIDENCE_DIAGNOSTIC_COLUMN not in evidence.EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS:
        evidence.EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS = (
            *evidence.EVIDENCE_SUPPORT_DIAGNOSTIC_COLUMNS,
            GOAL_EVIDENCE_DIAGNOSTIC_COLUMN,
        )

    bench.BenchmarkConfig = BenchmarkConfig
    gt.BenchmarkConfig = BenchmarkConfig
    bench._build_models = build_models
    gt._build_models = build_models
    bench._benchmark_config_metadata = benchmark_config_metadata
    bench._goal_state_space_patch_applied = True


def goal_state_space_metadata_for_config(config: object) -> dict[str, float]:
    return {
        'goal_state_space_transition_sigma_cm_sqrt_s': _positive_goal_parameter(
            config,
            'goal_state_space_transition_sigma_cm_sqrt_s',
            DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
        ),
        'goal_state_space_drift_speed_cm_s': _nonnegative_goal_parameter(
            config,
            'goal_state_space_drift_speed_cm_s',
            DEFAULT_GOAL_DRIFT_SPEED_CM_S,
        ),
        'goal_state_space_max_step_sigma': _positive_goal_parameter(
            config,
            'goal_state_space_max_step_sigma',
            DEFAULT_GOAL_MAX_STEP_SIGMA,
        ),
    }


def _goal_state_space_model(
    config: object,
    *,
    goal_candidates,
    name: str,
) -> GoalStateSpaceReplayModel:
    return GoalStateSpaceReplayModel(
        candidate_goals=goal_candidates,
        transition_sigma_cm_sqrt_s=_positive_goal_parameter(
            config,
            'goal_state_space_transition_sigma_cm_sqrt_s',
            DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
        ),
        drift_speed_cm_s=_nonnegative_goal_parameter(
            config,
            'goal_state_space_drift_speed_cm_s',
            DEFAULT_GOAL_DRIFT_SPEED_CM_S,
        ),
        max_step_sigma=_positive_goal_parameter(
            config,
            'goal_state_space_max_step_sigma',
            DEFAULT_GOAL_MAX_STEP_SIGMA,
        ),
        name=name,
    )


def _copy_config_with_models(config: object, models: tuple[str, ...]) -> object:
    if is_dataclass(config):
        return replace(config, models=models)
    data = dict(vars(config))
    data['models'] = models
    return SimpleNamespace(**data)


def _is_goal_model_name(name: str) -> bool:
    return name in GOAL_STATE_SPACE_MODEL_NAMES


def _positive_goal_parameter(config: object, name: str, default: float) -> float:
    return _positive_parameter(name, _cfg(config, name, default))


def _nonnegative_goal_parameter(config: object, name: str, default: float) -> float:
    return _nonnegative_parameter(name, _cfg(config, name, default))


def _positive_parameter(name: str, value: Any) -> float:
    return _numeric_parameter(name, value, positive=True)


def _nonnegative_parameter(name: str, value: Any) -> float:
    return _numeric_parameter(name, value, positive=False)


def _numeric_parameter(name: str, value: Any, *, positive: bool) -> float:
    if _contains_bool(value):
        raise ValueError(f'{name} must be a scalar numeric value, not boolean')
    array = _as_array(value)
    if array.shape != ():
        raise ValueError(f'{name} must be a scalar numeric value')
    try:
        numeric = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a scalar numeric value') from exc
    if positive:
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
    elif not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return numeric


def _coerce_position_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    return array


def _coerce_candidate_goals(value: Any, centers: np.ndarray) -> Any:
    if value is None:
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and centers.ndim == 2 and centers.shape[1] == 1:
        array = array[:, None]
    if array.ndim != 2:
        return array
    return _unique_rows_preserve_order(array)


def _unique_rows_preserve_order(values: np.ndarray) -> np.ndarray:
    '''Drop exact duplicate coordinate rows without reordering unique goals.'''

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] <= 1:
        return array
    _, first_indices = np.unique(array, axis=0, return_index=True)
    if first_indices.shape[0] == array.shape[0]:
        return array
    return array[np.sort(first_indices)]


def _coerce_goal_vector(value: Any, position_dim: int | None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 and position_dim == 1:
        return array.reshape(1)
    return array


def _contains_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    array = _as_array(value)
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in array.flat)
    return False


def _as_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value)
    except ValueError:
        return np.asarray(value, dtype=object)


def _cfg(config: object, name: str, default):
    return getattr(config, name, default)


apply_goal_state_space_parameter_validation_patch()
