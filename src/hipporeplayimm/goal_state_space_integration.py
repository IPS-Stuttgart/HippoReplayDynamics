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
_ORIGINALS_ATTR = '_hipporeplayimm_goal_state_space_parameter_validation_originals'


def apply_goal_state_space_parameter_validation_patch() -> None:
    '''Reject bool and array-like goal-state-space numeric parameters.'''

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
        return array[:, None]
    return value


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
