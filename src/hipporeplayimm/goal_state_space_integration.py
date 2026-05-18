'''Benchmark integration for exact goal-conditioned state-space replay.'''

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace

from .goal_state_space import GoalStateSpaceReplayModel

DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S = 85.0
DEFAULT_GOAL_DRIFT_SPEED_CM_S = 400.0
DEFAULT_GOAL_MAX_STEP_SIGMA = 4.0
GOAL_STATE_SPACE_MODEL_NAMES = frozenset(
    {'sorted-spike-state-space-goal', 'state-space-goal'}
)
GOAL_EVIDENCE_DIAGNOSTIC_COLUMN = 'diagnostic_goal_state_space_evidence_support'


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
        'goal_state_space_transition_sigma_cm_sqrt_s': float(
            _cfg(
                config,
                'goal_state_space_transition_sigma_cm_sqrt_s',
                DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
            )
        ),
        'goal_state_space_drift_speed_cm_s': float(
            _cfg(
                config,
                'goal_state_space_drift_speed_cm_s',
                DEFAULT_GOAL_DRIFT_SPEED_CM_S,
            )
        ),
        'goal_state_space_max_step_sigma': float(
            _cfg(
                config,
                'goal_state_space_max_step_sigma',
                DEFAULT_GOAL_MAX_STEP_SIGMA,
            )
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
        transition_sigma_cm_sqrt_s=float(
            _cfg(
                config,
                'goal_state_space_transition_sigma_cm_sqrt_s',
                DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
            )
        ),
        drift_speed_cm_s=float(
            _cfg(
                config,
                'goal_state_space_drift_speed_cm_s',
                DEFAULT_GOAL_DRIFT_SPEED_CM_S,
            )
        ),
        max_step_sigma=float(
            _cfg(
                config,
                'goal_state_space_max_step_sigma',
                DEFAULT_GOAL_MAX_STEP_SIGMA,
            )
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


def _cfg(config: object, name: str, default):
    return getattr(config, name, default)
