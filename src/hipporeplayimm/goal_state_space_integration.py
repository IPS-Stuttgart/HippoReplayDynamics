'''Benchmark integration for exact goal-conditioned state-space replay.'''

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from types import SimpleNamespace

from .goal_state_space import GoalStateSpaceReplayModel

DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S = 85.0
DEFAULT_GOAL_LATERAL_SIGMA_SCALE = 1.0
DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT = 0.0
DEFAULT_GOAL_DRIFT_SPEED_CM_S = 400.0
DEFAULT_GOAL_MAX_STEP_SIGMA = 4.0
DEFAULT_GOAL_RESET_PROBABILITY = 0.0
DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT = 0.0
DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY = 0.0
DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE = 'all'
DEFAULT_GOAL_TERMINAL_PRIOR_SIGMA_CM = 0.0
DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT = 1.0
DEFAULT_GOAL_INITIAL_PRIOR_SIGMA_CM = 0.0
DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT = 1.0
DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT = 0.5
DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT = 1.0
DEFAULT_GOAL_FORWARD_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT = 0.9
DEFAULT_GOAL_REVERSE_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT = 0.1
DEFAULT_GOAL_SWITCHING_COMPONENT_SWITCH_PROBABILITY = 0.03
GOAL_STATE_SPACE_MODEL_NAMES = frozenset(
    {
        'sorted-spike-state-space-goal',
        'sorted-spike-state-space-goal-bidirectional',
        'sorted-spike-state-space-goal-forward-biased',
        'sorted-spike-state-space-goal-forward-biased-switching',
        'sorted-spike-state-space-goal-reverse-biased',
        'state-space-goal',
        'state-space-goal-bidirectional',
        'state-space-goal-forward-biased',
        'state-space-goal-forward-biased-switching',
        'state-space-goal-reverse-biased',
    }
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
        goal_state_space_lateral_sigma_scale: float = DEFAULT_GOAL_LATERAL_SIGMA_SCALE
        goal_state_space_diffusion_mixture_weight: float = DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT
        goal_state_space_drift_speed_cm_s: float = DEFAULT_GOAL_DRIFT_SPEED_CM_S
        goal_state_space_max_step_sigma: float = DEFAULT_GOAL_MAX_STEP_SIGMA
        goal_state_space_reset_probability: float = DEFAULT_GOAL_RESET_PROBABILITY
        goal_state_space_reset_initial_position_prior_weight: float = DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT
        goal_state_space_component_switch_probability: float = DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY
        goal_state_space_initial_position_prior_direction_mode: str = DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE
        goal_state_space_terminal_prior_sigma_cm: float = DEFAULT_GOAL_TERMINAL_PRIOR_SIGMA_CM
        goal_state_space_terminal_goal_prior_weight: float = DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT
        goal_state_space_initial_goal_prior_sigma_cm: float = DEFAULT_GOAL_INITIAL_PRIOR_SIGMA_CM
        goal_state_space_initial_goal_prior_weight: float = DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT
        goal_state_space_toward_direction_prior_weight: float = DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT
        goal_state_space_reverse_terminal_position_prior_weight: float = DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT

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


def goal_state_space_metadata_for_config(config: object) -> dict[str, object]:
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
        'goal_state_space_lateral_sigma_scale': float(
            _cfg(
                config,
                'goal_state_space_lateral_sigma_scale',
                DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
            )
        ),
        'goal_state_space_diffusion_mixture_weight': float(
            _cfg(
                config,
                'goal_state_space_diffusion_mixture_weight',
                DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
            )
        ),
        'goal_state_space_max_step_sigma': float(
            _cfg(
                config,
                'goal_state_space_max_step_sigma',
                DEFAULT_GOAL_MAX_STEP_SIGMA,
            )
        ),
        'goal_state_space_reset_probability': float(
            _cfg(
                config,
                'goal_state_space_reset_probability',
                DEFAULT_GOAL_RESET_PROBABILITY,
            )
        ),
        'goal_state_space_reset_initial_position_prior_weight': float(
            _cfg(
                config,
                'goal_state_space_reset_initial_position_prior_weight',
                DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
            )
        ),
        'goal_state_space_component_switch_probability': float(
            _cfg(
                config,
                'goal_state_space_component_switch_probability',
                DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
            )
        ),
        'goal_state_space_initial_position_prior_direction_mode': str(
            _cfg(
                config,
                'goal_state_space_initial_position_prior_direction_mode',
                DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
            )
        ),
        'goal_state_space_terminal_prior_sigma_cm': float(
            _cfg(
                config,
                'goal_state_space_terminal_prior_sigma_cm',
                DEFAULT_GOAL_TERMINAL_PRIOR_SIGMA_CM,
            )
        ),
        'goal_state_space_terminal_goal_prior_weight': float(
            _cfg(
                config,
                'goal_state_space_terminal_goal_prior_weight',
                DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
            )
        ),
        'goal_state_space_initial_goal_prior_sigma_cm': float(
            _cfg(
                config,
                'goal_state_space_initial_goal_prior_sigma_cm',
                DEFAULT_GOAL_INITIAL_PRIOR_SIGMA_CM,
            )
        ),
        'goal_state_space_initial_goal_prior_weight': float(
            _cfg(
                config,
                'goal_state_space_initial_goal_prior_weight',
                DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
            )
        ),
        'goal_state_space_toward_direction_prior_weight': float(
            _cfg(
                config,
                'goal_state_space_toward_direction_prior_weight',
                DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
            )
        ),
        'goal_state_space_reverse_terminal_position_prior_weight': float(
            _cfg(
                config,
                'goal_state_space_reverse_terminal_position_prior_weight',
                DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
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
        lateral_sigma_scale=float(
            _cfg(
                config,
                'goal_state_space_lateral_sigma_scale',
                DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
            )
        ),
        diffusion_mixture_weight=float(
            _cfg(
                config,
                'goal_state_space_diffusion_mixture_weight',
                DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
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
        reset_probability=float(
            _cfg(
                config,
                'goal_state_space_reset_probability',
                DEFAULT_GOAL_RESET_PROBABILITY,
            )
        ),
        reset_initial_position_prior_weight=float(
            _cfg(
                config,
                'goal_state_space_reset_initial_position_prior_weight',
                DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
            )
        ),
        component_switch_probability=_component_switch_probability_for_goal_model_name(
            name,
            config,
        ),
        initial_position_prior_direction_mode=str(
            _cfg(
                config,
                'goal_state_space_initial_position_prior_direction_mode',
                DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
            )
        ),
        terminal_goal_prior_sigma_cm=float(
            _cfg(
                config,
                'goal_state_space_terminal_prior_sigma_cm',
                DEFAULT_GOAL_TERMINAL_PRIOR_SIGMA_CM,
            )
        ),
        terminal_goal_prior_weight=float(
            _cfg(
                config,
                'goal_state_space_terminal_goal_prior_weight',
                DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
            )
        ),
        initial_goal_prior_sigma_cm=float(
            _cfg(
                config,
                'goal_state_space_initial_goal_prior_sigma_cm',
                DEFAULT_GOAL_INITIAL_PRIOR_SIGMA_CM,
            )
        ),
        initial_goal_prior_weight=float(
            _cfg(
                config,
                'goal_state_space_initial_goal_prior_weight',
                DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
            )
        ),
        toward_direction_prior_weight=_toward_direction_prior_weight_for_goal_model_name(
            name,
            config,
        ),
        reverse_terminal_position_prior_weight=float(
            _cfg(
                config,
                'goal_state_space_reverse_terminal_position_prior_weight',
                DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
            )
        ),
        direction_mode=_direction_mode_for_goal_model_name(name),
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


def _direction_mode_for_goal_model_name(name: str) -> str:
    if (
        name.endswith('-goal-bidirectional')
        or name.endswith('-goal-forward-biased')
        or name.endswith('-goal-forward-biased-switching')
        or name.endswith('-goal-reverse-biased')
    ):
        return 'bidirectional'
    return 'toward'


def _toward_direction_prior_weight_for_goal_model_name(name: str, config: object) -> float:
    if name.endswith('-goal-forward-biased-switching'):
        return DEFAULT_GOAL_FORWARD_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT
    if name.endswith('-goal-forward-biased'):
        return DEFAULT_GOAL_FORWARD_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT
    if name.endswith('-goal-reverse-biased'):
        return DEFAULT_GOAL_REVERSE_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT
    return float(
        _cfg(
            config,
            'goal_state_space_toward_direction_prior_weight',
            DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
        )
    )


def _component_switch_probability_for_goal_model_name(name: str, config: object) -> float:
    if name.endswith('-goal-forward-biased-switching'):
        return DEFAULT_GOAL_SWITCHING_COMPONENT_SWITCH_PROBABILITY
    return float(
        _cfg(
            config,
            'goal_state_space_component_switch_probability',
            DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
        )
    )


def _cfg(config: object, name: str, default):
    return getattr(config, name, default)
