import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_model_evidence.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_model_evidence", _SCRIPT)
assert _SPEC is not None
benchmark_model_evidence = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(benchmark_model_evidence)

_events = benchmark_model_evidence._events
_clusterless_mark_config = benchmark_model_evidence._clusterless_mark_config
_add_evidence_columns = benchmark_model_evidence._add_evidence_columns
_family = benchmark_model_evidence._family
_goal_prior_weights_for_event = benchmark_model_evidence._goal_prior_weights_for_event
_initial_position_prior_weights_for_event = benchmark_model_evidence._initial_position_prior_weights_for_event
_reverse_terminal_position_prior_weights_for_event = (
    benchmark_model_evidence._reverse_terminal_position_prior_weights_for_event
)
_model_for_event = benchmark_model_evidence._model_for_event
_models = benchmark_model_evidence._models


class _SessionStub:
    ripple_count = 10

    @staticmethod
    def ripple_indices_in_run():
        return np.array([2, 4, 6, 8], dtype=int)


class _GoalPriorSessionStub:
    well_sequence = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    position = np.array(
        [
            [4.1, 0.0, 0.0],
            [4.5, 0.0, 0.0],
            [4.9, 0.0, 0.0],
            [6.0, 1.0, 0.0],
            [9.1, 1.0, 0.0],
            [9.5, 1.0, 0.0],
            [9.9, 1.0, 0.0],
        ]
    )

    @staticmethod
    def ripple(index):
        return SimpleNamespace(peak=6.0)


def test_model_evidence_accepts_sorted_spike_state_space_models():
    args = argparse.Namespace(
        models="sorted-spike-state-space-diffusion sorted-spike-state-space-momentum sorted-spike-state-space-imm",
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_candidate_top_k=17,
        goal_state_space_transition_sigma_cm_sqrt_s=85.0,
        goal_state_space_drift_speed_cm_s=400.0,
        goal_state_space_max_step_sigma=4.0,
    )

    models = _models(args)

    assert list(models) == [
        "sorted-spike-state-space-diffusion",
        "sorted-spike-state-space-momentum",
        "sorted-spike-state-space-imm",
    ]
    assert models["sorted-spike-state-space-diffusion"].name == "sorted-spike-state-space-diffusion"
    assert models["sorted-spike-state-space-momentum"].name == "sorted-spike-state-space-momentum"
    assert models["sorted-spike-state-space-imm"].name == "sorted-spike-state-space-imm"
    assert models["sorted-spike-state-space-diffusion"].config.diffusion_sigma_cm_sqrt_s == 42.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_sigma_cm_sqrt_s == 43.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_initial_sigma_cm_sqrt_s == 44.0
    assert models["sorted-spike-state-space-momentum"].config.momentum_velocity_decay == 0.8
    assert models["sorted-spike-state-space-momentum"].config.momentum_candidate_top_k == 17
    assert models["sorted-spike-state-space-imm"].config.imm_mode_stickiness == 0.91


def test_model_evidence_accepts_clusterless_state_space_models():
    args = argparse.Namespace(
        models="clusterless-state-space-diffusion clusterless-state-space-momentum clusterless-state-space-imm",
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_candidate_top_k=17,
        goal_state_space_transition_sigma_cm_sqrt_s=85.0,
        goal_state_space_drift_speed_cm_s=400.0,
        goal_state_space_max_step_sigma=4.0,
    )

    models = _models(args)

    assert list(models) == [
        "clusterless-state-space-diffusion",
        "clusterless-state-space-momentum",
        "clusterless-state-space-imm",
    ]
    assert models["clusterless-state-space-diffusion"].name == "clusterless-state-space-diffusion"
    assert models["clusterless-state-space-momentum"].name == "clusterless-state-space-momentum"
    assert models["clusterless-state-space-imm"].name == "clusterless-state-space-imm"
    assert models["clusterless-state-space-diffusion"].config.diffusion_sigma_cm_sqrt_s == 42.0
    assert models["clusterless-state-space-momentum"].config.momentum_sigma_cm_sqrt_s == 43.0


def test_model_evidence_accepts_goal_state_space_model():
    args = argparse.Namespace(
        models=(
            "sorted-spike-state-space-goal "
            "sorted-spike-state-space-goal-bidirectional "
            "sorted-spike-state-space-goal-forward-biased "
            "sorted-spike-state-space-goal-forward-biased-switching "
            "sorted-spike-state-space-goal-reverse-biased "
            "state-space-goal "
            "state-space-goal-bidirectional "
            "state-space-goal-forward-biased "
            "state-space-goal-forward-biased-switching "
            "state-space-goal-reverse-biased"
        ),
        candidate_top_k=64,
        stationary_sigma_cm=2.0,
        diffusion_sigma_cm=12.0,
        momentum_sigma_cm=12.0,
        velocity_decay=0.95,
        mode_stickiness=0.94,
        state_space_stationary_sigma_cm=1.5,
        state_space_diffusion_sigma_cm_sqrt_s=42.0,
        state_space_max_step_sigma=3.0,
        state_space_imm_mode_stickiness=0.91,
        state_space_momentum_sigma_cm_sqrt_s=43.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=44.0,
        state_space_momentum_velocity_decay=0.8,
        state_space_momentum_candidate_top_k=17,
        goal_state_space_transition_sigma_cm_sqrt_s=55.0,
        goal_state_space_lateral_sigma_scale=0.4,
        goal_state_space_diffusion_mixture_weight=0.25,
        goal_state_space_drift_speed_cm_s=250.0,
        goal_state_space_max_step_sigma=3.5,
        goal_state_space_reset_probability=0.03,
        goal_state_space_reset_initial_position_prior_weight=0.4,
        goal_state_space_component_switch_probability=0.07,
        goal_state_space_initial_position_prior_direction_mode="toward",
        goal_state_space_terminal_prior_sigma_cm=12.0,
        goal_state_space_terminal_goal_prior_weight=0.7,
        goal_state_space_initial_goal_prior_sigma_cm=14.0,
        goal_state_space_initial_goal_prior_weight=0.9,
        goal_state_space_toward_direction_prior_weight=0.8,
        goal_state_space_reverse_terminal_position_prior_weight=0.6,
    )

    models = _models(args)

    assert list(models) == [
        "sorted-spike-state-space-goal",
        "sorted-spike-state-space-goal-bidirectional",
        "sorted-spike-state-space-goal-forward-biased",
        "sorted-spike-state-space-goal-forward-biased-switching",
        "sorted-spike-state-space-goal-reverse-biased",
        "state-space-goal",
        "state-space-goal-bidirectional",
        "state-space-goal-forward-biased",
        "state-space-goal-forward-biased-switching",
        "state-space-goal-reverse-biased",
    ]
    assert isinstance(models["sorted-spike-state-space-goal"], GoalStateSpaceReplayModel)
    assert models["sorted-spike-state-space-goal"].transition_sigma_cm_sqrt_s == 55.0
    assert models["sorted-spike-state-space-goal"].lateral_sigma_scale == 0.4
    assert models["sorted-spike-state-space-goal"].diffusion_mixture_weight == 0.25
    assert models["sorted-spike-state-space-goal"].drift_speed_cm_s == 250.0
    assert models["sorted-spike-state-space-goal"].max_step_sigma == 3.5
    assert models["sorted-spike-state-space-goal"].reset_probability == 0.03
    assert models["sorted-spike-state-space-goal"].reset_initial_position_prior_weight == 0.4
    assert models["sorted-spike-state-space-goal"].component_switch_probability == 0.07
    assert models["sorted-spike-state-space-goal"].initial_position_prior_direction_mode == "toward"
    assert models["sorted-spike-state-space-goal"].terminal_goal_prior_sigma_cm == 12.0
    assert models["sorted-spike-state-space-goal"].terminal_goal_prior_weight == 0.7
    assert models["sorted-spike-state-space-goal"].initial_goal_prior_sigma_cm == 14.0
    assert models["sorted-spike-state-space-goal"].initial_goal_prior_weight == 0.9
    assert models["sorted-spike-state-space-goal"].toward_direction_prior_weight == 0.8
    assert models["sorted-spike-state-space-goal"].reverse_terminal_position_prior_weight == 0.6
    assert models["sorted-spike-state-space-goal"].direction_mode == "toward"
    assert models["sorted-spike-state-space-goal-bidirectional"].direction_mode == "bidirectional"
    assert models["sorted-spike-state-space-goal-forward-biased"].direction_mode == "bidirectional"
    assert models["sorted-spike-state-space-goal-forward-biased"].toward_direction_prior_weight == 0.9
    assert models["sorted-spike-state-space-goal-forward-biased-switching"].direction_mode == "bidirectional"
    assert models["sorted-spike-state-space-goal-forward-biased-switching"].toward_direction_prior_weight == 0.9
    assert models["sorted-spike-state-space-goal-forward-biased-switching"].component_switch_probability == 0.03
    assert models["sorted-spike-state-space-goal-reverse-biased"].direction_mode == "bidirectional"
    assert models["sorted-spike-state-space-goal-reverse-biased"].toward_direction_prior_weight == 0.1
    assert models["state-space-goal"].name == "state-space-goal"
    assert models["state-space-goal-bidirectional"].name == "state-space-goal-bidirectional"
    assert models["state-space-goal-forward-biased"].name == "state-space-goal-forward-biased"
    assert models["state-space-goal-forward-biased-switching"].name == "state-space-goal-forward-biased-switching"
    assert models["state-space-goal-reverse-biased"].name == "state-space-goal-reverse-biased"


def test_model_evidence_active_goal_prior_uses_current_task_well():
    args = argparse.Namespace(goal_state_space_active_goal_prior_weight=0.8)

    weights = _goal_prior_weights_for_event(args, _GoalPriorSessionStub(), 0, 2)

    assert np.allclose(weights, [0.2, 0.8])


def test_model_evidence_goal_model_gets_event_specific_prior():
    args = argparse.Namespace(goal_state_space_active_goal_prior_weight=0.8)
    model = GoalStateSpaceReplayModel(candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]))

    event_model = _model_for_event(
        args,
        _GoalPriorSessionStub(),
        0,
        model,
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert event_model is not model
    assert isinstance(event_model, GoalStateSpaceReplayModel)
    assert np.allclose(event_model.goal_prior_weights, [0.2, 0.8])


def test_model_evidence_ripple_position_prior_uses_peak_position():
    args = argparse.Namespace(goal_state_space_ripple_position_prior_sigma_cm=0.25)

    weights = _initial_position_prior_weights_for_event(
        args,
        _GoalPriorSessionStub(),
        0,
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert weights[1] > 0.99


def test_model_evidence_ripple_position_prior_weight_blends_uniform_start():
    args = argparse.Namespace(
        goal_state_space_ripple_position_prior_sigma_cm=0.25,
        goal_state_space_ripple_position_prior_weight=0.5,
    )

    weights = _initial_position_prior_weights_for_event(
        args,
        _GoalPriorSessionStub(),
        0,
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert weights[1] > 0.74
    assert weights[1] < 0.76
    assert np.isclose(weights.sum(), 1.0)


def test_model_evidence_rejects_invalid_ripple_position_prior_weight():
    args = argparse.Namespace(
        goal_state_space_ripple_position_prior_sigma_cm=0.25,
        goal_state_space_ripple_position_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match="ripple-position-prior-weight"):
        _initial_position_prior_weights_for_event(
            args,
            _GoalPriorSessionStub(),
            0,
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        )


def test_model_evidence_goal_model_gets_event_specific_initial_position_prior():
    args = argparse.Namespace(goal_state_space_ripple_position_prior_sigma_cm=0.25)
    model = GoalStateSpaceReplayModel(candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]))

    event_model = _model_for_event(
        args,
        _GoalPriorSessionStub(),
        0,
        model,
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert event_model is not model
    assert isinstance(event_model, GoalStateSpaceReplayModel)
    assert event_model.initial_position_prior_weights[1] > 0.99


def test_model_evidence_reverse_terminal_position_prior_uses_peak_position():
    args = argparse.Namespace(
        goal_state_space_reverse_terminal_position_prior_sigma_cm=0.25,
    )

    weights = _reverse_terminal_position_prior_weights_for_event(
        args,
        _GoalPriorSessionStub(),
        0,
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert weights[1] > 0.99


def test_model_evidence_goal_model_gets_event_specific_reverse_terminal_prior():
    args = argparse.Namespace(
        goal_state_space_reverse_terminal_position_prior_sigma_cm=0.25,
    )
    model = GoalStateSpaceReplayModel(candidate_goals=np.array([[0.0, 0.0], [1.0, 0.0]]))

    event_model = _model_for_event(
        args,
        _GoalPriorSessionStub(),
        0,
        model,
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert event_model is not model
    assert isinstance(event_model, GoalStateSpaceReplayModel)
    assert event_model.reverse_terminal_position_prior_weights[1] > 0.99


def test_model_evidence_rejects_invalid_reverse_terminal_position_prior_weight():
    args = argparse.Namespace(
        goal_state_space_reverse_terminal_position_prior_sigma_cm=0.25,
        goal_state_space_reverse_terminal_position_prior_weight=1.1,
    )

    with pytest.raises(ValueError, match="reverse-terminal-position-prior-weight"):
        _reverse_terminal_position_prior_weights_for_event(
            args,
            _GoalPriorSessionStub(),
            0,
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        )


def test_model_evidence_clusterless_config_records_rate_floor():
    args = argparse.Namespace(
        bin_size_cm=6.0,
        smoothing_sigma_bins=2.0,
        min_speed_cm_s=5.0,
        clusterless_mark_smoothing_sigma_bins=1.5,
        clusterless_mark_prior_count=0.25,
        clusterless_mark_variance_floor=0.75,
        clusterless_rate_floor_hz=1e-3,
    )

    config = _clusterless_mark_config(args)

    assert config.encoding is not None
    assert config.encoding.bin_size_cm == 6.0
    assert config.mark_smoothing_sigma_bins == 1.5
    assert config.mark_prior_count == 0.25
    assert config.mark_variance_floor == 0.75
    assert config.rate_floor_hz == 1e-3


def test_model_evidence_classifies_state_space_families():
    assert _family("sorted-spike-state-space-stationary") == "nontrajectory"
    assert _family("sorted-spike-state-space-diffusion") == "trajectory"
    assert _family("sorted-spike-state-space-fragmented") == "trajectory"
    assert _family("sorted-spike-state-space-goal") == "trajectory"
    assert _family("sorted-spike-state-space-goal-bidirectional") == "trajectory"
    assert _family("sorted-spike-state-space-goal-forward-biased") == "trajectory"
    assert _family("sorted-spike-state-space-goal-forward-biased-switching") == "trajectory"
    assert _family("sorted-spike-state-space-goal-reverse-biased") == "trajectory"
    assert _family("sorted-spike-state-space-momentum") == "trajectory"
    assert _family("sorted-spike-state-space-imm") == "trajectory"
    assert _family("state-space-goal") == "trajectory"
    assert _family("state-space-goal-bidirectional") == "trajectory"
    assert _family("state-space-goal-forward-biased") == "trajectory"
    assert _family("state-space-goal-forward-biased-switching") == "trajectory"
    assert _family("state-space-goal-reverse-biased") == "trajectory"
    assert _family("clusterless-state-space-stationary") == "nontrajectory"
    assert _family("clusterless-state-space-diffusion") == "trajectory"
    assert _family("clusterless-state-space-fragmented") == "trajectory"
    assert _family("clusterless-state-space-momentum") == "trajectory"
    assert _family("clusterless-state-space-imm") == "trajectory"


def test_model_evidence_run_event_selection_uses_session_event_ids():
    assert _events("run", _SessionStub()) == [2, 4, 6, 8]
    assert _events("run:1-2", _SessionStub()) == [4, 6]


def test_model_evidence_excludes_truncated_lower_bounds_from_exact_probabilities():
    scored = _add_evidence_columns(
        pd.DataFrame(
            [
                _score_row("random", 0.0),
                _score_row("stationary", -2.0),
                _score_row(
                    "momentum",
                    100.0,
                    diagnostic_candidate_evidence_support="truncated_full_grid",
                ),
            ]
        )
    )

    random = scored[scored["model"] == "random"].iloc[0]
    momentum = scored[scored["model"] == "momentum"].iloc[0]

    assert bool(random["is_best_model"])
    assert random["best_model"] == "random"
    assert not bool(momentum["evidence_comparable"])
    assert not bool(momentum["is_best_model"])
    assert pd.isna(momentum["relative_log_evidence"])
    assert pd.isna(momentum["model_probability"])
    assert momentum["best_truncated_lower_bound_model"] == "momentum"
    assert bool(momentum["is_best_truncated_lower_bound"])
    assert momentum["truncated_relative_log_evidence"] == 0.0


def _score_row(model: str, log_evidence: float, **extra: object) -> dict[str, object]:
    row = {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": 0,
        "model": model,
        "requested_model": model,
        "model_family": _family(model),
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
    }
    row.update(extra)
    return row
