import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_SCRIPT = _SCRIPTS_DIR / "aggregate_model_evidence_shards.py"
_SPEC = importlib.util.spec_from_file_location("aggregate_model_evidence_shards", _SCRIPT)
assert _SPEC is not None
aggregate_model_evidence_shards = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(aggregate_model_evidence_shards)

_validate_constant_settings = aggregate_model_evidence_shards._validate_constant_settings


def test_validate_constant_settings_accepts_consistent_shards():
    _validate_constant_settings(
        pd.DataFrame(
            [
                _row(event_index=0, spike_rate_scale=2.0),
                _row(event_index=1, spike_rate_scale=2.0),
            ]
        )
    )


def test_validate_constant_settings_rejects_mixed_spike_rate_scale():
    frame = pd.DataFrame(
        [
            _row(event_index=0, spike_rate_scale=1.0),
            _row(event_index=1, spike_rate_scale=2.0),
        ]
    )

    with pytest.raises(ValueError, match="spike_rate_scale"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_clusterless_hyperparameters():
    frame = pd.DataFrame(
        [
            _row(event_index=0, clusterless_mark_prior_count=0.5),
            _row(event_index=1, clusterless_mark_prior_count=1.0),
        ]
    )

    with pytest.raises(ValueError, match="clusterless_mark_prior_count"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_clusterless_rate_floor():
    frame = pd.DataFrame(
        [
            _row(event_index=0, clusterless_rate_floor_hz=1e-4),
            _row(event_index=1, clusterless_rate_floor_hz=1e-3),
        ]
    )

    with pytest.raises(ValueError, match="clusterless_rate_floor_hz"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_goal_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_active_goal_prior_weight=0.0),
            _row(event_index=1, goal_state_space_active_goal_prior_weight=0.7),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_active_goal_prior_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_goal_reset_probability():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_reset_probability=0.0),
            _row(event_index=1, goal_state_space_reset_probability=0.05),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_reset_probability"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_goal_lateral_sigma_scale():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_lateral_sigma_scale=1.0),
            _row(event_index=1, goal_state_space_lateral_sigma_scale=0.5),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_lateral_sigma_scale"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_goal_diffusion_mixture_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_diffusion_mixture_weight=0.0),
            _row(event_index=1, goal_state_space_diffusion_mixture_weight=0.25),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_diffusion_mixture_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_reset_initial_position_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_reset_initial_position_prior_weight=0.0),
            _row(event_index=1, goal_state_space_reset_initial_position_prior_weight=1.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_reset_initial_position_prior_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_component_switch_probability():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_component_switch_probability=0.0),
            _row(event_index=1, goal_state_space_component_switch_probability=0.05),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_component_switch_probability"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_initial_position_prior_direction_mode():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_initial_position_prior_direction_mode="all"),
            _row(event_index=1, goal_state_space_initial_position_prior_direction_mode="toward"),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_initial_position_prior_direction_mode"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_terminal_goal_prior_sigma():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_terminal_prior_sigma_cm=0.0),
            _row(event_index=1, goal_state_space_terminal_prior_sigma_cm=20.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_terminal_prior_sigma_cm"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_terminal_goal_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_terminal_goal_prior_weight=0.5),
            _row(event_index=1, goal_state_space_terminal_goal_prior_weight=1.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_terminal_goal_prior_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_initial_goal_prior_sigma():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_initial_goal_prior_sigma_cm=0.0),
            _row(event_index=1, goal_state_space_initial_goal_prior_sigma_cm=20.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_initial_goal_prior_sigma_cm"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_initial_goal_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_initial_goal_prior_weight=0.5),
            _row(event_index=1, goal_state_space_initial_goal_prior_weight=1.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_initial_goal_prior_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_toward_direction_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_toward_direction_prior_weight=0.5),
            _row(event_index=1, goal_state_space_toward_direction_prior_weight=0.8),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_toward_direction_prior_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_ripple_position_prior_sigma():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_ripple_position_prior_sigma_cm=0.0),
            _row(event_index=1, goal_state_space_ripple_position_prior_sigma_cm=20.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_ripple_position_prior_sigma_cm"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_ripple_position_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_ripple_position_prior_weight=0.5),
            _row(event_index=1, goal_state_space_ripple_position_prior_weight=1.0),
        ]
    )

    with pytest.raises(ValueError, match="goal_state_space_ripple_position_prior_weight"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_reverse_terminal_position_prior_sigma():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_reverse_terminal_position_prior_sigma_cm=0.0),
            _row(event_index=1, goal_state_space_reverse_terminal_position_prior_sigma_cm=20.0),
        ]
    )

    with pytest.raises(
        ValueError,
        match="goal_state_space_reverse_terminal_position_prior_sigma_cm",
    ):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_reverse_terminal_position_prior_weight():
    frame = pd.DataFrame(
        [
            _row(event_index=0, goal_state_space_reverse_terminal_position_prior_weight=0.5),
            _row(event_index=1, goal_state_space_reverse_terminal_position_prior_weight=1.0),
        ]
    )

    with pytest.raises(
        ValueError,
        match="goal_state_space_reverse_terminal_position_prior_weight",
    ):
        _validate_constant_settings(frame)


def _row(
    *,
    event_index: int,
    spike_rate_scale: float = 1.0,
    goal_state_space_lateral_sigma_scale: float = 1.0,
    goal_state_space_diffusion_mixture_weight: float = 0.0,
    goal_state_space_reset_probability: float = 0.0,
    goal_state_space_reset_initial_position_prior_weight: float = 0.0,
    goal_state_space_component_switch_probability: float = 0.0,
    goal_state_space_initial_position_prior_direction_mode: str = "all",
    goal_state_space_terminal_prior_sigma_cm: float = 0.0,
    goal_state_space_terminal_goal_prior_weight: float = 1.0,
    goal_state_space_initial_goal_prior_sigma_cm: float = 0.0,
    goal_state_space_initial_goal_prior_weight: float = 1.0,
    goal_state_space_toward_direction_prior_weight: float = 0.5,
    goal_state_space_active_goal_prior_weight: float = 0.0,
    goal_state_space_ripple_position_prior_sigma_cm: float = 0.0,
    goal_state_space_ripple_position_prior_weight: float = 1.0,
    goal_state_space_reverse_terminal_position_prior_sigma_cm: float = 0.0,
    goal_state_space_reverse_terminal_position_prior_weight: float = 1.0,
    clusterless_mark_prior_count: float = 1.0,
    clusterless_rate_floor_hz: float = 1e-4,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "model": "clusterless-state-space-diffusion",
        "requested_model": "clusterless-state-space-diffusion",
        "model_family": "trajectory",
        "log_evidence": -1.0,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
        "spike_rate_scale": spike_rate_scale,
        "goal_state_space_lateral_sigma_scale": goal_state_space_lateral_sigma_scale,
        "goal_state_space_diffusion_mixture_weight": goal_state_space_diffusion_mixture_weight,
        "goal_state_space_reset_probability": goal_state_space_reset_probability,
        "goal_state_space_reset_initial_position_prior_weight": goal_state_space_reset_initial_position_prior_weight,
        "goal_state_space_component_switch_probability": goal_state_space_component_switch_probability,
        "goal_state_space_initial_position_prior_direction_mode": (
            goal_state_space_initial_position_prior_direction_mode
        ),
        "goal_state_space_terminal_prior_sigma_cm": goal_state_space_terminal_prior_sigma_cm,
        "goal_state_space_terminal_goal_prior_weight": goal_state_space_terminal_goal_prior_weight,
        "goal_state_space_initial_goal_prior_sigma_cm": goal_state_space_initial_goal_prior_sigma_cm,
        "goal_state_space_initial_goal_prior_weight": goal_state_space_initial_goal_prior_weight,
        "goal_state_space_toward_direction_prior_weight": goal_state_space_toward_direction_prior_weight,
        "goal_state_space_active_goal_prior_weight": goal_state_space_active_goal_prior_weight,
        "goal_state_space_ripple_position_prior_sigma_cm": goal_state_space_ripple_position_prior_sigma_cm,
        "goal_state_space_ripple_position_prior_weight": goal_state_space_ripple_position_prior_weight,
        "goal_state_space_reverse_terminal_position_prior_sigma_cm": (
            goal_state_space_reverse_terminal_position_prior_sigma_cm
        ),
        "goal_state_space_reverse_terminal_position_prior_weight": (
            goal_state_space_reverse_terminal_position_prior_weight
        ),
        "clusterless_mark_smoothing_sigma_bins": 1.0,
        "clusterless_mark_prior_count": clusterless_mark_prior_count,
        "clusterless_mark_variance_floor": 1.0,
        "clusterless_rate_floor_hz": clusterless_rate_floor_hz,
    }
