from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hipporeplayimm.clusterless_ground_truth import (
    _clusterless_model_config_for_scores,
    _clusterless_state_space_model,
)
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.ground_truth import compare_scores_to_ground_truth


def test_clusterless_state_space_model_preserves_duration_and_displacement_config():
    config = SimpleNamespace(
        state_space_imm_mode_stickiness=0.77,
        state_space_imm_switch_tau_s=0.060,
        state_space_trajectory_imm_mode_stickiness=0.88,
        state_space_trajectory_imm_momentum_initial_probability=0.12,
        state_space_trajectory_imm_momentum_switch_probability=0.03,
        state_space_displacement_radius_bins=4,
        state_space_displacement_position_sigma_cm=1.25,
        state_space_displacement_transition_sigma_cm_sqrt_s=55.0,
        state_space_displacement_prior_sigma_cm=2.5,
    )

    model = _clusterless_state_space_model(config, "displacement-imm")

    assert model.config.imm_mode_stickiness == 0.77
    assert model.config.imm_switch_tau_s == 0.060
    assert model.config.trajectory_imm_mode_stickiness == 0.88
    assert model.config.trajectory_imm_momentum_initial_probability == 0.12
    assert model.config.trajectory_imm_momentum_switch_probability == 0.03
    assert model.config.displacement_radius_bins == 4
    assert model.config.displacement_position_sigma_cm == 1.25
    assert model.config.displacement_transition_sigma_cm_sqrt_s == 55.0
    assert model.config.displacement_prior_sigma_cm == 2.5


def test_clusterless_model_config_for_scores_recovers_state_space_metadata():
    scores = pd.DataFrame(
        {
            "state_space_imm_switch_tau_s": [0.060],
            "state_space_trajectory_imm_mode_stickiness": [0.88],
            "state_space_displacement_radius_bins": [4],
            "diagnostic_state_space_displacement_position_sigma_cm": [1.25],
            "state_space_displacement_transition_sigma_cm_sqrt_s": [55.0],
            "diagnostic_state_space_displacement_prior_sigma_cm": [2.5],
        }
    )

    config = _clusterless_model_config_for_scores(
        scores,
        model_names=("clusterless-state-space-displacement-imm",),
        **_clusterless_config_defaults(),
    )

    assert config.state_space_imm_switch_tau_s == 0.060
    assert config.state_space_trajectory_imm_mode_stickiness == 0.88
    assert config.state_space_displacement_radius_bins == 4
    assert config.state_space_displacement_position_sigma_cm == 1.25
    assert config.state_space_displacement_transition_sigma_cm_sqrt_s == 55.0
    assert config.state_space_displacement_prior_sigma_cm == 2.5


def test_compare_scores_to_ground_truth_decodes_clusterless_state_space(
    monkeypatch, tmp_path: Path
):
    position_times = np.linspace(0.0, 5.0, 51)
    x = np.where(position_times < 2.5, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.2, 0.4, 0.6, 3.2, 3.4, 3.6, 4.2])
    marks = np.array([[0.0], [0.1], [-0.1], [10.0], [10.2], [9.8], [10.1]])
    cell_ids = np.ones(mark_times.shape[0], dtype=int)
    session = ReplaySession(
        rat="Rat1",
        name="Open1",
        path=tmp_path,
        position=position,
        spikes=np.column_stack([mark_times, cell_ids]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[4.0, 4.5, 4.25, 1.0, 1.0, 1.0]]),
        run_times=np.array([[0.0, 5.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=marks,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
        ),
    )
    wells = pd.DataFrame(
        {
            "well_id": [1, 2],
            "well_x": [0.0, 10.0],
            "well_y": [0.0, 0.0],
            "n_estimates": [1, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "status": ["success"],
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["clusterless-state-space-diffusion"],
            "requested_model": ["clusterless-state-space-diffusion"],
            "log_evidence": [0.0],
            "bin_size_cm": [10.0],
            "smoothing_sigma_bins": [0.0],
            "min_speed_cm_s": [0.0],
            "encoding_arena_padding_cm": [5.0],
            "time_bin_s": [0.5],
            "clusterless_mark_smoothing_sigma_bins": [0.0],
            "clusterless_mark_prior_count": [0.1],
            "clusterless_mark_variance_floor": [0.05],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "ripple_peak": [4.25],
            "active_goal_id": [np.nan],
            "true_well_id": [2],
            "true_well_x": [10.0],
            "true_well_y": [0.0],
            "arrival_time": [4.5],
            "time_to_arrival_s": [0.25],
            "valid_label": [True],
            "exclude_reason": [""],
        }
    )

    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.load_open_field_sessions",
        lambda _root: [session],
    )
    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.infer_well_locations",
        lambda _session, _config=None: wells,
    )

    comparison = compare_scores_to_ground_truth(
        tmp_path,
        scores,
        ground_truth=ground_truth,
    )

    assert np.isfinite(comparison.loc[0, "decoded_endpoint_x"])
    assert comparison.loc[0, "decoded_well_id"] == 2
    assert bool(comparison.loc[0, "goal_correct"])


def _clusterless_config_defaults() -> dict[str, object]:
    return {
        "state_space_stationary_sigma_cm": 2.0,
        "state_space_diffusion_sigma_cm_sqrt_s": 85.0,
        "state_space_max_step_sigma": 4.0,
        "state_space_imm_mode_stickiness": 0.95,
        "state_space_imm_switch_tau_s": 0.0,
        "state_space_trajectory_imm_mode_stickiness": None,
        "state_space_trajectory_imm_momentum_initial_probability": None,
        "state_space_trajectory_imm_momentum_switch_probability": None,
        "state_space_momentum_sigma_cm_sqrt_s": 85.0,
        "state_space_momentum_initial_sigma_cm_sqrt_s": 85.0,
        "state_space_momentum_velocity_decay": 0.95,
        "state_space_momentum_velocity_decay_tau_s": 0.0,
        "state_space_momentum_candidate_top_k": 128,
        "state_space_momentum_candidate_mass_threshold": None,
        "state_space_momentum_candidate_min_k": 1,
        "state_space_momentum_candidate_max_k": 0,
        "state_space_momentum_predicted_candidate_top_k": 8,
        "state_space_momentum_candidate_source": "emission",
        "state_space_displacement_radius_bins": 2,
        "state_space_displacement_position_sigma_cm": 0.0,
        "state_space_displacement_transition_sigma_cm_sqrt_s": 0.0,
        "state_space_displacement_prior_sigma_cm": 0.0,
        "state_space_valid_occupancy_threshold_s": 0.0,
        "clusterless_mark_smoothing_sigma_bins": 1.0,
        "clusterless_mark_prior_count": 1.0,
        "clusterless_mark_variance_floor": 1.0,
        "clusterless_rate_floor_hz": 1e-4,
        "clusterless_mark_likelihood": "local-kde",
        "clusterless_mark_kde_bandwidth": None,
        "clusterless_mark_kde_spatial_sigma_bins": None,
        "clusterless_mark_kde_max_neighbors": 256,
    }
