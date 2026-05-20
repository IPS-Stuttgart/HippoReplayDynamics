from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, EncodingModel, LogEmissionTensor
from hipporeplayimm.ground_truth import (
    GroundTruthConfig,
    GroundTruthSensitivityConfig,
    _add_ground_truth_metrics,
    _decoded_row,
    _emission_config_for_scores,
    assign_endpoint_to_well,
    compare_scores_to_ground_truth,
    compare_scores_to_ground_truth_sensitivity,
    first_post_ripple_well_visit,
    infer_well_locations_from_arrays,
    label_session_behavioral_ground_truth,
    trajectory_well_posterior_masses,
    well_posterior_masses,
)
from hipporeplayimm.models import EventScore


def test_shifted_well_coordinate_inference():
    times = np.linspace(0.0, 20.0, 201)
    x = np.where(times < 12.0, 10.0, 80.0)
    y = np.where(times < 12.0, 20.0, 90.0)
    position = np.column_stack([times, x, y, np.zeros_like(times)])
    well_sequence = np.array([[0.0, 1.0], [10.0, 2.0], [20.0, 1.0]])

    wells = infer_well_locations_from_arrays(position, well_sequence, well_arrival_window_s=1.0)

    well_1 = wells[wells["well_id"] == 1].iloc[0]
    well_2 = wells[wells["well_id"] == 2].iloc[0]
    assert well_1["well_x"] == pytest.approx(10.0)
    assert well_1["well_y"] == pytest.approx(20.0)
    assert well_2["well_x"] == pytest.approx(80.0)
    assert well_2["well_y"] == pytest.approx(90.0)


def test_first_post_ripple_well_visit_uses_dwell_threshold():
    times = np.linspace(0.0, 10.0, 101)
    x = np.where(times < 5.0, 0.0, 50.0)
    y = np.zeros_like(times)
    position = np.column_stack([times, x, y, np.zeros_like(times)])
    wells = pd.DataFrame(
        {
            "well_id": [1, 2],
            "well_x": [0.0, 50.0],
            "well_y": [0.0, 0.0],
            "n_estimates": [1, 1],
        }
    )

    visit = first_post_ripple_well_visit(
        position,
        wells,
        ripple_peak=4.0,
        visit_radius_cm=5.0,
        min_dwell_s=0.2,
        future_horizon_s=5.0,
    )

    assert visit is not None
    assert visit["well_id"] == 1


def test_label_session_behavioral_ground_truth_marks_valid_next_well(tmp_path: Path):
    times = np.linspace(0.0, 20.0, 401)
    x = np.where(times < 10.0, 10.0, 80.0)
    y = np.where(times < 10.0, 20.0, 90.0)
    position = np.column_stack([times, x, y, np.zeros_like(times)])
    session = _session(
        tmp_path,
        position=position,
        well_sequence=np.array([[0.0, 1.0], [10.0, 2.0], [20.0, 1.0]]),
        ripple_events=np.array([[8.0, 8.1, 8.05, 1.0, 1.0, 1.0]]),
    )

    labels = label_session_behavioral_ground_truth(
        session,
        GroundTruthConfig(future_horizon_s=5.0),
    )

    assert labels.loc[0, "valid_label"]
    assert labels.loc[0, "true_well_id"] == 1


def test_endpoint_assignment_and_true_well_posterior_mass():
    wells = pd.DataFrame(
        {
            "well_id": [1, 2],
            "well_x": [0.0, 10.0],
            "well_y": [0.0, 0.0],
            "n_estimates": [1, 1],
        }
    )
    bin_centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    log_posterior = np.log(np.array([0.25, 0.75]))

    assigned = assign_endpoint_to_well(np.array([8.0, 0.0]), wells)
    masses = well_posterior_masses(log_posterior, bin_centers, wells, radius_cm=2.0)

    assert assigned is not None
    assert assigned["well_id"] == 2
    assert masses[2] == pytest.approx(0.75)


def test_trajectory_well_posterior_mass_summaries():
    wells = pd.DataFrame(
        {
            "well_id": [1, 2],
            "well_x": [0.0, 10.0],
            "well_y": [0.0, 0.0],
            "n_estimates": [1, 1],
        }
    )
    bin_centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    trajectory_log_posterior = np.log(
        np.array(
            [
                [0.90, 0.10],
                [0.20, 0.80],
            ]
        )
    )

    masses = trajectory_well_posterior_masses(trajectory_log_posterior, bin_centers, wells, radius_cm=2.0)

    assert masses[1]["initial"] == pytest.approx(0.90)
    assert masses[2]["max"] == pytest.approx(0.80)
    assert masses[2]["max_time_index"] == 1
    assert masses[2]["trajectory"] == pytest.approx(0.45)


def test_decoded_row_adds_time_resolved_well_metrics():
    wells = pd.DataFrame(
        {
            "well_id": [1, 2],
            "well_x": [0.0, 30.0],
            "well_y": [0.0, 0.0],
            "n_estimates": [1, 1],
        }
    )
    bin_centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    terminal_log_posterior = np.log(np.array([0.8, 0.2]))
    trajectory_log_posterior = np.log(np.array([[0.1, 0.9], [0.8, 0.2]]))

    row = _decoded_row(
        "Rat1/Open1",
        0,
        "diffusion",
        terminal_log_posterior,
        trajectory_log_posterior,
        bin_centers,
        wells,
    )

    assert row["decoded_well_id"] == 1
    assert row["decoded_max_posterior_well_id"] == 2
    assert row["decoded_integrated_well_id"] == 2
    assert row["well_2_max_posterior"] == pytest.approx(0.9)
    assert row["well_2_integrated_posterior"] == pytest.approx(0.55)


def test_compare_scores_to_ground_truth_preserves_score_columns(tmp_path: Path):
    root = tmp_path / "dataset"
    session_path = root / "Rat1" / "Open1"
    session_path.mkdir(parents=True)
    _write_minimal_session(session_path)
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["random"],
            "heldout_log_likelihood": [-1.0],
            "delta_vs_best_static": [0.0],
            "bits_per_spike_vs_best_static": [0.0],
        }
    )

    comparison = compare_scores_to_ground_truth(
        root,
        scores,
        ground_truth_config=GroundTruthConfig(future_horizon_s=5.0),
    )

    assert "heldout_log_likelihood" in comparison.columns
    assert "goal_correct" in comparison.columns
    assert len(comparison) == 1


def test_emission_config_for_scores_reads_legacy_event_evidence_metadata():
    scores = pd.DataFrame(
        {
            "time_bin_s": [0.003],
            "spike_rate_scale": [4.0],
        }
    )

    config = _emission_config_for_scores(
        scores,
        EmissionConfig(time_bin_s=0.02, spike_rate_scale=1.0),
    )

    assert config.time_bin_s == pytest.approx(0.003)
    assert config.spike_rate_scale == pytest.approx(4.0)


def test_emission_config_for_scores_rejects_conflicting_metadata():
    scores = pd.DataFrame({"emission_time_bin_s": [0.02], "time_bin_s": [0.003]})

    with pytest.raises(ValueError, match="emission_time_bin_s / time_bin_s"):
        _emission_config_for_scores(scores, EmissionConfig())


def test_compare_scores_to_ground_truth_uses_benchmark_split_and_train_candidates(
    monkeypatch, tmp_path: Path
):
    times = np.linspace(0.0, 1.0, 11)
    position = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)]
    )
    session = _session(
        tmp_path,
        position=position,
        well_sequence=np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 1.0]]),
        ripple_events=np.array([[0.2, 0.24, 0.22, 1.0, 1.0, 1.0]]),
    )
    bin_centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    encoding = EncodingModel(
        x_edges=np.array([-1.0, 5.0, 11.0]),
        y_edges=np.array([-1.0, 1.0]),
        bin_centers=bin_centers,
        rates_hz=np.ones((4, 2)),
        occupancy_s=np.ones(2),
        cell_ids=np.array([1, 2, 3, 4]),
        config=EncodingConfig(),
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
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["diffusion"],
            "requested_model": ["diffusion"],
            "heldout_log_likelihood": [0.0],
            "train_log_likelihood": [0.0],
            "joint_log_likelihood": [0.0],
            "train_cell_ids": ["1,2,3"],
            "test_cell_ids": ["4"],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "ripple_peak": [0.22],
            "active_goal_id": [np.nan],
            "true_well_id": [2],
            "true_well_x": [10.0],
            "true_well_y": [0.0],
            "arrival_time": [0.5],
            "time_to_arrival_s": [0.28],
            "valid_label": [True],
            "exclude_reason": [""],
        }
    )

    built_cell_ids: list[tuple[int, ...]] = []
    seen: dict[str, object] = {}

    def fake_build_emissions(session_arg, encoding_arg, event_index_arg, emission_config_arg):
        del session_arg, event_index_arg, emission_config_arg
        built_cell_ids.append(tuple(int(cell_id) for cell_id in encoding_arg.cell_ids))
        return LogEmissionTensor(
            log_likelihood=np.array([[0.0, -1.0], [-1.0, 0.0]]),
            spike_counts=np.zeros((2, encoding_arg.n_cells), dtype=int),
            times=np.array([0.21, 0.23]),
            dt=0.02,
            cell_ids=encoding_arg.cell_ids,
            n_spikes=0,
        )

    class FakeCandidateModel:
        name = "diffusion"

        def candidate_indices(self, emissions):
            seen["candidate_cells"] = tuple(int(cell_id) for cell_id in emissions.cell_ids)
            return [np.array([0]), np.array([1])]

        def score(self, emissions, bin_centers_arg, candidate_indices=None):
            del bin_centers_arg
            seen["score_cells"] = tuple(int(cell_id) for cell_id in emissions.cell_ids)
            seen["candidate_indices"] = candidate_indices
            return EventScore(
                "diffusion",
                0.0,
                emissions.n_time,
                emissions.n_spikes,
                terminal_log_posterior=np.log(np.array([0.25, 0.75])),
                trajectory_log_posterior=np.log(
                    np.array(
                        [
                            [0.60, 0.40],
                            [0.25, 0.75],
                        ]
                    )
                ),
            )

    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.load_open_field_sessions",
        lambda _root: [session],
    )
    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.fit_place_field_encoding",
        lambda _session, _config: encoding,
    )
    monkeypatch.setattr("hipporeplayimm.ground_truth.build_emissions", fake_build_emissions)
    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.infer_well_locations",
        lambda _session, _config=None: wells,
    )
    monkeypatch.setattr(
        "hipporeplayimm.ground_truth._build_models",
        lambda _config, session=None: {"diffusion": FakeCandidateModel()},
    )

    comparison = compare_scores_to_ground_truth(
        tmp_path,
        scores,
        ground_truth=ground_truth,
    )

    assert built_cell_ids == [(1, 2, 3), (1, 2, 3, 4)]
    assert seen["candidate_cells"] == (1, 2, 3)
    assert seen["score_cells"] == (1, 2, 3, 4)
    assert [arr.tolist() for arr in seen["candidate_indices"]] == [[0], [1]]
    assert comparison.loc[0, "decoded_well_id"] == 2
    assert bool(comparison.loc[0, "goal_correct"])
    assert comparison.loc[0, "true_initial_well_posterior"] == pytest.approx(0.40)
    assert comparison.loc[0, "true_max_well_posterior"] == pytest.approx(0.75)
    assert comparison.loc[0, "true_trajectory_well_posterior"] == pytest.approx(0.575)
    assert comparison.loc[0, "true_initial_well_rank"] == 2
    assert comparison.loc[0, "true_max_well_rank"] == 1
    assert bool(comparison.loc[0, "max_over_time_goal_correct"])
    assert bool(comparison.loc[0, "trajectory_mean_goal_correct"])


def test_compare_scores_to_ground_truth_adds_exact_bayesian_model_average(
    monkeypatch,
    tmp_path: Path,
):
    times = np.linspace(0.0, 1.0, 11)
    position = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)]
    )
    session = _session(
        tmp_path,
        position=position,
        well_sequence=np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 1.0]]),
        ripple_events=np.array([[0.2, 0.24, 0.22, 1.0, 1.0, 1.0]]),
    )
    bin_centers = np.array([[0.0, 0.0], [30.0, 0.0]])
    encoding = EncodingModel(
        x_edges=np.array([-1.0, 15.0, 31.0]),
        y_edges=np.array([-1.0, 1.0]),
        bin_centers=bin_centers,
        rates_hz=np.ones((2, 2)),
        occupancy_s=np.ones(2),
        cell_ids=np.array([1, 2]),
        config=EncodingConfig(),
    )
    wells = pd.DataFrame(
        {
            "well_id": [1, 2],
            "well_x": [0.0, 30.0],
            "well_y": [0.0, 0.0],
            "n_estimates": [1, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["left", "right"],
            "requested_model": ["left", "right"],
            "status": ["success", "success"],
            "evidence_support": ["exact_full_grid", "exact_full_grid"],
            "log_evidence": [0.0, np.log(3.0)],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "ripple_peak": [0.22],
            "active_goal_id": [np.nan],
            "true_well_id": [2],
            "true_well_x": [30.0],
            "true_well_y": [0.0],
            "arrival_time": [0.5],
            "time_to_arrival_s": [0.28],
            "valid_label": [True],
            "exclude_reason": [""],
        }
    )

    def fake_build_emissions(session_arg, encoding_arg, event_index_arg, emission_config_arg):
        del session_arg, event_index_arg, emission_config_arg
        return LogEmissionTensor(
            log_likelihood=np.zeros((1, 2)),
            spike_counts=np.zeros((1, encoding_arg.n_cells), dtype=int),
            times=np.array([0.22]),
            dt=0.02,
            cell_ids=encoding_arg.cell_ids,
            n_spikes=0,
        )

    class FakeModel:
        def __init__(self, name: str, posterior: np.ndarray):
            self.name = name
            self.posterior = posterior

        def score(self, emissions, bin_centers_arg):
            del bin_centers_arg
            return EventScore(
                self.name,
                0.0,
                emissions.n_time,
                emissions.n_spikes,
                terminal_log_posterior=np.log(self.posterior),
            )

    monkeypatch.setattr("hipporeplayimm.ground_truth.load_open_field_sessions", lambda _root: [session])
    monkeypatch.setattr("hipporeplayimm.ground_truth.fit_place_field_encoding", lambda _session, _config: encoding)
    monkeypatch.setattr("hipporeplayimm.ground_truth.build_emissions", fake_build_emissions)
    monkeypatch.setattr("hipporeplayimm.ground_truth.infer_well_locations", lambda _session, _config=None: wells)
    monkeypatch.setattr(
        "hipporeplayimm.ground_truth._build_models",
        lambda _config, session=None: {
            "left": FakeModel("left", np.array([0.9, 0.1])),
            "right": FakeModel("right", np.array([0.1, 0.9])),
        },
    )

    comparison = compare_scores_to_ground_truth(tmp_path, scores, ground_truth=ground_truth)

    average = comparison[comparison["model"] == "bayesian-model-average"].iloc[0]
    assert set(comparison["model"]) == {"left", "right", "bayesian-model-average"}
    assert int(average["bma_component_count"]) == 2
    assert average["bma_component_models"] == "left,right"
    assert average["decoded_endpoint_x"] == pytest.approx(21.0)
    assert average["well_2_posterior"] == pytest.approx(0.7)
    assert bool(average["goal_correct"])


def test_ground_truth_metrics_treat_missing_valid_label_as_invalid():
    comparison = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 1, 2],
            "model": ["random", "random", "random"],
            "true_well_id": [np.nan, 2.0, 1.0],
            "true_well_x": [np.nan, 10.0, 0.0],
            "true_well_y": [np.nan, 0.0, 0.0],
            "valid_label": [np.nan, True, "False"],
            "decoded_endpoint_x": [0.0, 10.0, 0.0],
            "decoded_endpoint_y": [0.0, 0.0, 0.0],
            "decoded_well_id": [1.0, 2.0, 1.0],
            "well_1_posterior": [0.8, 0.2, 0.9],
            "well_2_posterior": [0.2, 0.8, 0.1],
        }
    )

    result = _add_ground_truth_metrics(
        comparison,
        decoded=pd.DataFrame(),
        gt_frame=pd.DataFrame(),
    )

    assert pd.isna(result.loc[0, "goal_correct"])
    assert pd.isna(result.loc[0, "endpoint_error_cm"])
    assert pd.isna(result.loc[0, "true_well_posterior"])
    assert pd.isna(result.loc[0, "true_well_rank"])
    assert bool(result.loc[1, "goal_correct"])
    assert result.loc[1, "true_well_posterior"] == pytest.approx(0.8)
    assert result.loc[1, "true_well_rank"] == 1
    assert pd.isna(result.loc[2, "goal_correct"])
    assert pd.isna(result.loc[2, "true_well_posterior"])
    assert pd.isna(result.loc[2, "true_well_rank"])


def test_ground_truth_sensitivity_relabels_without_redecoding(monkeypatch, tmp_path: Path):
    base_comparison = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["left", "right"],
            "log_likelihood": [-3.0, -2.0],
            "decoded_endpoint_x": [0.0, 10.0],
            "decoded_endpoint_y": [0.0, 0.0],
            "decoded_well_id": [1.0, 2.0],
            "well_1_posterior": [0.9, 0.1],
            "well_2_posterior": [0.1, 0.9],
            "true_well_id": [99.0, 99.0],
            "true_well_x": [99.0, 99.0],
            "true_well_y": [99.0, 99.0],
            "valid_label": [True, True],
            "goal_correct": [False, False],
        }
    )
    compare_calls: list[GroundTruthConfig] = []
    generated_configs: list[GroundTruthConfig] = []

    def fake_compare_scores_to_ground_truth(root, scores, **kwargs):
        del root, scores
        compare_calls.append(kwargs["ground_truth_config"])
        return base_comparison.copy()

    def fake_generate_behavioral_ground_truth(root, config=None):
        del root
        assert config is not None
        generated_configs.append(config)
        true_well_id = 1 if config.future_horizon_s < 20.0 else 2
        true_x = 0.0 if true_well_id == 1 else 10.0
        return pd.DataFrame(
            {
                "session": ["Rat1/Open1"],
                "event_index": [0],
                "ripple_peak": [1.0],
                "active_goal_id": [np.nan],
                "true_well_id": [true_well_id],
                "true_well_x": [true_x],
                "true_well_y": [0.0],
                "arrival_time": [2.0],
                "time_to_arrival_s": [1.0],
                "valid_label": [True],
                "exclude_reason": [""],
            }
        )

    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.compare_scores_to_ground_truth",
        fake_compare_scores_to_ground_truth,
    )
    monkeypatch.setattr(
        "hipporeplayimm.ground_truth.generate_behavioral_ground_truth",
        fake_generate_behavioral_ground_truth,
    )

    result = compare_scores_to_ground_truth_sensitivity(
        tmp_path,
        pd.DataFrame({"session": ["Rat1/Open1"]}),
        sensitivity_config=GroundTruthSensitivityConfig(
            visit_radii_cm=(10.0,),
            min_dwells_s=(0.2,),
            future_horizons_s=(10.0, 30.0),
        ),
    )

    assert len(compare_calls) == 1
    assert len(generated_configs) == 2
    assert set(result.rows["true_well_id"]) == {1, 2}
    assert result.rows["goal_correct"].tolist() == [True, False, False, True]
    assert set(result.per_setting_summary["future_horizon_s"]) == {10.0, 30.0}
    robustness = result.robustness_summary.set_index("model")
    assert robustness.loc["left", "settings"] == 2
    assert robustness.loc["left", "min_goal_accuracy"] == pytest.approx(0.0)
    assert robustness.loc["left", "max_goal_accuracy"] == pytest.approx(1.0)
    assert robustness.loc["right", "goal_accuracy_range"] == pytest.approx(1.0)


def test_ground_truth_metrics_rank_terminal_max_and_integrated_separately():
    comparison = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["diffusion"],
            "true_well_id": [2.0],
            "true_well_x": [30.0],
            "true_well_y": [0.0],
            "valid_label": [True],
            "decoded_endpoint_x": [6.0],
            "decoded_endpoint_y": [0.0],
            "decoded_well_id": [1.0],
            "decoded_max_posterior_well_id": [2.0],
            "decoded_integrated_endpoint_x": [16.5],
            "decoded_integrated_endpoint_y": [0.0],
            "decoded_integrated_well_id": [2.0],
            "well_1_posterior": [0.8],
            "well_2_posterior": [0.2],
            "well_1_max_posterior": [0.8],
            "well_2_max_posterior": [0.9],
            "well_1_integrated_posterior": [0.45],
            "well_2_integrated_posterior": [0.55],
        }
    )

    result = _add_ground_truth_metrics(
        comparison,
        decoded=pd.DataFrame(),
        gt_frame=pd.DataFrame(),
    )

    assert not bool(result.loc[0, "goal_correct"])
    assert bool(result.loc[0, "goal_correct_max_posterior"])
    assert bool(result.loc[0, "goal_correct_integrated"])
    assert result.loc[0, "endpoint_error_cm"] == pytest.approx(24.0)
    assert result.loc[0, "integrated_endpoint_error_cm"] == pytest.approx(13.5)
    assert result.loc[0, "true_well_posterior"] == pytest.approx(0.2)
    assert result.loc[0, "true_well_rank"] == 2
    assert result.loc[0, "true_well_max_posterior"] == pytest.approx(0.9)
    assert result.loc[0, "true_well_max_rank"] == 1
    assert result.loc[0, "true_well_integrated_posterior"] == pytest.approx(0.55)
    assert result.loc[0, "true_well_integrated_rank"] == 1


def _session(
    path: Path,
    *,
    position: np.ndarray,
    well_sequence: np.ndarray,
    ripple_events: np.ndarray,
) -> ReplaySession:
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=path,
        position=position,
        spikes=np.empty((0, 2)),
        tetrode_cell_ids=np.empty((0, 2)),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=ripple_events,
        run_times=np.array([[position[0, 0], position[-1, 0]]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=well_sequence,
        metadata={},
    )


def _write_minimal_session(session_path: Path) -> None:
    import scipy.io as sio

    times = np.linspace(0.0, 20.0, 401)
    x = np.where(times < 10.0, 10.0, 80.0)
    y = np.where(times < 10.0, 20.0, 90.0)
    position = np.column_stack([times, x, y, np.zeros_like(times)])
    sio.savemat(session_path / "Position_Data.mat", {"Position_Data": position})
    sio.savemat(
        session_path / "Ripple_Events.mat",
        {"Ripple_Events": np.array([[8.0, 8.1, 8.05, 1.0, 1.0, 1.0]])},
    )
    sio.savemat(
        session_path / "Spike_Data.mat",
        {
            "Spike_Data": np.array([[1.0, 1.0], [8.06, 1.0], [8.08, 2.0]]),
            "Tetrode_Cell_IDs": np.array([[1, 1], [1, 2]]),
            "Excitatory_Neurons": np.array([1, 2]),
            "Inhibitory_Neurons": np.array([]),
        },
    )
    sio.savemat(
        session_path / "Epochs.mat",
        {
            "Run_Times": np.array([0.0, 20.0]),
            "Sleep_Box_Immobile_Times": np.empty((0, 2)),
            "Sleep_Times": np.empty((0, 2)),
            "REM_Times": np.empty((0, 2)),
        },
    )
    sio.savemat(
        session_path / "Well_Sequence.mat",
        {"Well_Sequence": np.array([[0.0, 1.0], [10.0, 2.0], [20.0, 1.0]])},
    )
