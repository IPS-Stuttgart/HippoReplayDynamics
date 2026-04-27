from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.ground_truth import (
    GroundTruthConfig,
    assign_endpoint_to_well,
    compare_scores_to_ground_truth,
    first_post_ripple_well_visit,
    infer_well_locations_from_arrays,
    label_session_behavioral_ground_truth,
    well_posterior_masses,
)


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
