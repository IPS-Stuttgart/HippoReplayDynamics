from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession, _as_two_dimensional
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, fit_place_field_encoding
from hipporeplayimm.models import CandidateKinematicModel
from hipporeplayimm.state_space_utils import _gaussian_transition_matrix


def _minimal_session(position: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.asarray(position, dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def test_fit_place_field_encoding_rejects_malformed_position_array() -> None:
    session = _minimal_session(np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="position"):
        fit_place_field_encoding(session)


def test_fit_place_field_encoding_rejects_nonpositive_bin_size() -> None:
    session = _minimal_session(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
    )

    with pytest.raises(ValueError, match="bin_size_cm"):
        fit_place_field_encoding(session, EncodingConfig(bin_size_cm=0.0))


def test_encoding_model_positions_to_flat_bins_rejects_bad_shape() -> None:
    model = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.empty((0, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )

    with pytest.raises(ValueError, match="xy"):
        model.positions_to_flat_bins(np.array([0.5, 0.5]))


def test_empty_ripple_events_keep_six_column_schema() -> None:
    ripple_events = _as_two_dimensional(np.array([]), "Ripple_Events")

    assert ripple_events.shape == (0, 6)


def test_candidate_kinematic_model_rejects_negative_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        CandidateKinematicModel(mode="imm", top_k=-1)


def test_state_space_gaussian_transition_rejects_nonfinite_parameters() -> None:
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="sigma_cm"):
        _gaussian_transition_matrix(bin_centers, float("nan"), 4.0)

    with pytest.raises(ValueError, match="max_step_sigma"):
        _gaussian_transition_matrix(bin_centers, 1.0, float("nan"))
