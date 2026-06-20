from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.empirical_transition import fit_empirical_transition_matrix
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def test_fit_empirical_transition_matrix_rejects_invalid_self_loop_count() -> None:
    session = _minimal_session()
    encoding = _minimal_encoding()

    for add_self_loop_count in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="add_self_loop_count"):
            fit_empirical_transition_matrix(
                session,
                encoding,
                add_self_loop_count=add_self_loop_count,
            )


def test_fit_empirical_transition_matrix_rejects_invalid_min_speed_override() -> None:
    session = _minimal_session()
    encoding = _minimal_encoding()

    for min_speed_cm_s in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_speed_cm_s"):
            fit_empirical_transition_matrix(
                session,
                encoding,
                min_speed_cm_s=min_speed_cm_s,
            )


def test_fit_empirical_transition_matrix_is_column_stochastic_for_valid_inputs() -> None:
    transition = fit_empirical_transition_matrix(
        _minimal_session(),
        _minimal_encoding(),
        add_self_loop_count=0.5,
        teleport_probability=0.01,
        min_speed_cm_s=0.0,
    ).toarray()

    assert np.all(transition >= 0.0)
    np.testing.assert_allclose(transition.sum(axis=0), np.ones(transition.shape[1]))


def _minimal_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _minimal_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([-0.5, 0.5, 1.5, 2.5], dtype=float),
        y_edges=np.array([-0.5, 0.5], dtype=float),
        bin_centers=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=float,
        ),
        rates_hz=np.empty((0, 3), dtype=float),
        occupancy_s=np.ones(3, dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(min_speed_cm_s=0.0),
    )
