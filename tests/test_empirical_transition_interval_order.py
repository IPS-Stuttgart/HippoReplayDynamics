from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.empirical_transition import fit_empirical_transition_matrix
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def test_empirical_transition_is_invariant_to_shared_endpoint_interval_order() -> None:
    intervals = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)

    forward = fit_empirical_transition_matrix(
        _session(intervals),
        _encoding(),
        min_speed_cm_s=5.0,
        add_self_loop_count=0.0,
        teleport_probability=0.0,
    ).toarray()
    reversed_order = fit_empirical_transition_matrix(
        _session(intervals[::-1]),
        _encoding(),
        min_speed_cm_s=5.0,
        add_self_loop_count=0.0,
        teleport_probability=0.0,
    ).toarray()

    np.testing.assert_allclose(forward, reversed_order)
    np.testing.assert_allclose(forward[:, 0], np.array([0.0, 1.0]))


def _session(run_times: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 10.0, 0.0],
                [2.0, 10.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=run_times,
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([-1.0, 5.0, 15.0], dtype=float),
        y_edges=np.array([-1.0, 1.0], dtype=float),
        bin_centers=np.array([[2.0, 0.0], [10.0, 0.0]], dtype=float),
        rates_hz=np.empty((0, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(min_speed_cm_s=0.0),
    )
