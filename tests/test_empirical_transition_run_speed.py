from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.empirical_transition import fit_empirical_transition_matrix
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def test_empirical_transition_speed_threshold_is_local_to_each_run_bout() -> None:
    session = ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [10.0, 2.0, 0.0],
                [11.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0], [10.0, 11.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    encoding = EncodingModel(
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

    transition = fit_empirical_transition_matrix(
        session,
        encoding,
        min_speed_cm_s=0.5,
        add_self_loop_count=0.0,
        teleport_probability=0.0,
    ).toarray()

    # Both bouts move at 1 cm/s. A global gradient sees the 9 s inter-bout gap
    # and incorrectly reduces boundary-frame speeds, dropping both valid edges.
    np.testing.assert_allclose(transition[:, 0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(transition[:, 2], np.array([0.0, 1.0, 0.0]))
