from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.empirical_transition import fit_empirical_transition_matrix
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def test_empirical_transition_does_not_bridge_tracking_gap_within_run_interval() -> None:
    session = ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 1.0, 0.0],
                [0.2, np.nan, np.nan],
                [3.0, 2.0, 0.0],
                [3.1, 1.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 3.1]], dtype=float),
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
        min_speed_cm_s=0.0,
        add_self_loop_count=0.0,
        teleport_probability=0.0,
    ).toarray()

    # Cleaning removes the invalid tracking row. The remaining 0.1 -> 3.0 s
    # samples must not be treated as a direct 1 -> 2 behavioral transition.
    np.testing.assert_allclose(transition[:, 1], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(transition[:, 0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(transition[:, 2], np.array([0.0, 1.0, 0.0]))
