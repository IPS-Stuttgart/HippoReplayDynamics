import warnings
from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.empirical_transition import fit_empirical_transition_matrix
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _session(run_times: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=Path("."),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
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
        x_edges=np.array([-0.5, 0.5, 1.5, 2.5]),
        y_edges=np.array([-0.5, 0.5]),
        bin_centers=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        rates_hz=np.empty((0, 3), dtype=float),
        occupancy_s=np.ones(3, dtype=float),
        cell_ids=np.empty(0, dtype=int),
        config=EncodingConfig(min_speed_cm_s=0.0),
    )


@pytest.mark.parametrize(
    "run_times",
    [
        np.array([[0.0 + 1.0j, 2.0 + 2.0j]], dtype=np.complex128),
        np.array([[np.nan, 2.0]], dtype=float),
        np.array([[2.0, 1.0]], dtype=float),
        np.array([[False, True]], dtype=bool),
    ],
)
def test_fit_empirical_transition_rejects_invalid_run_intervals(
    run_times: np.ndarray,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="session.run_times"):
            fit_empirical_transition_matrix(
                _session(run_times),
                _encoding(),
                min_speed_cm_s=0.0,
                teleport_probability=0.0,
            )
