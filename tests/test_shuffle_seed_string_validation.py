from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.result_improvements import shuffle_spike_times_session


def _empty_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


@pytest.mark.parametrize("seed", ["4", np.str_("4")])
def test_shuffle_spike_times_rejects_string_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed.*string"):
        shuffle_spike_times_session(_empty_session(), random_seed=seed)
