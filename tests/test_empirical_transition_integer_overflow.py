import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hipporeplayimm.empirical_transition import EmpiricalTransitionStateSpaceReplayModel
from hipporeplayimm.encoding import LogEmissionTensor


def test_empirical_transition_model_rejects_integer_overflow_in_column_sum() -> None:
    max_int = np.iinfo(np.int64).max
    transition = csr_matrix(
        np.array(
            [
                [max_int, 0, 0],
                [max_int, 1, 0],
                [3, 0, 1],
            ],
            dtype=np.int64,
        )
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 3), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )

    # In int64 arithmetic, the first column's mathematical sum is 2**64 + 1
    # but wraps to 1. The validator must not perform its stochasticity reduction
    # in the sparse matrix's integer dtype.
    with pytest.raises(ValueError, match="columns must sum to 1"):
        EmpiricalTransitionStateSpaceReplayModel(transition).score(
            emissions,
            bin_centers,
        )
