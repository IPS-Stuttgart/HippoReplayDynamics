import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hipporeplayimm.empirical_transition import EmpiricalTransitionStateSpaceReplayModel
from hipporeplayimm.encoding import LogEmissionTensor


def test_empirical_transition_model_rejects_complex_transition_entries() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.empty((2, 0), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([], dtype=int),
        n_spikes=0,
    )
    transition = csr_matrix(
        np.array(
            [
                [1.0 + 2.0j, 0.0],
                [0.0, 1.0 + 0.0j],
            ],
            dtype=complex,
        )
    )

    with pytest.raises(ValueError, match="transition matrix entries must be real"):
        EmpiricalTransitionStateSpaceReplayModel(transition).score(
            emissions,
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        )
