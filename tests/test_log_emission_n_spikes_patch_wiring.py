import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import LogEmissionTensor


def test_package_runtime_patches_install_log_emission_n_spikes_validation() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="n_spikes must equal"):
        LogEmissionTensor(
            log_likelihood=np.zeros((1, 2), dtype=float),
            spike_counts=np.array([[1, 0]], dtype=int),
            times=np.array([0.0], dtype=float),
            dt=0.01,
            cell_ids=np.array([11, 12], dtype=int),
            n_spikes=0,
        )
