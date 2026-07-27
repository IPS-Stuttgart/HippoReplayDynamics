from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


@pytest.mark.parametrize(
    "cell_ids",
    [
        pytest.param(
            [str(2**53), str(2**53 + 1)],
            id="integer-text",
        ),
        pytest.param(
            [f"{2**53}.0", f"{2**53 + 1}.0"],
            id="decimal-text",
        ),
        pytest.param(
            [str(2**53).encode(), str(2**53 + 1).encode()],
            id="integer-bytes",
        ),
        pytest.param(
            [Decimal(2**53), Decimal(2**53 + 1)],
            id="decimal-objects",
        ),
    ],
)
def test_log_emission_tensor_preserves_exact_large_cell_ids(cell_ids: list[object]) -> None:
    tensor = LogEmissionTensor(
        log_likelihood=np.zeros((1, 1), dtype=float),
        spike_counts=np.zeros((1, 2), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.asarray(cell_ids, dtype=object),
        n_spikes=0,
    )

    assert tensor.cell_ids.tolist() == [2**53, 2**53 + 1]
