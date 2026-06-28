from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _tensor_with_cell_ids(cell_ids: object) -> LogEmissionTensor:
    raw = np.asarray(cell_ids)
    n_cells = 1 if raw.ndim == 0 else int(raw.shape[0])
    return LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, n_cells), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=cell_ids,
        n_spikes=0,
    )


def test_log_emission_tensor_rejects_boolean_cell_ids() -> None:
    with pytest.raises(ValueError, match="cell_ids.*boolean"):
        _tensor_with_cell_ids(np.array([True], dtype=bool))


def test_log_emission_tensor_rejects_fractional_cell_ids() -> None:
    with pytest.raises(ValueError, match="cell_ids.*integer-valued"):
        _tensor_with_cell_ids(np.array([1.5], dtype=float))


def test_log_emission_tensor_rejects_duplicate_cell_ids() -> None:
    with pytest.raises(ValueError, match="cell_ids.*unique"):
        _tensor_with_cell_ids(np.array([1, 1], dtype=int))


def test_log_emission_tensor_canonicalizes_integral_cell_ids() -> None:
    emissions = _tensor_with_cell_ids(np.array(["2", "3"], dtype=object))

    assert np.issubdtype(emissions.cell_ids.dtype, np.integer)
    np.testing.assert_array_equal(emissions.cell_ids, np.array([2, 3], dtype=int))
