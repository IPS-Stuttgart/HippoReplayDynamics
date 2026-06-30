import numpy as np
import pytest

from hipporeplayimm.simulation_recovery_count_validation import _validated_count_matrix


def test_validated_count_matrix_rejects_counts_outside_integer_range():
    oversized = np.array([[np.nextafter(float(np.iinfo(np.dtype(int)).max), np.inf)]])

    with pytest.raises(ValueError, match="integer count range"):
        _validated_count_matrix(oversized, n_cells=1)


def test_validated_count_matrix_accepts_integer_valued_float_counts():
    counts = _validated_count_matrix(
        np.array([[0.0, 2.0], [3.0, 4.0]]),
        n_cells=2,
    )

    np.testing.assert_array_equal(
        counts,
        np.array([[0, 2], [3, 4]], dtype=int),
    )
