from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.simulation_recovery_count_validation import _validated_count_matrix


@pytest.mark.parametrize(
    "counts",
    [
        np.array([[2**53 + 1]], dtype=np.int64),
        np.array([[2**53 + 1]], dtype=object),
        np.array([[Decimal("9007199254740993")]], dtype=object),
    ],
)
def test_validated_count_matrix_preserves_large_exact_integers(counts: np.ndarray) -> None:
    validated = _validated_count_matrix(counts, n_cells=1)

    assert validated.dtype == np.dtype(int)
    assert int(validated[0, 0]) == 2**53 + 1


def test_validated_count_matrix_rejects_fractional_decimal_above_float_precision() -> None:
    counts = np.array([[Decimal("9007199254740993.5")]], dtype=object)

    with pytest.raises(ValueError, match="integer-valued counts"):
        _validated_count_matrix(counts, n_cells=1)


def test_validated_count_matrix_rejects_unsigned_values_outside_integer_range() -> None:
    counts = np.array([[np.uint64(np.iinfo(np.dtype(int)).max) + np.uint64(1)]], dtype=np.uint64)

    with pytest.raises(ValueError, match="integer count range"):
        _validated_count_matrix(counts, n_cells=1)
