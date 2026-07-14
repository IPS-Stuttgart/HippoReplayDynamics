from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.simulation_recovery_count_validation import _validated_count_matrix


@pytest.mark.parametrize(
    "value",
    [
        2**53 + 1,
        np.int64(2**53 + 1),
        Decimal("9007199254740993"),
    ],
)
def test_validated_count_matrix_rejects_float_precision_loss(value: object) -> None:
    with pytest.raises(ValueError, match="exactly representable"):
        _validated_count_matrix(np.array([[value]], dtype=object), n_cells=1)


@pytest.mark.parametrize(
    "value",
    [
        2**53,
        2**53 + 2,
        np.int64(2**53 + 2),
        Decimal("9007199254740994"),
    ],
)
def test_validated_count_matrix_preserves_exactly_representable_large_counts(
    value: object,
) -> None:
    counts = _validated_count_matrix(np.array([[value]], dtype=object), n_cells=1)

    assert counts.dtype == np.dtype(int)
    assert counts.tolist() == [[int(value)]]
