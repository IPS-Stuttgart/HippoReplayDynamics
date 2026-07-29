from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.sign_flip_report import paired_sign_flip_test


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_exact_n", np.array([2])),
        ("n_permutations", np.array([10])),
        ("random_seed", np.array([1])),
        ("chunk_size", np.array([4])),
    ],
)
def test_sign_flip_integer_controls_reject_non_scalar_arrays(
    parameter: str,
    value: np.ndarray,
) -> None:
    kwargs = {parameter: value}

    with pytest.raises(ValueError, match=rf"{parameter} must be"):
        paired_sign_flip_test([1.0, -0.5, 0.25], **kwargs)


@pytest.mark.parametrize("value", [np.array(2), np.int64(2), 2.0, Decimal("2")])
def test_sign_flip_integer_controls_accept_exact_numeric_scalars(value: object) -> None:
    result = paired_sign_flip_test(
        [1.0, -0.5, 0.25],
        max_exact_n=value,
        n_permutations=value,
        random_seed=value,
        chunk_size=value,
    )

    assert result.method == "monte_carlo"
    assert result.permutations_evaluated == 2
    assert result.random_seed == 2
