from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.accuracy_replay_gain_gamma_patch import _coerce_spike_counts
from hipporeplayimm.accuracy_upgrades import gamma_poisson_predictive_log_emissions


_EXTRA_LONGDOUBLE_PRECISION = np.finfo(np.longdouble).nmant > np.finfo(float).nmant


def _gamma_prior() -> tuple[np.ndarray, np.ndarray]:
    shape = np.ones((1, 1), dtype=float)
    exposure = np.ones((1, 1), dtype=float)
    return shape, exposure


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1e-10, id="near-zero"),
        pytest.param(1.0 + 5e-10, id="near-one"),
        pytest.param(
            Decimal("9007199254740993.5"),
            id="decimal-above-binary64-precision",
        ),
        pytest.param(
            np.longdouble("9007199254740993.5"),
            marks=pytest.mark.skipif(
                not _EXTRA_LONGDOUBLE_PRECISION,
                reason="longdouble has no precision beyond binary64",
            ),
            id="longdouble-above-binary64-precision",
        ),
    ],
)
def test_gamma_poisson_rejects_fractional_spike_counts_exactly(value: object) -> None:
    shape, exposure = _gamma_prior()

    with pytest.raises(ValueError, match="spike_counts.*integer counts"):
        gamma_poisson_predictive_log_emissions(
            np.array([[value]], dtype=object),
            shape,
            exposure,
            0.02,
        )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(2**53 + 1, id="python-integer"),
        pytest.param(Decimal("9007199254740993"), id="decimal-integer"),
        pytest.param("9007199254740993.0", id="decimal-text"),
        pytest.param(b"9007199254740993", id="integer-bytes"),
    ],
)
def test_gamma_poisson_count_coercion_preserves_exact_large_integers(value: object) -> None:
    counts = _coerce_spike_counts(np.array([[value]], dtype=object))

    assert counts.dtype == np.dtype(int)
    assert int(counts[0, 0]) == 2**53 + 1


def test_gamma_poisson_count_coercion_rejects_platform_integer_overflow() -> None:
    value = int(np.iinfo(np.dtype(int)).max) + 1

    with pytest.raises(ValueError, match="spike_counts.*integer count range"):
        _coerce_spike_counts(np.array([[value]], dtype=object))


@pytest.mark.parametrize("value", [1.0 + 0.0j, 1.0 + 2.0j])
def test_gamma_poisson_rejects_complex_spike_counts(value: complex) -> None:
    shape, exposure = _gamma_prior()

    with pytest.raises(ValueError, match="spike_counts.*real integer counts"):
        gamma_poisson_predictive_log_emissions(
            np.array([[value]], dtype=complex),
            shape,
            exposure,
            0.02,
        )


@pytest.mark.parametrize("dtype", [int, float])
def test_gamma_poisson_count_coercion_keeps_regular_numeric_arrays(dtype: type) -> None:
    counts = _coerce_spike_counts(np.array([[0, 1], [2, 3]], dtype=dtype))

    np.testing.assert_array_equal(counts, np.array([[0, 1], [2, 3]], dtype=int))
    assert counts.dtype == np.dtype(int)
