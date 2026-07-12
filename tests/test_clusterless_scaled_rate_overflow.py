from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import clusterless_mark_group_validation


def test_scaled_log_rates_rejects_finite_product_overflow() -> None:
    with pytest.raises(
        ValueError,
        match="clusterless rate_hz becomes non-finite after spike-rate scaling",
    ):
        clusterless_mark_group_validation._scaled_log_rates(
            np.array([np.finfo(float).max]),
            2.0,
            "clusterless rate_hz",
        )


def test_scaled_log_rates_preserves_exact_zero_support() -> None:
    scaled, log_rates = clusterless_mark_group_validation._scaled_log_rates(
        np.array([0.0, 2.0]),
        3.0,
        "clusterless rate_hz",
    )

    np.testing.assert_array_equal(scaled, np.array([0.0, 6.0]))
    assert np.isneginf(log_rates[0])
    np.testing.assert_allclose(log_rates[1], np.log(6.0))
