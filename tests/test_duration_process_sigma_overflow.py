from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.duration_dynamics import _ps


def test_duration_process_sigma_rejects_finite_product_overflow() -> None:
    with np.errstate(over="raise", invalid="raise"), pytest.raises(
        ValueError,
        match="must produce a finite process sigma",
    ):
        _ps(np.finfo(float).max, 4.0)


def test_duration_process_sigma_preserves_large_finite_product() -> None:
    sigma_cm_sqrt_s = np.finfo(float).max / 4.0

    process_sigma = _ps(sigma_cm_sqrt_s, 4.0)

    assert np.isfinite(process_sigma)
    np.testing.assert_allclose(process_sigma, np.finfo(float).max / 2.0)
