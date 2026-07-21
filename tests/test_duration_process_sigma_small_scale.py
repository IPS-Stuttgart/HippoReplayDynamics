from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.duration_dynamics import _ps


def test_duration_process_sigma_preserves_representable_sub_epsilon_product() -> None:
    process_sigma = np.finfo(float).eps / 16.0

    assert _ps(process_sigma, 1.0) == process_sigma


def test_duration_process_sigma_rejects_product_underflow_to_zero() -> None:
    smallest_positive = np.nextafter(0.0, 1.0)

    with pytest.raises(
        ValueError,
        match="must produce a finite process sigma",
    ):
        _ps(smallest_positive, smallest_positive)
