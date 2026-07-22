from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.sign_flip_report import paired_sign_flip_test


def test_exact_sign_flip_handles_large_finite_deltas_without_overflow() -> None:
    result = paired_sign_flip_test([1e308, 1e308])

    assert np.isfinite(result.observed_mean)
    assert result.observed_mean == pytest.approx(1e308)
    assert result.method == "exact"
    assert result.permutations_evaluated == 4
    assert result.p_value == pytest.approx(0.5)
