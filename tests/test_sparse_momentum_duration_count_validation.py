from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.state_space_sparse_momentum as sparse_momentum


@pytest.mark.parametrize("n_time", [2.5, "2.5", np.float64(2.5), -1, np.nan])
def test_sparse_momentum_transition_duration_guard_rejects_invalid_time_counts(n_time: object) -> None:
    """Fractional, negative, and non-finite time-bin counts must not be truncated."""

    with pytest.raises(TypeError, match="integer count"):
        sparse_momentum._coerce_transition_durations([], n_time=n_time, fallback_dt=0.01)


def test_sparse_momentum_transition_duration_guard_accepts_integer_like_count() -> None:
    durations = sparse_momentum._coerce_transition_durations([], n_time=np.float64(3.0), fallback_dt=0.01)

    np.testing.assert_allclose(durations, np.array([0.01, 0.01], dtype=float))
