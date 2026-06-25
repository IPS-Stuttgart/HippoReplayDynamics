from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.duration_occupancy_metadata_guard import _coerce_transition_durations


def test_transition_duration_guard_rejects_mismatched_explicit_metadata() -> None:
    """Explicit duration metadata must not be replaced by fallback dt on shape mismatch."""

    with pytest.raises(ValueError, match="one finite positive value per transition"):
        _coerce_transition_durations([0.02], n_time=3, fallback_dt=0.01)


def test_transition_duration_guard_allows_empty_metadata_fallback() -> None:
    durations = _coerce_transition_durations([], n_time=3, fallback_dt=0.01)

    np.testing.assert_allclose(durations, np.array([0.01, 0.01], dtype=float))


def test_transition_duration_guard_rejects_nonfinite_explicit_metadata() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _coerce_transition_durations([0.02, np.nan], n_time=3, fallback_dt=0.01)
