from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.duration_occupancy_metadata_guard import _coerce_transition_durations


def test_transition_duration_guard_uses_fallback_only_for_missing_metadata():
    durations = _coerce_transition_durations([], n_time=3, fallback_dt=0.05)

    np.testing.assert_allclose(durations, np.array([0.05, 0.05], dtype=float))


def test_transition_duration_guard_rejects_wrong_length_explicit_metadata():
    with pytest.raises(ValueError, match="one finite positive value per transition"):
        _coerce_transition_durations([0.05], n_time=3, fallback_dt=0.02)


def test_transition_duration_guard_rejects_nonfinite_explicit_metadata():
    with pytest.raises(ValueError, match="finite and positive"):
        _coerce_transition_durations([0.05, float("nan")], n_time=3, fallback_dt=0.02)
