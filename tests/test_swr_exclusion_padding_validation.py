from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spike_matched_event_window_null import (  # noqa: E402
    _padded_intervals,
    spike_matched_null_windows,
)


class _MatchedNullSession:
    ripple_events = np.array([[1.0, 1.1]], dtype=float)
    run_times = np.array([[0.0, 2.0]], dtype=float)
    position = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]], dtype=float)
    spikes = np.array([[1.01, 1.0], [1.05, 2.0]], dtype=float)

    def ripple(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(start=1.0, end=1.1)

    def excitatory_spikes(self) -> np.ndarray:
        return self.spikes


@pytest.mark.parametrize("padding_s", [-0.01, -np.inf, np.inf, np.nan])
@pytest.mark.parametrize(
    "intervals",
    [np.array([[1.0, 1.1]], dtype=float), np.empty((0, 2), dtype=float)],
)
def test_padded_intervals_reject_invalid_padding_before_empty_shortcut(
    padding_s: float,
    intervals: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _padded_intervals(intervals, padding_s=padding_s)


@pytest.mark.parametrize("padding_s", [-0.04, -np.inf, np.inf, np.nan])
def test_spike_matched_null_windows_reject_invalid_exclusion_padding(
    padding_s: float,
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        spike_matched_null_windows(
            _MatchedNullSession(),
            0,
            nulls_per_event=1,
            random_seed=1,
            candidate_step_s=0.1,
            exclusion_padding_s=padding_s,
        )


def test_padded_intervals_preserve_zero_and_positive_padding() -> None:
    intervals = np.array([[1.0, 1.1]], dtype=float)

    np.testing.assert_allclose(
        _padded_intervals(intervals, padding_s=0.0),
        intervals,
    )
    np.testing.assert_allclose(
        _padded_intervals(intervals, padding_s=0.2),
        np.array([[0.8, 1.3]], dtype=float),
    )
