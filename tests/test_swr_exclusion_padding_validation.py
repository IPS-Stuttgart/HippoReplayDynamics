from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spike_matched_event_window_null import (  # noqa: E402
    _padded_intervals,
    spike_matched_null_windows,
)


_ERROR = "padding_s must be finite and nonnegative"


def _minimal_matched_null_session() -> SimpleNamespace:
    event = SimpleNamespace(start=1.0, end=1.1)
    return SimpleNamespace(
        ripple=lambda _event_index: event,
        excitatory_spikes=lambda: np.empty((0, 2), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        ripple_events=np.array([[1.0, 1.1]], dtype=float),
    )


@pytest.mark.parametrize("padding_s", [-0.04, -0.06, np.nan, np.inf, -np.inf])
def test_matched_null_windows_reject_invalid_swr_exclusion_padding(padding_s: float) -> None:
    with pytest.raises(ValueError, match=_ERROR):
        spike_matched_null_windows(
            _minimal_matched_null_session(),
            0,
            nulls_per_event=1,
            random_seed=1,
            candidate_step_s=0.1,
            exclusion_padding_s=padding_s,
        )


@pytest.mark.parametrize("padding_s", [-0.01, np.nan, np.inf, -np.inf])
def test_padding_validation_precedes_empty_interval_return(padding_s: float) -> None:
    with pytest.raises(ValueError, match=_ERROR):
        _padded_intervals(np.empty((0, 2), dtype=float), padding_s=padding_s)


def test_padded_intervals_preserve_zero_and_positive_padding() -> None:
    intervals = np.array([[1.0, 1.1]], dtype=float)

    np.testing.assert_allclose(_padded_intervals(intervals, padding_s=0.0), intervals)
    np.testing.assert_allclose(
        _padded_intervals(intervals, padding_s=0.05),
        np.array([[0.95, 1.15]], dtype=float),
    )
    np.testing.assert_allclose(intervals, np.array([[1.0, 1.1]], dtype=float))
