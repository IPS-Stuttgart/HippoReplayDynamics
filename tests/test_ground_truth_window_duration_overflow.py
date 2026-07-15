from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.ground_truth_window_scope import _window_from_score_rows


def test_window_from_score_rows_rejects_overflowing_duration() -> None:
    limit = np.finfo(float).max
    scores = pd.DataFrame(
        {
            "window_start_s": [-limit],
            "window_end_s": [limit],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"window_end_s - window_start_s must be finite",
    ):
        _window_from_score_rows(scores)


def test_window_from_score_rows_keeps_large_representable_duration() -> None:
    limit = np.finfo(float).max / 4.0
    scores = pd.DataFrame(
        {
            "window_start_s": [-limit],
            "window_end_s": [limit],
        }
    )

    window = _window_from_score_rows(scores)

    assert window is not None
    assert np.isfinite(window.peak)
    assert window.peak == 0.0
