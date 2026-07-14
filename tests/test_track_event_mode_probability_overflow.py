import numpy as np
import pytest

from scripts.track_event import _mode_probability_row


def test_mode_probability_row_normalizes_large_finite_weights_without_overflow():
    maximum = np.finfo(float).max

    row = _mode_probability_row(
        ("stationary", "diffusion"),
        np.array([maximum, maximum / 2.0]),
    )

    assert row["mode_stationary_probability"] == pytest.approx(2.0 / 3.0)
    assert row["mode_diffusion_probability"] == pytest.approx(1.0 / 3.0)
    assert row["mode_stationary_probability"] + row["mode_diffusion_probability"] == pytest.approx(1.0)
    assert row["most_likely_mode"] == "stationary"
