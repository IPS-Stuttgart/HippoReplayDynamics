from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.replay_emission_calibration import apply_replay_cell_gains


def test_apply_replay_cell_gains_rejects_finite_rate_product_overflow() -> None:
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[np.finfo(float).max]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )

    with pytest.raises(
        ValueError,
        match="replay gain scaling must produce finite rates",
    ):
        apply_replay_cell_gains(encoding, np.array([2.0], dtype=float))
