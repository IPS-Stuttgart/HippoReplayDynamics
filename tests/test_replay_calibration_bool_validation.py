from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)


@pytest.mark.parametrize(
    "max_gain",
    [
        True,
        False,
        np.bool_(True),
        np.asarray(True, dtype=object),
    ],
)
def test_replay_calibration_rejects_boolean_max_gain(max_gain: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="max_gain"):
        build_sorted_emissions_with_replay_calibration(
            object(),
            object(),
            object(),
            calibration=ReplayEmissionCalibration(max_gain=max_gain),
        )
