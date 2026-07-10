from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.observation_sweep import (
    ObservationSweepConfig,
    observation_parameter_grid,
)


@pytest.mark.parametrize(
    ("config", "field_name"),
    [
        (ObservationSweepConfig(bin_sizes_cm="12"), "bin_sizes_cm"),
        (ObservationSweepConfig(bin_sizes_cm=b"12"), "bin_sizes_cm"),
        (ObservationSweepConfig(bin_sizes_cm=("12",)), "bin_sizes_cm"),
        (ObservationSweepConfig(decode_bin_s="0.02"), "decode_bin_s"),
        (ObservationSweepConfig(n_folds="5"), "n_folds"),
        (
            ObservationSweepConfig(simulation_events_per_model=np.str_("10")),
            "simulation_events_per_model",
        ),
    ],
)
def test_observation_parameter_grid_rejects_text_numeric_config(
    config: ObservationSweepConfig,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        observation_parameter_grid(config)
