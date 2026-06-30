import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm.observation_sweep import ObservationSweepConfig, observation_parameter_grid


@pytest.mark.parametrize(
    ("kwargs", "field_name", "expected"),
    [
        ({"bin_sizes_cm": (np.array([4.0]),)}, "bin_sizes_cm", "finite scalars"),
        (
            {"likelihood_temperatures": (np.array([[1.0]]),)},
            "likelihood_temperatures",
            "finite scalars",
        ),
        ({"decode_bin_s": np.array([0.02])}, "decode_bin_s", "finite scalars"),
        ({"n_folds": np.array([5])}, "n_folds", "positive integer"),
        (
            {"simulation_events_per_model": np.array([[10]])},
            "simulation_events_per_model",
            "positive integer",
        ),
    ],
)
def test_observation_parameter_grid_rejects_array_shaped_config_values(
    kwargs,
    field_name,
    expected,
):
    apply_runtime_patches()
    config = ObservationSweepConfig(**kwargs)

    with pytest.raises(ValueError) as exc_info:
        observation_parameter_grid(config)

    message = str(exc_info.value)
    assert field_name in message
    assert expected in message


def test_observation_parameter_grid_accepts_numpy_scalar_config_values():
    apply_runtime_patches()
    config = ObservationSweepConfig(
        bin_sizes_cm=(np.array(4.0),),
        decode_bin_s=np.array(0.02),
        n_folds=np.array(5),
        simulation_events_per_model=np.int64(10),
    )

    rows = observation_parameter_grid(config)

    assert len(rows) == 1
    assert rows[0]["bin_size_cm"] == 4.0
