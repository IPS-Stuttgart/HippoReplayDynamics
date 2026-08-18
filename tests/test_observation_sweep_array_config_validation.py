import numpy as np
import pytest

from hipporeplayimm import apply_runtime_patches
from hipporeplayimm.observation_sweep import ObservationSweepConfig, observation_parameter_grid


def _nested_object_scalar(value: object, *, depth: int = 1) -> np.ndarray:
    current = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = current
        current = wrapper
    return current


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


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"bin_sizes_cm": (_nested_object_scalar(True, depth=2),)}, "bin_sizes_cm"),
        (
            {"smoothing_sigmas_bins": (_nested_object_scalar(np.bool_(False), depth=2),)},
            "smoothing_sigmas_bins",
        ),
        ({"decode_bin_s": _nested_object_scalar(True, depth=2)}, "decode_bin_s"),
        ({"n_folds": _nested_object_scalar(True, depth=2)}, "n_folds"),
        (
            {"simulation_events_per_model": _nested_object_scalar(np.bool_(True), depth=2)},
            "simulation_events_per_model",
        ),
    ],
)
def test_observation_parameter_grid_rejects_nested_array_wrapped_booleans(
    kwargs,
    field_name,
):
    apply_runtime_patches()
    config = ObservationSweepConfig(**kwargs)

    with pytest.raises(ValueError, match=field_name):
        observation_parameter_grid(config)


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        (
            {"bin_sizes_cm": (_nested_object_scalar(np.array([4.0])),)},
            "bin_sizes_cm",
        ),
        ({"n_folds": _nested_object_scalar(np.array([5]))}, "n_folds"),
    ],
)
def test_observation_parameter_grid_rejects_nested_non_scalar_arrays(
    kwargs,
    field_name,
):
    apply_runtime_patches()
    config = ObservationSweepConfig(**kwargs)

    with pytest.raises(ValueError, match=field_name):
        observation_parameter_grid(config)
