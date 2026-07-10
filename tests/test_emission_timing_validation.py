import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _tensor_kwargs(**overrides):
    kwargs = {
        "log_likelihood": np.zeros((2, 1), dtype=float),
        "spike_counts": np.zeros((2, 1), dtype=int),
        "times": np.array([0.0, 1.0], dtype=float),
        "dt": 1.0,
        "cell_ids": np.array([1]),
        "n_spikes": 0,
        "bin_durations": np.ones(2, dtype=float),
        "transition_durations": np.ones(1, dtype=float),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("times", np.array([False, True])),
        ("dt", True),
        ("bin_durations", np.array([True, True])),
        ("transition_durations", np.array([True])),
    ],
)
def test_log_emission_tensor_rejects_boolean_timing_metadata(field, value):
    with pytest.raises(ValueError, match=field):
        LogEmissionTensor(**_tensor_kwargs(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_likelihood", np.array([[0.0 + 1.0j], [0.0 + 0.0j]])),
        ("spike_counts", np.array([[0.0 + 1.0j], [0.0 + 0.0j]])),
        ("times", np.array([0.0 + 1.0j, 1.0 + 0.0j])),
        ("dt", np.complex128(1.0 + 1.0j)),
        ("cell_ids", np.array([1.0 + 1.0j])),
        ("n_spikes", np.complex128(0.0 + 1.0j)),
        ("bin_durations", np.array([1.0 + 1.0j, 1.0 + 0.0j])),
        ("transition_durations", np.array([1.0 + 1.0j])),
    ],
)
def test_log_emission_tensor_rejects_complex_metadata_before_float_coercion(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*complex"):
        LogEmissionTensor(**_tensor_kwargs(**{field: value}))


def test_log_emission_tensor_accepts_numeric_timing_metadata():
    emissions = LogEmissionTensor(**_tensor_kwargs())

    assert emissions.dt == 1.0
    np.testing.assert_allclose(emissions.bin_durations, np.ones(2))
    np.testing.assert_allclose(emissions.transition_durations, np.ones(1))
