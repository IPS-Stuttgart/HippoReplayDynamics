import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _boxed(value: object) -> np.ndarray:
    return np.array(value)


def _nested_scalar(value: object) -> np.ndarray:
    nested = np.empty((), dtype=object)
    nested[()] = _boxed(value)
    return nested


def _object_array(shape: tuple[int, ...], *values: object) -> np.ndarray:
    array = np.empty(shape, dtype=object)
    for index, value in zip(np.ndindex(shape), values, strict=True):
        array[index] = _boxed(value)
    return array


def _nested_object_array(shape: tuple[int, ...], *values: object) -> np.ndarray:
    array = np.empty(shape, dtype=object)
    for index, value in zip(np.ndindex(shape), values, strict=True):
        array[index] = _nested_scalar(value)
    return array


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
        (
            "log_likelihood",
            _object_array((2, 1), np.complex128(0.0 + 1.0j), np.float64(0.0)),
        ),
        (
            "spike_counts",
            _object_array((2, 1), np.complex128(0.0 + 1.0j), np.float64(0.0)),
        ),
        (
            "times",
            _object_array((2,), np.complex128(0.0 + 1.0j), np.float64(1.0)),
        ),
        ("dt", _nested_scalar(np.complex128(1.0 + 1.0j))),
        ("cell_ids", _object_array((1,), np.complex128(1.0 + 1.0j))),
        ("n_spikes", _nested_scalar(np.complex128(0.0 + 1.0j))),
        (
            "bin_durations",
            _object_array((2,), np.complex128(1.0 + 1.0j), np.float64(1.0)),
        ),
        (
            "transition_durations",
            _object_array((1,), np.complex128(1.0 + 1.0j)),
        ),
    ],
)
def test_log_emission_tensor_rejects_nested_complex_scalars(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*complex"):
        LogEmissionTensor(**_tensor_kwargs(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_likelihood", _object_array((2, 1), np.bool_(False), np.bool_(True))),
        ("spike_counts", _nested_object_array((2, 1), np.bool_(False), np.bool_(False))),
        ("times", _object_array((2,), np.bool_(False), np.bool_(True))),
        ("dt", _nested_scalar(np.bool_(True))),
        ("cell_ids", _nested_object_array((1,), np.bool_(True))),
        ("n_spikes", _nested_scalar(np.bool_(False))),
        ("bin_durations", _object_array((2,), np.bool_(True), np.bool_(True))),
        ("transition_durations", _object_array((1,), np.bool_(True))),
    ],
)
def test_log_emission_tensor_rejects_nested_boolean_scalars(field, value):
    with pytest.raises(ValueError, match=rf"{field}.*boolean"):
        LogEmissionTensor(**_tensor_kwargs(**{field: value}))


def test_log_emission_tensor_accepts_nested_real_scalars():
    emissions = LogEmissionTensor(
        **_tensor_kwargs(
            log_likelihood=_object_array((2, 1), np.float64(0.0), np.float64(0.0)),
            spike_counts=_object_array((2, 1), np.float64(0.0), np.float64(0.0)),
            times=_object_array((2,), np.float64(0.0), np.float64(1.0)),
            dt=_nested_scalar(np.float64(1.0)),
            cell_ids=_object_array((1,), np.float64(1.0)),
            n_spikes=_nested_scalar(np.float64(0.0)),
            bin_durations=_object_array((2,), np.float64(1.0), np.float64(1.0)),
            transition_durations=_object_array((1,), np.float64(1.0)),
        )
    )

    np.testing.assert_allclose(emissions.log_likelihood, np.zeros((2, 1)))
    np.testing.assert_array_equal(emissions.spike_counts, np.zeros((2, 1), dtype=int))
    np.testing.assert_allclose(emissions.times, np.array([0.0, 1.0]))
    assert emissions.dt == 1.0
    np.testing.assert_array_equal(emissions.cell_ids, np.array([1]))
    assert emissions.n_spikes == 0
