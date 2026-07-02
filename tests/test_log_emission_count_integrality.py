from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _valid_tensor_kwargs(n_time: int = 1) -> dict[str, object]:
    return {
        "log_likelihood": np.zeros((n_time, 1), dtype=float),
        "spike_counts": np.zeros((n_time, 1), dtype=int),
        "times": np.arange(n_time, dtype=float) * 0.02,
        "dt": 0.02,
        "cell_ids": np.array([1], dtype=int),
        "n_spikes": 0,
    }


def test_log_emission_tensor_rejects_fractional_spike_counts_with_integer_total() -> None:
    with pytest.raises(ValueError, match="spike_counts.*integer-valued"):
        LogEmissionTensor(
            log_likelihood=np.zeros((2, 1), dtype=float),
            spike_counts=np.array([[0.5], [0.5]], dtype=float),
            times=np.array([0.0, 0.02], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=1,
        )


def test_log_emission_tensor_rejects_boolean_spike_counts() -> None:
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        LogEmissionTensor(
            log_likelihood=np.zeros((1, 1), dtype=float),
            spike_counts=np.array([[True]], dtype=bool),
            times=np.array([0.0], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=1,
        )


def test_log_emission_tensor_rejects_boolean_n_spikes() -> None:
    with pytest.raises(ValueError, match="n_spikes.*boolean"):
        LogEmissionTensor(
            log_likelihood=np.zeros((1, 1), dtype=float),
            spike_counts=np.array([[1]], dtype=int),
            times=np.array([0.0], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=True,
        )


@pytest.mark.parametrize("n_spikes", [np.array([1]), np.array([[1]])])
def test_log_emission_tensor_rejects_array_shaped_n_spikes(n_spikes: object) -> None:
    with pytest.raises(ValueError, match="n_spikes.*scalar"):
        LogEmissionTensor(
            log_likelihood=np.zeros((1, 1), dtype=float),
            spike_counts=np.array([[1]], dtype=int),
            times=np.array([0.0], dtype=float),
            dt=0.02,
            cell_ids=np.array([1], dtype=int),
            n_spikes=n_spikes,
        )


@pytest.mark.parametrize("bad_dt", [True, np.bool_(False), np.array(True, dtype=object)])
def test_log_emission_tensor_rejects_boolean_dt_duration(bad_dt: object) -> None:
    kwargs = _valid_tensor_kwargs()
    kwargs["dt"] = bad_dt

    with pytest.raises(ValueError, match="dt.*boolean"):
        LogEmissionTensor(**kwargs)


def test_log_emission_tensor_rejects_array_shaped_dt_duration() -> None:
    kwargs = _valid_tensor_kwargs()
    kwargs["dt"] = np.array([0.02], dtype=float)

    with pytest.raises(ValueError, match="dt.*scalar"):
        LogEmissionTensor(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("bin_durations", np.array([True, False], dtype=bool), "bin_durations.*boolean"),
        ("transition_durations", np.array([True], dtype=bool), "transition_durations.*boolean"),
    ],
)
def test_log_emission_tensor_rejects_boolean_duration_arrays(field: str, value: object, match: str) -> None:
    kwargs = _valid_tensor_kwargs(n_time=2)
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        LogEmissionTensor(**kwargs)
