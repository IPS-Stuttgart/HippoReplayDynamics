import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


_HUGE_INTEGER = 10**1000


def _valid_tensor_kwargs() -> dict[str, object]:
    return {
        "log_likelihood": np.zeros((2, 1), dtype=float),
        "spike_counts": np.zeros((2, 1), dtype=int),
        "times": np.array([0.0, 0.02]),
        "dt": 0.02,
        "cell_ids": np.array([1]),
        "n_spikes": 0,
        "bin_durations": np.array([0.02, 0.02]),
        "transition_durations": np.array([0.02]),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_likelihood", [[_HUGE_INTEGER], [0.0]]),
        ("spike_counts", [[_HUGE_INTEGER], [0]]),
        ("times", [_HUGE_INTEGER, 0.02]),
        ("dt", _HUGE_INTEGER),
        ("bin_durations", [_HUGE_INTEGER, 0.02]),
        ("transition_durations", [_HUGE_INTEGER]),
    ],
)
def test_log_emission_tensor_normalizes_numeric_overflow(field, value):
    kwargs = _valid_tensor_kwargs()
    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match=rf"{field} must contain values representable as floating point",
    ):
        LogEmissionTensor(**kwargs)
