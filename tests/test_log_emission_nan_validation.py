from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _valid_tensor_kwargs() -> dict[str, object]:
    return {
        "log_likelihood": np.array([[0.0, -1.0], [-0.5, -0.25]], dtype=float),
        "spike_counts": np.zeros((2, 1), dtype=int),
        "times": np.array([0.0, 0.02], dtype=float),
        "dt": 0.02,
        "cell_ids": np.array([1], dtype=int),
        "n_spikes": 0,
    }


def test_log_emission_tensor_rejects_nan_log_likelihood() -> None:
    kwargs = _valid_tensor_kwargs()
    kwargs["log_likelihood"] = np.array([[0.0, np.nan]], dtype=float)
    kwargs["spike_counts"] = np.zeros((1, 1), dtype=int)
    kwargs["times"] = np.array([0.0], dtype=float)

    with pytest.raises(ValueError, match="log_likelihood.*NaN"):
        LogEmissionTensor(**kwargs)


@pytest.mark.parametrize(
    "spike_counts, expected_error",
    [
        (np.array([[0.5], [1.0]], dtype=float), "integer-valued counts"),
        (np.array([[True], [False]], dtype=bool), "boolean values"),
        (np.array([[0.0], [np.inf]], dtype=float), "finite nonnegative"),
        (np.array([[0.0], [-1.0]], dtype=float), "finite nonnegative"),
    ],
)
def test_log_emission_tensor_rejects_invalid_direct_spike_counts(
    spike_counts: np.ndarray,
    expected_error: str,
) -> None:
    kwargs = _valid_tensor_kwargs()
    kwargs["spike_counts"] = spike_counts
    numeric_counts = np.asarray(spike_counts, dtype=float)
    kwargs["n_spikes"] = (
        int(np.sum(numeric_counts)) if np.all(np.isfinite(numeric_counts)) else 0
    )

    with pytest.raises(ValueError, match=expected_error):
        LogEmissionTensor(**kwargs)
