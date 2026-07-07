from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.models import RandomModel


def _base_tensor_kwargs(**overrides):
    log_likelihood = np.asarray(overrides.pop("log_likelihood", np.zeros((1, 1), dtype=float)))
    spike_counts = np.asarray(overrides.pop("spike_counts", np.zeros((log_likelihood.shape[0], 1), dtype=int)))
    n_cells = spike_counts.shape[1] if spike_counts.ndim == 2 else 1
    kwargs = {
        "log_likelihood": log_likelihood,
        "spike_counts": spike_counts,
        "times": np.arange(log_likelihood.shape[0], dtype=float) * 0.1,
        "dt": 0.1,
        "cell_ids": np.arange(1, n_cells + 1, dtype=int),
        "n_spikes": 0,
    }
    kwargs.update(overrides)
    return kwargs


def _tensor_with_log_likelihood(log_likelihood: np.ndarray) -> LogEmissionTensor:
    return LogEmissionTensor(**_base_tensor_kwargs(log_likelihood=log_likelihood))


def test_log_emission_tensor_rejects_nan_likelihood_entries_at_construction():
    with pytest.raises(ValueError, match="NaN"):
        _tensor_with_log_likelihood(np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float))


def test_log_emission_tensor_rejects_text_likelihood_entries_at_construction():
    with pytest.raises(ValueError, match="log_likelihood.*text"):
        LogEmissionTensor(**_base_tensor_kwargs(log_likelihood=np.array([["0.0"]], dtype=object)))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dt": "0.1"}, "dt.*text"),
        ({"bin_durations": np.array(["0.1"], dtype=object)}, "bin_durations.*text"),
        (
            {
                "log_likelihood": np.zeros((2, 1), dtype=float),
                "spike_counts": np.zeros((2, 1), dtype=int),
                "transition_durations": np.array(["0.1"], dtype=object),
            },
            "transition_durations.*text",
        ),
        ({"spike_counts": np.array([["0"]], dtype=object)}, "spike_counts.*text"),
        ({"n_spikes": "0"}, "n_spikes.*text"),
    ],
)
def test_log_emission_tensor_rejects_text_metadata_before_numeric_coercion(overrides, match):
    with pytest.raises(ValueError, match=match):
        LogEmissionTensor(**_base_tensor_kwargs(**overrides))


def test_replay_model_rejects_rows_without_finite_likelihood_mass():
    tensor = _tensor_with_log_likelihood(
        np.array([[0.0, -1.0], [-np.inf, -np.inf]], dtype=float)
    )

    with pytest.raises(ValueError, match="at least one finite spatial-bin"):
        RandomModel().score(tensor, np.zeros((2, 2), dtype=float))


def test_log_emission_tensor_allows_individual_impossible_bins():
    tensor = _tensor_with_log_likelihood(np.array([[0.0, -np.inf], [-1.0, 0.0]], dtype=float))

    assert tensor.n_time == 2
    assert tensor.n_bins == 2
    assert np.isneginf(tensor.log_likelihood[0, 1])
