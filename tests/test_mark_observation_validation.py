from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, ClusterlessMarkEncoding


def _simple_mark_encoding() -> ClusterlessMarkEncoding:
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rate_hz=np.array([1.0]),
        occupancy_s=np.array([1.0]),
        effective_spike_count=np.array([1.0]),
        mark_mean=np.array([[0.0]]),
        mark_variance=np.array([[1.0]]),
        mark_feature_names=("amplitude",),
        spike_mark_source="synthetic",
        config=ClusterlessMarkConfig(mark_likelihood="diagonal-gaussian"),
        mark_likelihood="diagonal-gaussian",
    )


@pytest.mark.parametrize(
    "bad_marks",
    [
        np.array([[True]], dtype=bool),
        [[False]],
        np.array([["0.0"]], dtype=object),
        np.array([[1.0 + 1.0j]], dtype=complex),
        np.array([[np.complex128(1.0 + 1.0j)]], dtype=object),
    ],
)
def test_clusterless_mark_likelihood_rejects_lossy_observation_marks(bad_marks) -> None:
    encoding = _simple_mark_encoding()

    with pytest.raises(ValueError, match="marks"):
        encoding.log_mark_likelihood(bad_marks)


@pytest.mark.parametrize(
    "marks",
    [
        np.array([[0.0 + 0.0j]], dtype=complex),
        np.array([[np.complex128(0.0 + 0.0j)]], dtype=object),
    ],
)
def test_clusterless_mark_likelihood_accepts_zero_imaginary_observations(marks) -> None:
    encoding = _simple_mark_encoding()

    likelihood = encoding.log_mark_likelihood(marks)

    assert likelihood.shape == (1, encoding.n_bins)
    assert np.isfinite(likelihood).all()
