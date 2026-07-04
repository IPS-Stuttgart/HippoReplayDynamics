from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, ClusterlessMarkEncoding


def _minimal_clusterless_encoding() -> ClusterlessMarkEncoding:
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rate_hz=np.array([1.0], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        effective_spike_count=np.array([1.0], dtype=float),
        mark_mean=np.zeros((1, 2), dtype=float),
        mark_variance=np.ones((1, 2), dtype=float),
        mark_feature_names=("m0", "m1"),
        spike_mark_source="unit-test",
        config=ClusterlessMarkConfig(mark_likelihood="diagonal-gaussian", mark_group_by="none"),
        mark_likelihood="diagonal-gaussian",
    )


@pytest.mark.parametrize(
    ("marks", "match"),
    [
        (np.array([[True, False]], dtype=bool), "boolean"),
        (np.array([[np.nan, 0.0]], dtype=float), "finite"),
        (np.array([[1.0 + 1.0j, 0.0 + 0.0j]], dtype=complex), "complex"),
    ],
)
def test_clusterless_mark_likelihood_rejects_invalid_direct_mark_values(marks: np.ndarray, match: str) -> None:
    encoding = _minimal_clusterless_encoding()

    with pytest.raises(ValueError, match=match):
        encoding.log_mark_likelihood(marks)


def test_clusterless_mark_likelihood_accepts_real_zero_imaginary_complex_marks() -> None:
    encoding = _minimal_clusterless_encoding()

    likelihood = encoding.log_mark_likelihood(np.array([[1.0 + 0.0j, 0.0 + 0.0j]], dtype=complex))

    assert likelihood.shape == (1, 1)
    assert np.all(np.isfinite(likelihood))
