import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, ClusterlessMarkEncoding


def make_encoding() -> ClusterlessMarkEncoding:
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rate_hz=np.ones(2, dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        effective_spike_count=np.ones(2, dtype=float),
        mark_mean=np.zeros((2, 1), dtype=float),
        mark_variance=np.ones((2, 1), dtype=float),
        mark_feature_names=("amplitude",),
        spike_mark_source="synthetic",
        config=ClusterlessMarkConfig(mark_likelihood="diagonal-gaussian"),
        mark_likelihood="diagonal-gaussian",
    )


def test_clusterless_mark_likelihood_rejects_nan_marks() -> None:
    encoding = make_encoding()
    bad_marks = np.array([[np.nan]], dtype=float)

    with pytest.raises(ValueError, match="marks must contain finite values"):
        encoding.log_mark_likelihood(bad_marks)
