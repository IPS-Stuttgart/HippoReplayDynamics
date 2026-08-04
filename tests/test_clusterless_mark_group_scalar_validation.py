from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, ClusterlessMarkEncoding


def _encoding_with_group_ids(group_ids: np.ndarray) -> ClusterlessMarkEncoding:
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.0, 0.0]]),
        rate_hz=np.array([1.0]),
        occupancy_s=np.array([1.0]),
        effective_spike_count=np.array([1.0]),
        mark_mean=np.array([[0.0]]),
        mark_variance=np.array([[1.0]]),
        mark_feature_names=("mark0",),
        spike_mark_source="synthetic:marks",
        config=ClusterlessMarkConfig(
            mark_likelihood="diagonal-gaussian",
            mark_group_by="tetrode",
        ),
        mark_likelihood="diagonal-gaussian",
        group_ids=np.asarray(group_ids),
        group_rate_hz=np.ones((len(group_ids), 1)),
        group_effective_spike_count=np.ones((len(group_ids), 1)),
        group_mark_mean=np.zeros((len(group_ids), 1, 1)),
        group_mark_variance=np.ones((len(group_ids), 1, 1)),
    )


def _nested_object_scalar(value: object) -> np.ndarray:
    values = np.empty(1, dtype=object)
    values[0] = np.asarray(value, dtype=object)
    return values


@pytest.mark.parametrize(
    "value",
    [
        np.complex128(1.0 + 2.0j),
        np.complex128(1.0 + 0.0j),
        np.clongdouble(1.0 + 2.0j),
    ],
)
def test_clusterless_mark_likelihood_rejects_numpy_complex_group_ids(
    value: object,
) -> None:
    encoding = _encoding_with_group_ids(np.array([1], dtype=object))

    with pytest.raises(ValueError, match="real integer"):
        encoding.log_mark_likelihood(
            np.array([[0.0]]),
            group_ids=np.array([value], dtype=object),
        )


def test_clusterless_mark_likelihood_rejects_nested_numpy_complex_group_ids() -> None:
    encoding = _encoding_with_group_ids(np.array([1], dtype=object))

    with pytest.raises(ValueError, match="real integer"):
        encoding.log_mark_likelihood(
            np.array([[0.0]]),
            group_ids=_nested_object_scalar(np.clongdouble(1.0 + 2.0j)),
        )


def test_clusterless_mark_likelihood_rejects_fractional_extended_precision_group_ids() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("long double does not provide extended precision on this platform")
    value = np.longdouble("9007199254740993.5")
    assert not bool(value.is_integer())
    assert float(value).is_integer()
    encoding = _encoding_with_group_ids(np.array([9007199254740993], dtype=object))

    with pytest.raises(ValueError, match="integer-valued"):
        encoding.log_mark_likelihood(
            np.array([[0.0]]),
            group_ids=np.array([value], dtype=object),
        )


def test_clusterless_mark_likelihood_preserves_integral_extended_precision_group_ids() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("long double does not provide extended precision on this platform")
    value = np.longdouble("9007199254740993")
    encoding = _encoding_with_group_ids(np.array([9007199254740993], dtype=object))

    group_indices = encoding._coerce_group_indices(
        np.array([value], dtype=object),
        n_marks=1,
    )

    np.testing.assert_array_equal(group_indices, np.array([0]))
