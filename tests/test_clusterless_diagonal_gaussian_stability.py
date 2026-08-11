from __future__ import annotations

import warnings

import numpy as np

from hipporeplayimm.clusterless import ClusterlessMarkEncoding


def _diagonal_encoding(*, mean: float, variance: float) -> ClusterlessMarkEncoding:
    encoding = object.__new__(ClusterlessMarkEncoding)
    encoding.mark_mean = np.array([[mean]], dtype=float)
    encoding.mark_variance = np.array([[variance]], dtype=float)
    encoding.mark_likelihood = "diagonal-gaussian"
    return encoding


def test_diagonal_clusterless_gaussian_keeps_large_variance_normalizer_finite() -> None:
    variance = np.finfo(float).max
    encoding = _diagonal_encoding(mean=0.0, variance=variance)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        log_likelihood = encoding.log_mark_likelihood(np.array([[0.0]], dtype=float))

    expected = -0.5 * (np.log(2.0 * np.pi) + np.log(variance))
    assert np.isfinite(log_likelihood[0, 0])
    np.testing.assert_allclose(log_likelihood[0, 0], expected, rtol=1e-15)


def test_diagonal_clusterless_gaussian_standardizes_before_squaring() -> None:
    variance = 1.0e308
    mark = 1.0e200
    encoding = _diagonal_encoding(mean=0.0, variance=variance)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        log_likelihood = encoding.log_mark_likelihood(np.array([[mark]], dtype=float))

    standardized = mark / np.sqrt(variance)
    expected = -0.5 * (
        standardized * standardized
        + np.log(2.0 * np.pi)
        + np.log(variance)
    )
    assert np.isfinite(log_likelihood[0, 0])
    np.testing.assert_allclose(log_likelihood[0, 0], expected, rtol=1e-15)
