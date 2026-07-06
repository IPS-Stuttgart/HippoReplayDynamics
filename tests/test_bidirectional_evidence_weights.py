from __future__ import annotations

import numpy as np

from hipporeplayimm.reverse_models import _evidence_mixture_weights


def test_evidence_mixture_weights_split_positive_infinite_evidence() -> None:
    weights = _evidence_mixture_weights(np.asarray([np.inf, 0.0, np.inf], dtype=float))

    np.testing.assert_allclose(weights, np.asarray([0.5, 0.0, 0.5], dtype=float))
    assert np.all(np.isfinite(weights))


def test_evidence_mixture_weights_ignore_nan_and_negative_infinity() -> None:
    weights = _evidence_mixture_weights(
        np.asarray([np.nan, 2.0, -np.inf, 0.0], dtype=float)
    )

    assert np.all(np.isfinite(weights))
    assert weights[0] == 0.0
    assert weights[2] == 0.0
    np.testing.assert_allclose(weights[[1, 3]].sum(), 1.0)
    assert weights[1] > weights[3]


def test_evidence_mixture_weights_fall_back_for_all_nonfinite_nonwinning_values() -> None:
    weights = _evidence_mixture_weights(np.asarray([np.nan, -np.inf], dtype=float))

    np.testing.assert_allclose(weights, np.asarray([0.5, 0.5], dtype=float))
    assert np.all(np.isfinite(weights))
