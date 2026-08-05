from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from hipporeplayimm.bidirectional_infinite_evidence_patch import (
    _equal_prior_logp_and_weights,
    _safe_mixture_log_posterior,
    _terminal_log_posterior_from_score,
)


def _nested_complex_scalar(value: complex) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = np.complex128(value)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0 + 2.0j, 0.0]),
        [np.complex128(1.0 + 0.0j), 0.0],
        np.array([_nested_complex_scalar(1.0 + 2.0j), 0.0], dtype=object),
    ],
)
def test_bidirectional_evidence_rejects_complex_values_without_lossy_cast(
    values: object,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="log likelihoods must be real"):
            _equal_prior_logp_and_weights(values)


def test_bidirectional_posterior_mixture_rejects_nested_complex_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="posterior arrays must be real"):
            _safe_mixture_log_posterior(
                [
                    _nested_complex_scalar(0.0 + 1.0j),
                    np.log(np.array([0.25, 0.75], dtype=float)),
                ],
                np.array([0.5, 0.5], dtype=float),
            )


def test_bidirectional_terminal_rejects_nested_complex_values() -> None:
    score = SimpleNamespace(
        terminal_log_posterior=_nested_complex_scalar(0.0 + 1.0j),
        trajectory_log_posterior=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="posterior arrays must be real"):
            _terminal_log_posterior_from_score(score)


def test_bidirectional_evidence_keeps_real_numeric_strings() -> None:
    logp, weights = _equal_prior_logp_and_weights(["0.0", "-1.0"])

    assert np.isfinite(logp)
    np.testing.assert_allclose(
        weights,
        np.array([0.7310585786300049, 0.2689414213699951], dtype=float),
    )
