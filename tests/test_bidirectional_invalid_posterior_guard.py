import numpy as np

from hipporeplayimm.bidirectional_infinite_evidence_patch import _safe_mixture_log_posterior


def test_safe_bidirectional_mixture_skips_nan_log_posterior_branch():
    valid = np.log(np.array([0.25, 0.75], dtype=float))
    invalid = np.array([np.nan, np.nan], dtype=float)

    mixed = _safe_mixture_log_posterior(
        [invalid, valid],
        np.array([0.5, 0.5], dtype=float),
    )

    assert mixed is not None
    np.testing.assert_allclose(mixed, valid)


def test_safe_bidirectional_mixture_skips_all_impossible_log_posterior_branch():
    valid = np.log(np.array([0.8, 0.2], dtype=float))
    impossible = np.array([-np.inf, -np.inf], dtype=float)

    mixed = _safe_mixture_log_posterior(
        [impossible, valid],
        np.array([0.5, 0.5], dtype=float),
    )

    assert mixed is not None
    np.testing.assert_allclose(mixed, valid)
