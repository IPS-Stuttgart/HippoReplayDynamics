import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from scripts.verify_position_free_assembly_prediction import (
    audit_fit,
    compare_table,
    dense_posterior,
    direct_scores,
    raw_binned,
    verify_outputs,
)


def test_dense_posterior_exact_enumeration():
    ll = np.log(np.array([[0.2, 0.8], [0.7, 0.3], [0.9, 0.1]]))
    w = np.array([0.3, 0.7])
    centers = np.array([0, 0.004, 0.024])
    expected = np.zeros_like(ll)
    total = 0
    for sequence in itertools.product(range(2), repeat=3):
        probability = w[sequence[0]] * np.exp(ll[0, sequence[0]])
        for t in (1, 2):
            rho = np.exp(-(centers[t] - centers[t - 1]) / 0.06)
            probability *= (rho * (sequence[t] == sequence[t - 1]) + (1 - rho) * w[sequence[t]]) * np.exp(ll[t, sequence[t]])
        total += probability
        for t, state in enumerate(sequence):
            expected[t, state] += probability
    np.testing.assert_allclose(np.exp(dense_posterior(ll, w, centers, "persistent")), expected / total)


def test_spike_endpoint_and_padded_last_bin():
    times = [-1, 0, 0.01, 0.02, 0.025, 0.027, 0.03]
    np.testing.assert_array_equal(raw_binned(times, np.array([0, 0.01, 0.02, 0.03]), 0.027), [1, 1, 2])


def test_zero_heldout_and_no_inference_leakage():
    counts = np.array([[2, 1, 0, 0], [1, 4, 0, 0]])
    p = np.array([[0.4, 0.1], [0.1, 0.4], [0.3, 0.1], [0.2, 0.4]])
    w = np.array([0.4, 0.6])
    result = direct_scores(counts, np.array([0, 0.02]), [0, 1], [2, 3], p, w, p.mean(axis=1))
    np.testing.assert_allclose(list(result.values()), 0, atol=1e-12)
    q = dense_posterior(np.zeros((2, 2)), w, np.array([0, 0.02]), "persistent")
    np.testing.assert_allclose(logsumexp(q, axis=1), 0, atol=1e-12)


def test_missing_and_changed_summary_fail():
    frame = pd.DataFrame({"id": [1, 2], "value": [2.0, 3.0]})
    with pytest.raises(ValueError, match="missing"):
        compare_table(frame, frame.iloc[:1], ["id"], ["value"], "test")
    with pytest.raises(ValueError, match="mismatch"):
        compare_table(frame, frame.assign(value=0), ["id"], ["value"], "test")


def test_incomplete_run_rejected(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        verify_outputs(tmp_path, {"status": "running"})


def test_unconverged_fit_rejected():
    counts = np.ones((10, 4), dtype=int)
    p = np.full((4, 2), 0.25)
    with pytest.raises(ValueError, match="unconverged"):
        audit_fit(counts, p, np.array([0.5, 0.5]), p[:, 0], np.array([-2, -1]), SimpleNamespace(converged=False), ())
