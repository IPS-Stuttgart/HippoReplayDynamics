import numpy as np
import pytest

from scripts.verify_tirole_crossfit_mixture import (
    failure_reason,
    reference_fit,
    reference_likelihoods,
    reference_templates,
    reference_weights,
)


@pytest.mark.parametrize("pi", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_independent_root_recovers_known_mixture(pi):
    ll = np.array([[0.8, 0.2], [0.2, 0.8]])
    weights = (1 - pi) * ll[:, 0] + pi * ll[:, 1]
    estimate, complete = reference_fit(ll, weights)
    assert complete
    assert estimate == pytest.approx(pi, abs=1e-12)


def test_invalid_fit_reasons_are_distinct():
    ll = np.array([[0.8, 0.2], [np.nan, np.nan]])
    assert failure_reason(ll, np.ones(2)) == "insufficient_training_anchors"
    assert failure_reason(ll, np.array([1.0, 0.0])) == "valid"
    assert failure_reason(ll, np.zeros(2)) == "no_target_observations"
    assert failure_reason(np.full((2, 2), 0.5), np.ones(2)) == "identical_components"
    assert np.isnan(reference_fit(ll, np.ones(2))[0])


def test_reference_histograms_exclude_heldout_fold_and_bootstrap_duplicates():
    q = np.random.default_rng(8).uniform(0, 1, (20, 5, 2))
    folds = np.arange(20) % 5
    a = reference_templates(q, folds, np.ones(20))
    q[folds == 2] = 0.99
    b = reference_templates(q, folds, np.ones(20))
    np.testing.assert_equal(a[2], b[2])
    weights = np.zeros(20)
    weights[0] = 20
    assert np.isnan(reference_templates(q, folds, weights)).all()


def test_reference_lookup_includes_closed_endpoint_and_equal_anchor_weight():
    q = np.zeros((20, 5, 2))
    q[:, :, 1] = 1.0
    q[0, 1:] = np.nan
    folds = np.arange(20) % 5
    template = reference_templates(q, folds, np.ones(20))
    ll = reference_likelihoods(q, folds, template)
    np.testing.assert_allclose(ll[1, 0, 1], template[1, 0, :, 4])
    weight = reference_weights(q, np.ones((20, 5), bool), pi=0.75)
    np.testing.assert_allclose(weight.sum(axis=(1, 2)), 1.0)
    assert weight[:, :, 1].sum() / weight.sum() == pytest.approx(0.75)
