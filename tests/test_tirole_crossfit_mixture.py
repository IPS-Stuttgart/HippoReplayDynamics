import numpy as np
import pandas as pd
import pytest

from scripts.crossfit_tirole_composition_mixture import fit_mixture, folds_for, likelihoods, target_weights, templates


@pytest.mark.parametrize("pi", [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0])
def test_known_categorical_mixture_recovers_population_fraction(pi):
    ll = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    observed = np.array([[(1 - pi) * 0.8 + pi * 0.2, (1 - pi) * 0.2 + pi * 0.8]])
    estimated, complete = fit_mixture(ll, observed)
    assert complete[0]
    np.testing.assert_allclose(estimated, pi, atol=1e-12)


def test_identical_templates_and_empty_target_do_not_pass():
    result, _ = fit_mixture(np.full((1, 2, 2), 0.5), np.ones((1, 2)))
    assert np.isnan(result).all()
    result, complete = fit_mixture(np.array([[[0.8, 0.2], [0.2, 0.8]]]), np.zeros((1, 2)))
    assert np.isnan(result).all() and not complete.any()


def test_missing_required_likelihood_is_not_silently_dropped():
    result, complete = fit_mixture(np.array([[[0.8, 0.2], [np.nan, np.nan]]]), np.ones((1, 2)))
    assert np.isnan(result).all() and not complete.any()


def test_no_anchor_copy_enters_own_fold_training():
    rng = np.random.default_rng(8)
    q = rng.uniform(0, 1, (20, 5, 2))
    folds = np.arange(20) % 5
    a = templates(q, folds, np.ones((1, 20)))
    q[folds == 2] = 0.99
    b = templates(q, folds, np.ones((1, 20)))
    np.testing.assert_equal(a[:, 2], b[:, 2])
    assert not np.array_equal(a[:, 1], b[:, 1])


def test_bootstrap_refits_templates_and_requires_unique_anchors():
    q = np.random.default_rng(5).uniform(0, 1, (20, 5, 2))
    folds = np.arange(20) % 5
    w = np.ones((2, 20))
    w[1, 0] = 10
    x = templates(q, folds, w)
    assert not np.array_equal(x[0, 1], x[1, 1])
    w = np.zeros((1, 20))
    w[0, 0] = 20
    assert np.isnan(templates(q, folds, w)).all()


def test_template_likelihood_lookup_uses_correct_track_and_bin():
    q = np.random.default_rng(2).uniform(0, 1, (20, 5, 2))
    folds = np.arange(20) % 5
    c = templates(q, folds, np.ones((1, 20)))
    ll = likelihoods(q, folds, c)
    for a, s, t in [(1, 3, 0), (5, 0, 1), (19, 4, 1)]:
        b = min(int(q[a, s, t] * 5), 4)
        np.testing.assert_equal(ll[0, a, s, t], c[0, folds[a], s, :, b])


def test_equal_anchor_weight_and_known_mixture_despite_missing_splits():
    q = np.ones((2, 5, 2)) * 0.5
    q[0, 1:] = np.nan
    w = target_weights(q, np.ones((2, 5), bool), prior=0.75)
    np.testing.assert_allclose(w.sum(axis=(1, 2)), 1.0)
    np.testing.assert_allclose(w[:, :, 1].sum() / w.sum(), 0.75)


def test_selection_does_not_renormalize_retained_anchor():
    q = np.ones((2, 2, 2)) * 0.5
    selection = np.zeros_like(q, bool)
    selection[0, 0, 1] = True
    w = target_weights(q, np.ones((2, 2), bool), accepted=selection)
    assert w.sum() == 0.25


def test_fold_assignment_stable_when_anchor_table_reordered():
    x = pd.DataFrame({"event_id": range(20), "epoch": ["POST"] * 20, "primary_ripple_candidate": [True] * 20})
    a = dict(zip(x.event_id, folds_for(x, "test"), strict=True))
    y = x.sample(frac=1, random_state=5)
    b = dict(zip(y.event_id, folds_for(y, "test"), strict=True))
    assert a == b


def test_duplicate_split_copies_cannot_increase_anchor_weight():
    q = np.random.default_rng(2).uniform(0, 1, (20, 5, 2))
    folds = np.arange(20) % 5
    estimates = []
    for x in [q, np.tile(q, (1, 2, 1))]:
        ll = likelihoods(x, folds, templates(x, folds, np.ones((1, len(x)))))
        w = target_weights(x, np.ones(x.shape[:2], bool), prior=0.75)
        estimates.append(fit_mixture(ll.reshape(1, -1, 2), w.reshape(1, -1))[0])
    np.testing.assert_allclose(*estimates)
