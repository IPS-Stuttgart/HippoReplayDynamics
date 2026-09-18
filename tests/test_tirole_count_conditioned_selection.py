from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.two_track_content import classify_sequence, field_shift_posteriors
from scripts.calibrate_tirole_count_conditioned_selection import compare_unchanged
from scripts.calibrate_tirole_selection_transport import simulate_anchor
from scripts.report_tirole_likelihood_selection import GROUPS, metrics, paired_metrics


def direct(counts, rates, valid, conditional):
    lam = np.maximum(rates, 1e-4)
    if conditional:
        lam = lam / lam.sum(axis=1, keepdims=True)
    ll = np.array([[(np.log(lam[k]) * c[:, None]).sum(axis=0) - (0 if conditional else 0.02 * lam[k].sum(axis=0)) for k in range(2)] for c in counts])
    ll -= np.log(valid.sum(axis=1))[None, :, None]
    ll[:, ~valid] = -np.inf
    return np.exp(ll - logsumexp(ll, axis=(1, 2), keepdims=True))


@pytest.mark.parametrize("conditional", [False, True])
def test_shifted_likelihood_matches_direct_multinomial_or_poisson(conditional):
    rng = np.random.default_rng(37)
    rates = rng.uniform(0.001, 4, (2, 8, 11))
    counts = rng.poisson(1, (7, 8))
    valid = rng.random((2, 11)) > 0.2
    shifts = rng.integers(0, 11, (10, 2, 8))
    expected = []
    for shift in shifts:
        rolled = np.array([[np.roll(rates[k, u], shift[k, u]) for u in range(8)] for k in range(2)])
        expected.append(direct(counts, rolled, valid, conditional))
    actual = field_shift_posteriors(counts, rates, valid, shifts, conditional_count=conditional)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_conditional_likelihood_normalizes_after_floor_and_ignores_common_local_gain():
    rng = np.random.default_rng(15)
    rates = rng.uniform(0.1, 10, (2, 8, 15))
    counts = rng.poisson(1, (5, 8))
    valid = np.ones((2, 15), bool)
    shifts = np.zeros((1, 2, 8), int)
    a = field_shift_posteriors(counts, rates, valid, shifts, conditional_count=True)
    b = field_shift_posteriors(counts, rates * rng.uniform(0.1, 20, (2, 1, 15)), valid, shifts, conditional_count=True)
    np.testing.assert_allclose(a, b, atol=1e-14)
    rates[:, 0, 0] = 1e-10
    rates[:, 1:, 0] = 1000
    np.testing.assert_allclose(field_shift_posteriors(counts, rates, valid, shifts, conditional_count=True)[0], direct(counts, rates, valid, True), atol=1e-14)


def test_likelihood_callback_does_not_change_observations_or_evaluation():
    rng = np.random.default_rng(3)
    info = {"session": "fixture", "animal": "RAT"}
    parts = {"splits": [{"inference": list(range(10)), "evaluation": [10, 11]}]}
    event = {"event_id": 0, "epoch": "POST", "primary_ripple_candidate": True}
    original = rng.poisson(0.4, (8, 12))
    maps = {"rates": rng.uniform(0.1, 10, (2, 12, 10)), "valid_bins": np.ones((2, 10), bool), "bin_centers_cm": np.arange(10) * 10.0}
    old, oc = simulate_anchor(info, parts, event, original, maps, n_draws=1, n_repeats=1)
    new, nc = simulate_anchor(info, parts, event, original, maps, n_draws=1, n_repeats=1, sequence_classifier=partial(classify_sequence, conditional_count=True))
    compare_unchanged(new, old, nc, oc)
    broken = nc.copy()
    broken.loc[0, "n_evaluation_spikes"] += 1
    with pytest.raises(AssertionError):
        compare_unchanged(new, old, broken, oc)
    broken = new.copy()
    broken.loc[0, "n_inference_spikes"] += 1
    with pytest.raises(AssertionError):
        compare_unchanged(broken, old, nc, oc)


def anchor_fixture():
    rows = []
    for a in range(3):
        for track in [1, 2]:
            for group in GROUPS:
                accepted = float(group == "full" or (group in ["half", "retained"] and track == 2) or (group == "lost" and track == 1))
                rows.append(
                    {
                        "anchor_event": a,
                        "truth_track": track,
                        "group": group,
                        "group_mass": accepted,
                        "label2_mass": accepted * (track == 2),
                        "correct_label_mass": accepted,
                        "poisson_z_numerator": 2 * accepted,
                        "poisson_z_denominator": accepted,
                        "conditional_count_z_numerator": accepted,
                        "conditional_count_z_denominator": accepted,
                    }
                )
    return pd.DataFrame(rows)


def test_metrics_count_original_anchors_and_preserve_unknown_groups():
    actual = metrics(anchor_fixture(), np.ones(3))
    assert actual["full_true_track2_fraction"][0] == 0.5
    assert actual["half_true_track2_fraction"][0] == 1
    assert actual["half_minus_full_true_track2_fraction"][0] == 0.5
    assert np.isnan(actual["gained_true_track2_fraction"][0])
    assert actual["lost_conditional_count_true_signed_z"][0] == 1
    with pytest.raises(ValueError, match="incomplete"):
        metrics(anchor_fixture().iloc[1:], np.ones(3))


def test_paired_bias_reduction_positive_only_when_absolute_distortion_shrinks():
    old = metrics(anchor_fixture(), np.ones(3))
    new = {k: v.copy() for k, v in old.items()}
    new["half_absolute_distortion"] = np.array([0.2])
    actual = paired_metrics(old, new)
    assert actual["half_absolute_distortion_reduction"][0] == pytest.approx(0.3)
    assert actual["half_absolute_distortion_conditional_minus_poisson"][0] == pytest.approx(-0.3)
