import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_kleinman_spatial_expression import (
    change_moments,
    conditional_moments,
    estimate_reference,
    probability,
    score,
    select_anchors,
    target_profile,
)


def test_known_unchanged_map_has_zero_mean_for_any_occupancy():
    r = np.array([0.1, 5.0, 1.0, 20.0])
    for exposure in ([1.0, 2.0, 3.0, 4.0], [0.0, 8.0, 2.0, 1.0], [1.0, 0.0, 0.0, 0.0]):
        expected, variance = conditional_moments(r, r, exposure)
        assert expected == pytest.approx(0, abs=1e-12)
        assert variance >= 0


def test_moments_match_exact_enumeration_of_two_independent_spikes():
    old = np.array([1.0, 4.0, 2.0])
    after = np.array([3.0, 1.0, 2.0])
    ref = np.array([1.2, 3.0, 2.5])
    b, a = np.array([2.0, 1.0, 3.0]), np.array([1.0, 4.0, 1.0])
    values, weights = [], []
    for i in range(3):
        for j in range(3):
            base = np.eye(3)[i]
            future = np.eye(3)[j]
            values.append(score(future, ref, a) - score(base, ref, b))
            weights.append(probability(old, b)[i] * probability(after, a)[j])
    v, w = np.asarray(values), np.asarray(weights)
    mean = w @ v
    var = w @ (v - mean) ** 2
    assert change_moments(old, after, ref, b, a) == pytest.approx((mean, var))


def test_scalar_gain_changes_neither_conditional_mean_nor_variance():
    r = np.array([0.01, 4.0, 2.0, 10.0])
    ref = np.array([1.0, 3.0, 1.0, 7.0])
    t = np.array([1.0, 4.0, 2.0, 3.0])
    a = conditional_moments(r, ref, t)
    assert conditional_moments(4 * r, ref, t) == pytest.approx(a)
    assert conditional_moments(r, 7 * ref, t) == pytest.approx(a)
    assert np.sqrt(a[1] / 5) / np.sqrt(a[1] / 80) == pytest.approx(4)


def test_sharpening_is_positive_but_not_an_unchanged_map():
    r = np.array([0.01, 1.0, 20.0, 1.0, 0.01])
    t = np.ones(5)
    sharp = target_profile(r, "sharpen_1p5", 1)
    broad = target_profile(r, "broaden_0p5", 1)
    assert not np.allclose(probability(r, t), probability(sharp, t))
    assert conditional_moments(sharp, r, t)[0] > 0
    assert conditional_moments(broad, r, t)[0] < 0


def test_estimated_reference_can_create_occupancy_bias_without_true_map_change():
    truth = np.array([1.0, 2.0, 5.0])
    wrong = np.array([2.0, 1.0, 6.0])
    b, a = np.array([3.0, 1.0, 1.0]), np.array([1.0, 1.0, 3.0])
    expected, _ = change_moments(truth, truth, wrong, b, a)
    assert abs(expected) > 0.01
    assert change_moments(truth, truth, truth, b, a)[0] == pytest.approx(0)


def test_no_spikes_have_no_conditional_spatial_score():
    assert np.isnan(score(np.zeros(3), np.ones(3), np.ones(3)))
    with pytest.raises(ValueError):
        score(np.array([1, 0, 0]), np.ones(3), np.array([0, 1, 1]))


def test_translation_does_not_wrap():
    r = np.ones(40) * 1e-5
    r[35] = 20
    shifted = target_profile(r, "shift_20cm", 1)
    assert shifted.max() == pytest.approx(1e-5)


def test_reference_fit_uses_only_reference_counts_and_exposure():
    c = np.arange(20)
    t = np.ones(20)
    r, included = estimate_reference(c, t)
    assert included
    assert np.isfinite(r).all() and (r > 0).all()
    zero, kept = estimate_reference(np.zeros(20), t)
    assert not kept and np.all(zero == 1e-5)


def test_anchor_selection_deterministic_and_independent_of_future_scores():
    rows = [
        {
            "animal": f"Rat{animal}",
            "drug": drug,
            "novel": novel,
            "direction": d,
            "session": f"session{choice}",
            "opportunity_id": f"event{choice}",
            "original_run_pass": True,
            "minimum_units_descriptor": True,
            "status": "audited",
        }
        for animal in range(6)
        for drug in range(2)
        for novel in range(2)
        for d in range(2)
        for choice in range(3)
    ]
    frame = pd.DataFrame(rows)
    first = select_anchors(frame)
    shuffled = frame.sample(frac=1, random_state=11)
    shuffled["future_outcome"] = np.arange(len(shuffled))
    shuffled["ripple_spikes"] = 500000
    other = select_anchors(shuffled)
    assert len(first) == 48
    assert first.selection_key.tolist() == other.selection_key.tolist()


def test_missing_anchor_stratum_is_not_vacuous_completeness():
    with pytest.raises(ValueError, match="48"):
        select_anchors(
            pd.DataFrame(
                [
                    {
                        "animal": "one",
                        "drug": 0,
                        "novel": 0,
                        "direction": 0,
                        "session": "a",
                        "opportunity_id": "b",
                        "original_run_pass": True,
                        "minimum_units_descriptor": True,
                        "status": "audited",
                    }
                ]
            )
        )
