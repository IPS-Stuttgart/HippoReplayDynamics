import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext, identity_likelihood, score_event
from scripts.audit_2d_mua_rate_transfer import PRIMARY, K, calibrate, contrasts, decision, permuted_times, shuffled_indices, shuffled_prediction, validate_factors


def test_calibration_preserves_cell_field_shape_and_matches_raw_composition():
    rates = np.array([[1.0, 2, 4], [5, 1, 2], [2, 7, 1]])
    probability, gain, prior = calibrate(rates, np.array([20, 40, 140]), 100)
    adapted = rates * gain[:, None]
    np.testing.assert_allclose(adapted / rates, np.broadcast_to(gain[:, None], rates.shape))
    np.testing.assert_allclose(adapted.mean(axis=1) / adapted.mean(axis=1).sum(), probability)
    np.testing.assert_allclose(probability, (np.array([20, 40, 140]) + 100 * prior) / 300)


def test_gain_is_shared_between_maps_not_refit_to_wrong_spatial_order():
    rates = np.array([[1.0, 2, 4], [5, 1, 2]])
    calibration = np.array([10, 60])
    a = calibrate(rates, calibration, 100)
    b = calibrate(rates[:, [2, 0, 1]], calibration, 100)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y)


def fixture_scores():
    context = SpatialPredictionContext(np.column_stack([np.arange(6) * 8, np.zeros(6)]))
    rates = np.array([[7.0, 4, 1, 1, 1, 1], [1, 1, 1, 2, 4, 7], [8, 4, 2, 1, 1, 1], [1, 1, 1, 2, 3, 8]])
    counts = np.array([[3, 0, 2, 0], [2, 1, 2, 1], [1, 2, 0, 3]])
    edges = np.array([0, 0.02, 0.04, 0.052])
    times = (edges[:-1] + edges[1:]) / 2
    train, held, permutation = np.array([0, 1]), np.array([2, 3]), np.array([0, 3, 5, 1, 4, 2])
    metadata = {"dataset": "tanni2022", "animal": "r", "session": "s", "event_id": 1, "split": 0, "n_heldout_spikes": int(counts[:, held].sum())}
    unadapted = pd.DataFrame([metadata | row for row in score_event(counts, times, rates, train, held, context, permutation, np.ones(4) / 4)])
    original, shuffled = [], []
    for alpha in (100, 1000):
        p, gain, _ = calibrate(rates, np.array([15, 10, 30, 45]), alpha)
        adapted = rates * gain[:, None]
        originals = score_event(counts, times, adapted, train, held, context, permutation, p)
        original.extend(metadata | {"alpha": alpha} | r for r in originals)
        if alpha == 100:
            tll, hll = identity_likelihood(counts[:, train], adapted[train]), identity_likelihood(counts[:, held], adapted[held])
            for k in range(K):
                order = shuffled_indices(("tanni2022", "r", "s"), 1, len(counts), k)
                r = shuffled_prediction(tll[order], hll[order], permuted_times(edges, order), context, permutation)
                shuffled.extend(metadata | {"shuffle": k} | x for x in r)
    old_order = pd.DataFrame([metadata | {"contrast": f"first_order_imm__{name}", "delta": 0.1} for name in ("real_order_advantage", "order_map_interaction")])
    return pd.DataFrame(original), pd.DataFrame(shuffled), unadapted, old_order


def test_complete_factorial_and_paired_contrasts():
    original, shuffled, old, order = fixture_scores()
    split, events = contrasts(original, shuffled, old, order)
    assert len(split) == len(events) == 28
    assert set(PRIMARY) <= set(split.contrast)
    rows = original[original.alpha.eq(100)].set_index("map")
    direct = rows.loc["real", "score_first_order_imm"] - rows.loc["population_code_permuted", "score_first_order_imm"]
    assert split.set_index("contrast").loc["alpha100__imm_real_minus_wrong", "delta"] == pytest.approx(direct)
    for col in ("score_iid_position", "score_static_location"):
        assert np.ptp(shuffled[col]) < 1e-10


def test_missing_shuffle_and_empty_inputs_fail():
    original, shuffled, _, _ = fixture_scores()
    with pytest.raises(ValueError, match="missing whole-bin"):
        validate_factors(original, shuffled.iloc[1:])
    with pytest.raises(ValueError, match="empty"):
        validate_factors(original.iloc[:0], shuffled)


def test_calibration_improvement_alone_does_not_pass():
    rows = [
        {"dataset": dataset, "contrast": contrast, "mean": 1.0, "ci_low": 0.1, "positive_animals": n, "animals": n}
        for dataset, n in (("pfeiffer_foster", 4), ("tanni2022", 5))
        for contrast in PRIMARY
    ]
    table = pd.DataFrame(rows)
    assert decision(table).full_observation_transfer_gate.all()
    table.loc[table.dataset.eq("tanni2022") & table.contrast.eq("alpha100__imm_minus_event_global"), "ci_low"] = -1
    result = decision(table).set_index("dataset")
    assert not result.loc["tanni2022", "full_observation_transfer_gate"]
    assert result.loc["pfeiffer_foster", "full_observation_transfer_gate"]


def test_heldout_changes_cannot_update_training_posterior_after_gain_fit():
    rates = np.array([[5.0, 1], [1, 5], [6, 1], [1, 7]])
    p, gain, _ = calibrate(rates, np.array([20, 30, 40, 50]), 100)
    adapted = rates * gain[:, None]
    context = SpatialPredictionContext(np.array([[0.0, 0], [8, 0]]))
    counts = np.array([[2, 0, 3, 0], [0, 2, 0, 3]])
    a = score_event(counts, np.array([0.01, 0.03]), adapted, [0, 1], [2, 3], context, [1, 0], p)
    counts[:, 2:] = [[0, 3], [3, 0]]
    b = score_event(counts, np.array([0.01, 0.03]), adapted, [0, 1], [2, 3], context, [1, 0], p)
    assert a[0]["training_imm_posterior_sha256"] == b[0]["training_imm_posterior_sha256"]
    assert a[0]["score_first_order_imm"] != b[0]["score_first_order_imm"]


def test_independent_factorial_reconstruction_matches():
    from scripts.verify_2d_mua_rate_transfer import independent_contrasts

    original, shuffled, parent, old_order = fixture_scores()
    actual, _ = contrasts(original, shuffled, parent, old_order)
    expected, _ = independent_contrasts(original, shuffled, parent, old_order)
    key = ["dataset", "animal", "session", "event_id", "split", "contrast"]
    pd.testing.assert_frame_equal(actual.set_index(key).sort_index(), expected.set_index(key).sort_index(), atol=1e-10, rtol=1e-10)


def test_independent_hierarchical_interval_matches_parent():
    from scripts.audit_2d_count_conditioned_prediction import aggregate
    from scripts.verify_2d_mua_rate_transfer import reference_interval

    frame = pd.DataFrame(
        [
            {"dataset": "x", "animal": str(a), "session": str(s), "event_id": e, "contrast": "c", "delta": float(a + e), "delta_per_heldout_spike": float(a + e) / (e + 1)}
            for a in range(3)
            for s in range(2)
            for e in range(2)
        ]
    )
    actual = aggregate(frame)[0].iloc[0]
    expected = reference_interval(frame)
    for key in ("mean", "ci_low", "ci_high", "mean_per_heldout_spike", "per_spike_ci_low", "per_spike_ci_high"):
        assert actual[key] == pytest.approx(expected[key])
