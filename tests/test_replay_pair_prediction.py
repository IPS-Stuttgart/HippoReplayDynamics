"""Prediction algebra and grouping; not six-generator spike-level calibration."""
from dataclasses import replace

import numpy as np
import pytest

from scripts._replay_pair_prediction import (
    PairData, aggregate_event_orders, fit, hierarchical_weights, nuisance_features,
    prediction_check, validate,
)


def fixture(effect=0.0, seed=42, pauses=3, pairs=30):
    rng = np.random.default_rng(seed)
    n = 4 * pauses * pairs
    animal = np.repeat([f"rat{i}" for i in range(4)], pauses * pairs)
    session = np.array([f"{a}/day" for a in animal])
    pause = np.tile(np.repeat([f"pause{i}" for i in range(pauses)], pairs), 4)
    pre = rng.normal(size=n)
    order = rng.normal(size=(n, 21))
    # Orthogonal, deterministic null/positive fixtures isolate implementation,
    # not plausible neural generators or empirical false-positive calibration.
    baseline = pre[:, None]
    post = 0.3 * pre + effect * order[:, 0]
    return PairData(animal, session, pause, np.tile(np.arange(pairs) * 2, 4 * pauses),
                    np.tile(np.arange(pairs) * 2 + 1, 4 * pauses), pre, post, baseline, order)


def test_endpoint_is_change_and_preceding_coordination_is_mandatory():
    data = fixture()
    np.testing.assert_allclose(data.change, -0.7 * data.pre)
    with pytest.raises(ValueError, match="unchanged PRE"):
        validate(replace(data, baseline=np.ones((len(data.pre), 1))))


def test_existing_similarity_and_regression_to_mean_do_not_create_order_gain():
    data = fixture()
    data = replace(data, order=data.order + 4 * data.pre[:, None])
    result = prediction_check(data)
    assert abs(result["summary"]["original_mean_animal_gain"]) < 1e-20
    assert abs(result["summary"]["original_minus_animal_median_shuffle_gain"]) < 1e-20
    np.testing.assert_allclose(result["predictions"], data.change[:, None] + np.zeros_like(data.order), atol=1e-12)


def test_order_specific_signal_predicts_heldout_animals_beyond_pre():
    result = prediction_check(fixture(effect=2.0))
    assert result["summary"]["original_mean_animal_gain"] > 1
    assert result["summary"]["original_minus_animal_median_shuffle_gain"] > 1
    assert all(row["heldout_mse_improvement"] > 1 for row in result["by_animal"] if row["condition"] == 0)
    assert not result["summary"]["biological_calibration_complete"]
    assert not result["summary"]["goal_complete"]


def test_complete_refit_for_every_shuffle_and_every_heldout_animal(monkeypatch):
    from scripts import _replay_pair_prediction as core
    calls = []
    original = core.fit

    def record(*args, **kwargs):
        calls.append(kwargs.get("order_column"))
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "fit", record)
    result = core.prediction_check(fixture())
    assert len(calls) == 4 * 21 * 2
    assert calls.count(None) == 4 * 21
    assert len(result["folds"]) == 4 * 21


def test_withheld_animal_response_never_changes_its_predictions():
    data = fixture(effect=1.0)
    heldout = data.animal == "rat0"
    reference = prediction_check(data)
    post = data.post.copy()
    post[heldout] += np.arange(heldout.sum()) * 100
    altered = prediction_check(replace(data, post=post))
    np.testing.assert_array_equal(reference["predictions"][heldout], altered["predictions"][heldout])
    np.testing.assert_array_equal(reference["baseline_predictions"][heldout], altered["baseline_predictions"][heldout])


def test_feature_scaling_uses_training_rows_only():
    x = np.array([[1., 2.], [2., 4.], [3., 8.]])
    model = fit(x, np.array([2., 4., 6.]), np.ones(3))
    expected = np.sqrt(np.mean(x * x, axis=0))
    np.testing.assert_allclose(model.scale, expected)
    before = model.scale.copy()
    model.predict(np.array([[1e10, -1e12]]))
    np.testing.assert_array_equal(model.scale, before)


def test_nuisance_features_and_prediction_are_orientation_equivariant():
    data = fixture(effect=1.0)
    n = len(data.pre)
    rng = np.random.default_rng(8)
    a, b = rng.uniform(0, 50, (2, n))
    ea, eb = rng.integers(0, 20, (2, n))
    participation = rng.uniform(size=n)
    baseline = nuisance_features(data.pre, a, b, ea, eb, participation)
    flipped = nuisance_features(-data.pre, b, a, eb, ea, participation)
    np.testing.assert_array_equal(flipped, -baseline)
    normal = prediction_check(replace(data, baseline=baseline))
    negative = prediction_check(replace(data, pre=-data.pre, post=-data.post,
                                        baseline=flipped, order=-data.order))
    np.testing.assert_allclose(normal["predictions"], -negative["predictions"], atol=1e-12)
    np.testing.assert_allclose(normal["summary"]["original_mean_animal_gain"],
                               negative["summary"]["original_mean_animal_gain"], atol=1e-12)


def test_rowwise_orientation_flips_leave_predictions_and_scores_equivariant():
    data = fixture(effect=1)
    sign = np.where(np.arange(len(data.pre)) % 3 == 0, -1, 1)
    one = prediction_check(data)
    two = prediction_check(replace(data, pre=data.pre * sign, post=data.post * sign,
                                   baseline=data.baseline * sign[:, None], order=data.order * sign[:, None]))
    np.testing.assert_allclose(two["predictions"], one["predictions"] * sign[:, None], atol=1e-12)
    assert np.isclose(one["summary"]["original_mean_animal_gain"], two["summary"]["original_mean_animal_gain"])


def test_weights_do_not_treat_many_pairs_as_many_animals():
    animal = np.array(["A"] * 100 + ["B"] * 3)
    session = np.array(["A/day"] * 100 + ["B/day"] * 3)
    pause = np.array(["A/p"] * 100 + ["B/p"] * 3)
    w = hierarchical_weights(animal, session, pause)
    assert np.isclose(w[:100].sum(), .5) and np.isclose(w[100:].sum(), .5)


def test_weights_balance_recordings_and_pauses_within_animal():
    animal = np.array(["A"] * 5 + ["B"])
    session = np.array(["s0", "s0", "s0", "s1", "s1", "s2"])
    pause = np.array(["p0", "p0", "p1", "p0", "p0", "p0"])
    w = hierarchical_weights(animal, session, pause)
    np.testing.assert_allclose(w, [1 / 16, 1 / 16, 1 / 8, 1 / 8, 1 / 8, 1 / 2])


def test_zero_order_and_zero_event_rows_are_retained():
    data = fixture()
    result = prediction_check(replace(data, order=np.zeros_like(data.order)))
    assert result["summary"]["pair_rows"] == len(data.pre)
    assert all(row["order_coefficient_original_units"] == 0 for row in result["folds"])
    np.testing.assert_allclose(result["predictions"], result["baseline_predictions"], atol=1e-12)


@pytest.mark.parametrize("field", ["pre", "post", "baseline", "order"])
def test_nonfinite_measurements_cannot_enter_fit(field):
    data = fixture()
    value = getattr(data, field).copy()
    value.flat[0] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        prediction_check(replace(data, **{field: value}))


def test_duplicate_pair_and_missing_shuffle_fail_nonvacuously():
    data = fixture()
    a, b = data.unit_a.copy(), data.unit_b.copy()
    a[1], b[1] = a[0], b[0]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(replace(data, unit_a=a, unit_b=b))
    with pytest.raises(ValueError, match="shuffled"):
        validate(replace(data, order=data.order[:, :1]))


def test_animal_identity_mismatch_and_insufficient_animals_fail():
    data = fixture()
    session = data.session.copy()
    session[-1] = session[0]
    with pytest.raises(ValueError, match="multiple animals"):
        validate(replace(data, session=session))
    mask = np.isin(data.animal, ["rat0", "rat1"])
    subset = PairData(**{name: getattr(data, name)[mask] for name in data.__dataclass_fields__})
    with pytest.raises(ValueError, match="three animals"):
        prediction_check(subset)


def test_unsigned_rate_covariates_cannot_replace_antisymmetric_controls():
    with pytest.raises(ValueError, match="Invalid pair"):
        nuisance_features(np.ones(3), -np.ones(3), np.ones(3), np.ones(3), np.ones(3), np.ones(3))


def event(identity="e0", ids=(0, 2), value=.5):
    original = np.array([[0., value], [-value, 0.]])
    return {"event_id": identity, "validated_replay": True, "unit_ids": np.array(ids),
            "spike_counts": np.array([4, 6]), "order": original,
            "shuffle_order": np.stack([original / 2, -original])}


def test_event_aggregation_keeps_inactive_pairs_and_zero_event_pauses():
    one = aggregate_event_orders(np.array([0, 1, 2]), [event()], 2)
    np.testing.assert_array_equal(one["unit_a"], [0, 0, 1])
    np.testing.assert_array_equal(one["unit_b"], [1, 2, 2])
    np.testing.assert_array_equal(one["order"], [[0, 0, 0], [.5, .25, -.5], [0, 0, 0]])
    np.testing.assert_array_equal(one["participation"], [0, 1, 0])
    zero = aggregate_event_orders(np.array([0, 1, 2]), [], 2)
    assert zero["validated_events"] == 0 and zero["order"].shape == (3, 3)
    assert not zero["order"].any() and not zero["participation"].any()


def test_event_count_cannot_multiply_pause_pair_weight():
    one = aggregate_event_orders(np.array([0, 2]), [event()], 2)
    many = aggregate_event_orders(np.array([0, 2]), [event("e0"), event("e1")], 2)
    for name in ["order", "event_spikes_a", "event_spikes_b", "participation"]:
        np.testing.assert_array_equal(one[name], many[name])


def test_unvalidated_unknown_population_or_duplicate_event_cannot_enter():
    unvalidated = event()
    unvalidated["validated_replay"] = False
    with pytest.raises(ValueError, match="Unvalidated"):
        aggregate_event_orders(np.array([0, 2]), [unvalidated], 2)
    with pytest.raises(ValueError, match="population differs"):
        aggregate_event_orders(np.array([0, 1]), [event()], 2)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_event_orders(np.array([0, 2]), [event(), event()], 2)


def test_event_shuffle_completeness_and_antisymmetry_are_required():
    bad = event()
    bad["shuffle_order"] = bad["shuffle_order"][:1]
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_event_orders(np.array([0, 2]), [bad], 2)
    bad = event()
    bad["order"][0, 0] = .1
    with pytest.raises(ValueError, match="directional"):
        aggregate_event_orders(np.array([0, 2]), [bad], 2)


def test_independent_profile_solution_matches_all_predictions():
    from scripts.verify_replay_pair_prediction_development import reference_predictions
    data = fixture(effect=1)
    rng = np.random.default_rng(12)
    data = replace(data, baseline=np.column_stack((data.pre, rng.normal(size=(len(data.pre), 3)))))
    result = prediction_check(data)
    reference, baseline = reference_predictions({name: getattr(data, name) for name in data.__dataclass_fields__})
    np.testing.assert_allclose(reference, result["predictions"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(baseline, result["baseline_predictions"], rtol=1e-10, atol=1e-12)


def test_more_pairs_in_one_pause_cannot_multiply_that_pauses_weight():
    data = fixture(effect=1)
    repeats = np.where((data.animal == "rat0") & (data.pause == "pause0"), 3, 1)
    indices = np.repeat(np.arange(len(data.pre)), repeats)
    fields = {name: getattr(data, name)[indices] for name in data.__dataclass_fields__}
    # New synthetic IDs distinguish copies without altering their measurements.
    occurrence = np.concatenate([np.arange(r) for r in repeats]) * 10000
    fields["unit_a"] = fields["unit_a"] + occurrence
    fields["unit_b"] = fields["unit_b"] + occurrence
    one = prediction_check(data)
    many = prediction_check(PairData(**fields))
    for metric in ("original_mean_animal_gain", "original_minus_animal_median_shuffle_gain"):
        assert np.isclose(one["summary"][metric], many["summary"][metric], rtol=1e-10, atol=1e-12)


def test_negative_order_effect_is_not_mislabeled_as_positive_update():
    result = prediction_check(fixture(effect=-2))
    assert result["summary"]["original_mean_animal_gain"] > 1
    assert all(x["order_coefficient_original_units"] < 0 for x in result["folds"] if x["condition"] == 0)
