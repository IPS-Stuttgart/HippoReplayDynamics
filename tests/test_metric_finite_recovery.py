"""Synthetic recovery checks; no external dataset is required."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, logsumexp

from hipporeplayimm.metric_finite_recovery import (
    CONDITIONS,
    GENERATORS,
    MODELS,
    ORIGINS,
    SUPPORTS,
    TRIALS,
    decoded_origin,
    null_threshold,
    observation_stress,
    origin_target_indices,
    predict,
    rng_for,
    sample_identities,
    sample_path,
    score_batch,
)
from scripts.run_2d_metric_finite_recovery import aggregate, calibrate, reduce_trials


def toy():
    rates = np.array([[9.0, 1, 1, 2], [1, 9, 2, 1], [2, 1, 9, 1], [8, 1, 2, 1], [1, 8, 1, 2], [1, 2, 8, 1]])
    a = np.array([[0.5, 0.25, 0, 0.25], [0.25, 0.5, 0.25, 0], [0, 0.25, 0.5, 0.25], [0.25, 0, 0.25, 0.5]])
    b = 0.5 * np.eye(4) + 0.5 / 4
    return rates, {"physical": a @ a, "neural": b @ b}


def manual_ll(counts, rates):
    p = rates / rates.sum(axis=0)
    return counts @ np.log(p) + gammaln(counts.sum() + 1) - gammaln(counts + 1).sum()


def test_scores_against_independent_dense_calculation():
    rates, kernels = toy()
    x = np.array([[3, 1, 0], [0, 2, 1], [1, 1, 1], [0, 0, 3], [0, 0, 0], [2, 1, 0], [0, 1, 2]])
    y = x[:, ::-1].copy()
    states = np.array([0, 1, 2, 3, 1, 2, 0])
    offsets = [0, 4, 7]
    rows = score_batch(x, y, rates[:3], rates[3:], states, offsets, kernels)
    for row in rows:
        lo, hi = offsets[row["simulation_index"] : row["simulation_index"] + 2]
        for model in MODELS:
            expected = 0.0
            for t in range(lo, hi - 2):
                origin = np.eye(4)[states[t]]
                if row["origin"] == "decoded":
                    ll = manual_ll(x[t], rates[:3])
                    origin = np.exp(ll - logsumexp(ll))
                transition = kernels.get(model, np.eye(4) if model == "stationary" else np.ones((4, 4)) / 4)
                q = origin @ transition
                target_ll = manual_ll(y[t + 2], rates[3:])
                expected += np.log(np.sum(q * np.exp(target_ll)))
            assert row["score_" + model] == pytest.approx(expected, abs=1e-12)


def test_future_training_cannot_change_forecasts_or_scores():
    rates, kernels = toy()
    x = np.array([[4, 0, 1], [1, 2, 1], [1, 0, 1]])
    y = x.copy()
    states = np.array([0, 1, 2])
    base = score_batch(x, y, rates[:3], rates[3:], states, [0, 3], kernels)
    changed = x.copy()
    changed[1:] += 200
    assert base == score_batch(changed, y, rates[:3], rates[3:], states, [0, 3], kernels)


def test_held_mutation_cannot_change_predictions(monkeypatch):
    import hipporeplayimm.metric_finite_recovery as m

    rates, kernels = toy()
    x = np.array([[4, 0, 1], [1, 2, 1], [1, 0, 1]])
    saved = []
    original = m.mixture_scores

    def capture(q, ll):
        saved.append(q.copy())
        return original(q, ll)

    monkeypatch.setattr(m, "mixture_scores", capture)
    first = score_batch(x, x, rates[:3], rates[3:], np.arange(3), [0, 3], kernels)
    second = score_batch(x, x + 5, rates[:3], rates[3:], np.arange(3), [0, 3], kernels)
    for a, b in zip(saved[:8], saved[8:], strict=True):
        np.testing.assert_array_equal(a, b)
    assert first != second


def test_zero_targets_have_zero_score_and_uniform_unobserved_origin():
    rates, kernels = toy()
    x = np.zeros((3, 3), dtype=int)
    np.testing.assert_allclose(decoded_origin(x, rates[:3]), 0.25)
    rows = score_batch(x, x, rates[:3], rates[3:], np.arange(3), [0, 3], kernels)
    assert all(row["score_" + m] == 0 for row in rows for m in MODELS)


def test_generation_preserves_totals_and_seed():
    rates, _ = toy()
    states = np.array([0, 1, 2, 3])
    totals = np.array([0, 2, 5, 30])
    a = sample_identities(rates, np.arange(3), states, totals, rng_for("toy"))
    b = sample_identities(rates, np.arange(3), states, totals, rng_for("toy"))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a.sum(axis=1), totals)
    assert a.dtype == np.int32
    stationary = sample_path(np.eye(4), 20, rng_for("stationary"))
    assert len(set(stationary)) == 1
    with pytest.raises(ValueError):
        sample_identities(rates, np.arange(3), states, totals + 0.5, rng_for("toy"))


def test_stress_reproducible_preserves_mean_map_rates():
    rates, _ = toy()
    centers = np.array([[0, 0], [8, 0], [8, 8], [0, 8]])
    gain, perturbed = observation_stress(rates, centers, "toy")
    again = observation_stress(rates, centers, "toy")
    np.testing.assert_array_equal(gain, again[0])
    np.testing.assert_array_equal(perturbed, again[1])
    np.testing.assert_allclose(perturbed.mean(axis=1), rates.mean(axis=1))
    assert (gain > 0).all() and not np.allclose(perturbed, rates)


@pytest.mark.parametrize("offsets", [[], [0, 2], [1, 4], [0, 3.5], [0, np.nan]])
def test_invalid_offsets(offsets):
    with pytest.raises(ValueError):
        origin_target_indices(offsets)


def test_forecast_validation():
    with pytest.raises(ValueError):
        predict(np.ones((2, 4)), {"physical": np.eye(4), "neural": np.eye(4)})
    rates, kernels = toy()
    with pytest.raises(ValueError):
        score_batch(np.ones((3, 3)), np.ones((4, 3)), rates[:3], rates[3:], np.arange(3), [0, 3], kernels)


def test_conformal_threshold():
    assert null_threshold(np.arange(32)) == 31
    assert np.isinf(null_threshold(np.arange(10)))
    assert null_threshold(np.full(32, -np.inf)) == -np.inf
    with pytest.raises(ValueError):
        null_threshold([])
    with pytest.raises(ValueError):
        null_threshold([np.nan])


@pytest.fixture(scope="module")
def score_fixture():
    rows = []
    for dataset, generator, trial, condition, support, origin, split in product(("pf", "tanni"), GENERATORS, range(TRIALS), CONDITIONS, SUPPORTS, ORIGINS, range(5)):
        scores = dict.fromkeys(MODELS, -10.0)
        if generator in ("physical", "neural"):
            scores.update(physical=-3.0, neural=-3.0)
            scores[generator] = -2.0
        rows.append(
            {
                "dataset": dataset,
                "animal": dataset + "_rat",
                "session": "run1",
                "generator": generator,
                "trial": trial,
                "phase": "calibration" if trial < 32 else "evaluation",
                "condition": condition,
                "support": support,
                "origin": origin,
                "split": split,
                "n_held_target_spikes": 10,
                "n_train_origin_spikes": 20,
                **{"score_" + m: v for m, v in scores.items()},
            }
        )
    return pd.DataFrame(rows)


def test_positive_and_ambiguous_recovery(score_fixture):
    result = aggregate(score_fixture)
    assert result[6].native_matched_ready.item()
    assert result[6].robust_native_ready.item()
    assert not result[6].biological_mechanism_established.item()
    nulls = result[0][result[0].generator.isin(("stationary", "iid"))]
    assert nulls.winner.eq("ambiguous").all()
    assert not nulls.structured_detected.any()
    changed = score_fixture.copy()
    changed[["score_" + m for m in MODELS]] = 0.0
    failed = aggregate(changed)
    assert not failed[6].native_matched_ready.item()
    assert failed[4].query("metric == 'balanced_metric_accuracy'")["mean"].eq(0.5).all()


@pytest.mark.parametrize("failure", ["row", "arm", "split", "phase", "nan", "positive", "empty"])
def test_missing_or_invalid_factors_fail(score_fixture, failure):
    frame = score_fixture.copy()
    if failure == "row":
        frame = frame.iloc[1:]
    elif failure == "arm":
        frame = frame[frame.support.ne(4)]
    elif failure == "split":
        frame = frame[frame.split.ne(4)]
    elif failure == "phase":
        frame.loc[0, "phase"] = "evaluation"
    elif failure in ("nan", "positive"):
        frame.loc[0, "score_physical"] = np.nan if failure == "nan" else 1
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        reduce_trials(frame)


def test_evaluation_cannot_change_calibration(score_fixture):
    trials = reduce_trials(score_fixture)
    _, baseline = calibrate(trials)
    changed = trials.copy()
    changed.loc[changed.phase.eq("evaluation"), "structure_statistic"] = 1000
    _, after = calibrate(changed)
    pd.testing.assert_frame_equal(baseline, after)


def test_uninformative_controls_abstain(score_fixture):
    frame = score_fixture.copy()
    frame["n_held_target_spikes"] = 0
    frame[["score_" + m for m in MODELS]] = 0.0
    trials, thresholds = calibrate(reduce_trials(frame))
    assert not thresholds.finite_calibration.any()
    assert not trials.structured_detected.any()
