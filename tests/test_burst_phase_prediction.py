from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.burst_phase_prediction import event_phase, fit_burst_phase, phase_weights, predict_burst_phase, predictive_score, score_orders
from scripts.audit_2d_burst_phase_prediction import summarize
from scripts.audit_2d_predictive_order_map import shuffled_indices
from scripts.verify_2d_burst_phase_prediction import audit_summaries, independent_order, score, weights


def test_phase_weights_partition_and_endpoint_clamping():
    w = phase_weights(np.array([0, 0.1, 0.2, 0.3, 0.9, 1]), 5)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    np.testing.assert_allclose(w[2], [0.5, 0.5, 0, 0, 0])
    assert w[0, 0] == w[1, 0] == w[-1, -1] == 1
    np.testing.assert_allclose(phase_weights([0, 0.5, 1], 1), 1)


def test_physical_clock_includes_partial_bin():
    edges = np.array([100, 100.02, 100.04, 100.047])
    times = (edges[:-1] + edges[1:]) / 2
    np.testing.assert_allclose(event_phase(times, edges), [0.01 / 0.047, 0.03 / 0.047, 0.0435 / 0.047])
    with pytest.raises(ValueError):
        event_phase(times + 0.001, edges)
    with pytest.raises(ValueError):
        phase_weights([-0.1, 0.5], 5)


def test_known_time_varying_composition_predicts_new_events():
    rng = np.random.default_rng(47)
    phase = (np.arange(20) + 0.5) / 20
    truth = np.array([[0.75, 0.15, 0.10] if t < 10 else [0.10, 0.15, 0.75] for t in range(20)])
    calibration = [np.array([rng.multinomial(10, p) for p in truth]) for _ in range(80)]
    fit = fit_burst_phase(calibration, [phase] * len(calibration), 5)
    test = np.array([rng.multinomial(40, p) for p in truth])
    global_score = predictive_score(test, np.tile(fit.global_probability, (len(test), 1)), np.arange(3))
    assert predict_burst_phase(test, phase, np.arange(3), fit) > global_score + 100
    assert predict_burst_phase(test, phase, np.arange(3), fit) > predict_burst_phase(test, phase[::-1], np.arange(3), fit) + 100


def test_constant_composition_has_no_phase_gain():
    x = np.tile([3, 2, 1], (20, 1))
    phase = (np.arange(20) + 0.5) / 20
    fit = fit_burst_phase([x] * 30, [phase] * 30, 5)
    a = predict_burst_phase(x, phase, np.arange(3), fit)
    b = predictive_score(x, np.tile(fit.global_probability, (20, 1)), np.arange(3))
    assert abs(a - b) < 0.05


def test_normalization_and_zero_counts():
    x = np.array([[2, 0], [1, 1], [0, 2]])
    p = np.tile([0.25, 0.75], (1, 1))
    total = sum(np.exp(predictive_score(v[None, :], p, np.arange(2))) for v in x)
    assert total == pytest.approx(1.0)
    assert predictive_score(np.zeros((2, 2), int), np.ones((2, 2)), np.arange(2)) == 0


def test_target_cannot_modify_fit_or_change_nonheld_score():
    rng = np.random.default_rng(5)
    x = rng.poisson(1, (20, 6))
    phase = (np.arange(20) + 0.5) / 20
    fit = fit_burst_phase([x], [phase])
    frozen = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in asdict(fit).items()}
    a = predict_burst_phase(x, phase, np.arange(3), fit)
    x[:, 3:] += 5
    assert predict_burst_phase(x, phase, np.arange(3), fit) == a
    for k, v in asdict(fit).items():
        np.testing.assert_equal(v, frozen[k])


def test_vectorized_permutations_match_separate_predictions():
    rng = np.random.default_rng(8)
    x = rng.poisson(0.5, (9, 4))
    p = rng.uniform(0.01, 1, (9, 4))
    orders = np.array([rng.permutation(len(x)) for _ in range(20)])
    held = np.array([0, 2, 3])
    actual = score_orders(x, p, held, orders)
    expected = [predictive_score(x[o], p, held) for o in orders]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    orders[0, :] = 0
    with pytest.raises(ValueError, match="permutations"):
        score_orders(x, p, held, orders)


def test_independent_math_and_parent_permutation_contract():
    rng = np.random.default_rng(91)
    phase = rng.uniform(0, 1, 30)
    for n in (3, 5, 10):
        np.testing.assert_allclose(weights(phase, n), phase_weights(phase, n), atol=1e-14)
    x = rng.poisson(0.6, (30, 7))
    p = rng.uniform(0.1, 1, (30, 7))
    held = np.array([2, 3, 6])
    assert score(x, p, held) == pytest.approx(predictive_score(x, p, held), abs=1e-10)
    identity = ("dataset", "rat", "session")
    np.testing.assert_array_equal(independent_order(identity, 5, 30, 4), shuffled_indices(identity, 5, 30, 4))


def summary_fixture():
    rows = []
    for dataset, nanimals in [("pfeiffer_foster", 4), ("tanni2022", 5)]:
        for animal in range(nanimals):
            for session in range(2):
                for event in range(2):
                    for split in range(5):
                        for n in (3, 5, 10):
                            ns = 4 if event == 0 else 0
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "animal": animal,
                                    "session": session,
                                    "event_id": event,
                                    "split": split,
                                    "fold": 0,
                                    "n_knots": n,
                                    "n_heldout_spikes": ns,
                                    "score_phase": -5 if ns else 0,
                                    "score_global": -7 if ns else 0,
                                    "score_phase_averaged": -5.5 if ns else 0,
                                    "score_first_order_imm": -3 if ns else 0,
                                    "score_iid_position": -3.5 if ns else 0,
                                    "score_learned_hmm": -4 if ns else 0,
                                    "score_same_emissions_iid": -4.5 if ns else 0,
                                    "mean_shuffle_phase": (-6 if ns else 0) if n == 5 else np.nan,
                                    "mean_shuffle_hmm": -6 if ns else 0,
                                }
                            )
    return pd.DataFrame(rows)


def test_event_medians_denominators_and_independent_summaries(tmp_path):
    original = summary_fixture()
    tables = summarize(original)
    for name, frame in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
        frame.to_csv(tmp_path / f"burst_phase_{name}.csv.gz", index=False)
    events = tables[1]
    assert events[events.event_id.eq(1)].delta_per_spike.isna().all()
    assert events[events.event_id.eq(1)].valid_neural_splits.eq(0).all()
    audit_summaries(tmp_path, original)


def test_missing_neuron_splits_and_models_fail():
    original = summary_fixture()
    with pytest.raises(ValueError, match="knot"):
        summarize(original.iloc[:-1])
    with pytest.raises(ValueError, match="neural split"):
        summarize(original[original.split.ne(4)])
    original.loc[0, "score_first_order_imm"] = np.nan
    with pytest.raises(ValueError, match="comparator"):
        summarize(original)
