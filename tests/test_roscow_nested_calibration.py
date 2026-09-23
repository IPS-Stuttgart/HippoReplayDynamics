import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_roscow_task_arm_readout import (
    METHODS,
    TEMPERATURES,
    aggregate,
    calibrated_session,
    choose_temperature,
    evaluate,
    fold_logits,
    log_probabilities,
)
from scripts.validate_roscow_task_arm_readout import READOUTS, leave_one_out


def fixture_counts():
    labels = np.random.default_rng(9).permutation(np.repeat(np.arange(3), 4))
    counts = np.ones((len(labels), 6), dtype=int)
    counts[np.arange(len(labels)), labels] += 20
    return counts, labels


def test_original_likelihood_unchanged():
    counts, labels = fixture_counts()
    original = leave_one_out(counts, labels, 1.75)
    prediction, temperatures = calibrated_session(counts, labels, 1.75)
    for kind in READOUTS:
        np.testing.assert_allclose(np.exp(prediction[kind, "uncalibrated"]), original[kind])
        assert np.isin(temperatures[kind], TEMPERATURES).all()


def test_outer_label_and_spikes_never_select_own_temperature():
    counts, labels = fixture_counts()
    _, before = fold_logits(counts, labels, 1.75)
    new_counts, new_labels = counts.copy(), labels.copy()
    new_counts[0] += 1000
    new_labels[0] = (new_labels[0] + 1) % 3
    _, after = fold_logits(new_counts, new_labels, 1.75)
    for kind in READOUTS:
        np.testing.assert_allclose(before[kind][0, 1:], after[kind][0, 1:])
        assert choose_temperature(before[kind][0, 1:], labels[1:]) == choose_temperature(after[kind][0, 1:], new_labels[1:])


def test_clear_population_code_retains_predictive_information():
    counts, labels = fixture_counts()
    prob, _ = calibrated_session(counts, labels, 1.75)
    for kind in ["poisson", "composition"]:
        score = evaluate(prob[kind, "nested_temperature"], labels)
        assert score["balanced_accuracy"] == 1
        assert score["mean_log_score_above_chance"] > 0.8


def test_zero_information_selects_uniform_not_spurious_confidence():
    counts, labels = fixture_counts()
    prob, temperature = calibrated_session(np.zeros_like(counts), labels, 1.75)
    for kind in READOUTS:
        assert np.isinf(temperature[kind]).all()
        assert evaluate(prob[kind, "nested_temperature"], labels)["mean_log_score_above_chance"] == pytest.approx(0)


def test_gain_only_remains_uninformative_in_composition():
    _, labels = fixture_counts()
    counts = np.repeat(((labels + 1) * 10)[:, None], 6, axis=1)
    prob, _ = calibrated_session(counts, labels, 1.75)
    score = evaluate(prob["composition", "nested_temperature"], labels)
    assert score["balanced_accuracy"] == pytest.approx(1 / 3)
    assert score["mean_log_score_above_chance"] == pytest.approx(0)


def test_log_space_preserves_extremely_bad_predictions():
    raw = np.array([[0.0, -2000, -5000]])
    p = log_probabilities(raw, 1)
    assert np.isfinite(p).all()
    assert p[0, 1] == -2000
    calibrated = log_probabilities(raw, 128)
    assert np.exp(calibrated[0, 1]) > 0
    assert np.argmax(calibrated) == np.argmax(raw)


def test_every_inner_fit_removes_both_outer_and_inner_trials(monkeypatch):
    import scripts.calibrate_roscow_task_arm_readout as module

    counts, labels = fixture_counts()
    counts[:, -1] = np.arange(len(labels)) + 100
    original = module.fit_rates
    sizes = []

    def record(x, y, exposure):
        sizes.append(len(x))
        return original(x, y, exposure)

    monkeypatch.setattr(module, "fit_rates", record)
    fold_logits(counts, labels, 1.75)
    assert sizes.count(11) == 12
    assert sizes.count(10) == 12 * 11 / 2


def test_session_equal_aggregation_and_shared_null_draws():
    rows = []
    for method in METHODS:
        for session, n, score in [(1, 12, 0.1), (2, 30, 0.5)]:
            for shift in range(n):
                rows.append({"animal": "test", "readout": "composition", "method": method,
                             "session": session, "shift": shift, "uniform_fraction": 0,
                             "mean_log_score_above_chance": score if shift == 0 else -0.1})
    result = aggregate(pd.DataFrame(rows))
    assert np.allclose(result.mean_session_log_score_above_chance, 0.3)
    assert len(set(result.null_p95)) == 1
    assert result.useful_probability_screen.all()
