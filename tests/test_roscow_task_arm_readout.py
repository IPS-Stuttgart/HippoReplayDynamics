import numpy as np
import pandas as pd
import pytest

from scripts.validate_roscow_task_arm_readout import (
    READOUTS,
    count_windows,
    fit_rates,
    leave_one_out,
    score_session,
    scores,
    summarize_animals,
)


def synthetic(labels):
    counts = np.ones((len(labels), 6), dtype=int)
    counts[np.arange(len(labels)), labels] += 24
    return counts


def test_recover_arm_composition_not_population_total():
    labels = np.random.default_rng(123).permutation(np.repeat(np.arange(3), 10))
    metrics, prob = score_session(synthetic(labels), labels, 1.75)
    assert scores(prob["poisson"], labels)["balanced_accuracy"] == 1
    assert scores(prob["composition"], labels)["balanced_accuracy"] == 1
    assert scores(prob["count_only"], labels)["balanced_accuracy"] == pytest.approx(1 / 3)
    null = metrics[(metrics["shift"] > 0) & (metrics.readout == "composition")]
    assert null.balanced_accuracy.median() < 0.5


def test_population_gain_alone_cannot_pass_composition():
    labels = np.repeat(np.arange(3), 8)
    counts = np.repeat(((labels + 1) * 12)[:, None], 6, axis=1)
    prob = leave_one_out(counts, labels, 1.75)
    assert scores(prob["count_only"], labels)["balanced_accuracy"] > 0.9
    assert scores(prob["composition"], labels)["balanced_accuracy"] == pytest.approx(1 / 3)


def test_zero_spikes_are_not_silently_dropped_or_assigned_success():
    labels = np.repeat(np.arange(3), 3)
    prob = leave_one_out(np.zeros((9, 5)), labels, 1.75)
    for kind in READOUTS:
        assert np.allclose(prob[kind], 1 / 3)
        assert scores(prob[kind], labels)["balanced_accuracy"] == pytest.approx(1 / 3)


def test_no_test_trial_enters_rate_fit(monkeypatch):
    import scripts.validate_roscow_task_arm_readout as module

    labels = np.repeat(np.arange(3), 3)
    counts = synthetic(labels)
    counts[:, -1] = np.arange(9) + 100
    called = []
    original = module.fit_rates

    def audited_fit(c, lab, exposure):
        missing = set(counts[:, -1]) - set(c[:, -1])
        assert len(c) == 8 and len(missing) == 1
        called.append(missing.pop())
        return original(c, lab, exposure)

    monkeypatch.setattr(module, "fit_rates", audited_fit)
    leave_one_out(counts, labels, 1.75)
    assert called == list(counts[:, -1])


def test_training_classes_required():
    with pytest.raises(ValueError, match="three arms"):
        fit_rates(np.ones((4, 5)), np.array([0, 0, 1, 1]), 1.75)


def test_counting_half_open_prearrival_windows():
    counts = count_windows([np.array([8.0, 9.749, 9.75, 10.0, 18.0, 19.0])], np.array([10.0, 20.0]), [0, 30])
    assert counts[:, 0].tolist() == [2, 2]
    with pytest.raises(ValueError, match="overlapping"):
        count_windows([np.array([8.0])], np.array([10.0, 11.0]), [0, 30])


def test_conservative_null_preserves_periodic_action_sequence():
    labels = np.tile(np.arange(3), 4)
    metrics, _ = score_session(synthetic(labels), labels, 1.75)
    assert (metrics.loc[metrics.readout == "composition", "balanced_accuracy"] == 1).all()


def test_animal_summary_equal_session_weights_and_missing_animals_fail():
    rows = []
    for session, n, actual in [(1, 12, 0.6), (2, 30, 0.9)]:
        for kind in READOUTS:
            for shift in range(n):
                rows.append({"animal": "Quirinius", "session": session, "shift": shift,
                             "readout": kind, "balanced_accuracy": actual if shift == 0 else 0.3,
                             "mean_log_score_above_chance": 0.1 if shift == 0 else -0.1})
    summary = summarize_animals(pd.DataFrame(rows))
    q = summary[summary.animal == "Quirinius"]
    assert np.allclose(q.mean_session_balanced_accuracy, 0.75)
    assert q.screen_pass.all()
    assert not summary[summary.animal != "Quirinius"].screen_pass.any()
    pd.testing.assert_frame_equal(summary, summarize_animals(pd.DataFrame(rows)))
