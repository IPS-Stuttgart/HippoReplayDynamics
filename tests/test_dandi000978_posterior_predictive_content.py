import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from scripts.calibrate_dandi000978_posterior_predictive_content import (
    PARTITIONS,
    hierarchy_weights,
    likelihoods,
    partition_labels,
    score_readouts,
    summarize,
)


def data():
    rates = np.eye(4) * 3 + 0.25
    counts = np.eye(4, dtype=int) * 3
    return counts, rates


def test_soft_predictive_is_integral_not_expected_log():
    counts, rates = data()
    ca = counts + np.roll(counts, 1, axis=1)
    scores, _ = score_readouts(ca, counts, rates, rates)
    q = rates / rates.sum(axis=1, keepdims=True)
    pc, pp = softmax(ca @ np.log(q).T, axis=1), softmax(counts @ np.log(q).T, axis=1)
    np.testing.assert_allclose(scores["soft_predictive"][0], np.log(4 * pc @ pp.T))
    assert not np.allclose(scores["soft_predictive"], scores["soft_centered"])
    assert np.max(scores["soft_predictive"]) <= np.log(4) + 1e-12


def test_uniform_ca1_returns_no_predictive_information():
    counts, rates = data()
    scores, labels = score_readouts(np.zeros_like(counts), counts, rates, rates)
    np.testing.assert_allclose(scores["soft_predictive"], 0, atol=1e-12)
    np.testing.assert_array_equal(labels, np.full((1, 4), 5))


def test_zero_pfc_no_evidence_any_readout():
    counts, rates = data()
    scores, _ = score_readouts(counts, np.zeros_like(counts), rates, rates)
    for score in scores.values():
        np.testing.assert_allclose(score, 0, atol=1e-12)


def test_confident_reference_approaches_hard_prediction():
    counts, rates = data()
    scores, _ = score_readouts(counts * 100, counts, rates, rates)
    np.testing.assert_allclose(scores["soft_predictive"], scores["hard_predictive"], atol=1e-12)


def test_correct_shared_route_beats_independent_draw():
    counts, rates = data()
    scores, labels = score_readouts(counts, counts, rates, rates)
    assert np.trace(scores["soft_predictive"][0]) / 4 > scores["soft_predictive"].mean()
    np.testing.assert_array_equal(labels, np.zeros((1, 4), dtype=int))


def test_shared_wrong_estimate_does_not_become_correct_content():
    counts, rates = data()
    wrong = np.roll(counts, 1, axis=1)
    scores, labels = score_readouts(wrong, wrong, rates, rates)
    assert (np.diagonal(scores["soft_predictive"], axis1=1, axis2=2) > 0).all()
    np.testing.assert_array_equal(labels, np.full((1, 4), 3))


def test_exact_partition_categories():
    ca = np.array([[0, 1, 0, 0], [0, 0, 0, 0]])
    pf = np.array([[0, 0, 2, 0], [1, 1, 1, 1]])
    logca = np.full((2, 4, 4), -20.0)
    logpf = logca.copy()
    for block in range(2):
        for truth in range(4):
            logca[block, truth, ca[block, truth]] = 0
            logpf[block, truth, pf[block, truth]] = 0
    labels = partition_labels(logca, logpf)
    np.testing.assert_array_equal(labels[0], [0, 1, 2, 3])
    assert labels[1, 2] == 4
    logca[1, 3] = 0
    assert partition_labels(logca, logpf)[1, 3] == 5


@pytest.mark.parametrize("bad", [np.full((4, 4), -1), np.full((4, 4), 0.5), np.full((4, 4), np.nan)])
def test_invalid_counts_fail(bad):
    with pytest.raises(ValueError, match="counts"):
        likelihoods(bad, data()[1])


def test_bad_shapes_and_rates_fail():
    counts, rates = data()
    with pytest.raises(ValueError, match="blocks"):
        likelihoods(counts[:3], rates)
    with pytest.raises(ValueError, match="positive"):
        likelihoods(counts, rates * 0)


def test_hierarchy_does_not_overweight_repeats_or_epochs():
    blocks = pd.DataFrame({"file": ["x"] * 4, "heldout_epoch": [1, 1, 1, 2], "event_id": ["a", "a", "b", "c"]})
    w = hierarchy_weights(blocks)
    np.testing.assert_allclose(w.sum(), 1)
    np.testing.assert_allclose(w.sum(axis=1), [0.125, 0.125, 0.25, 0.5])


def test_stable_at_extreme_likelihoods():
    counts, rates = data()
    scores, _ = score_readouts(counts * 10000, np.roll(counts * 10000, 1, axis=1), rates, rates)
    assert all(np.isfinite(s).all() for s in scores.values())


def test_partitions_conserve_score_and_missing_designs(tmp_path):
    from scripts.calibrate_dandi000978_posterior_predictive_content import CONDITIONS

    counts, rates = data()
    scores, labels = score_readouts(counts, counts, rates, rates)
    blocks = pd.DataFrame({"animal": ["JS14"], "file": ["x"], "heldout_epoch": [1], "event_id": ["a"], "common_matched": [True]})
    bank, part = {}, {}
    for regime in ("native", "sleep_matched"):
        for condition in CONDITIONS:
            for name, values in scores.items():
                bank[f"{regime}_{condition}_{name}"] = values
            part[f"{regime}_{condition}"] = labels
    primary = pd.DataFrame({"animal": ["JS14"], "event_id": ["missing"]})
    (tmp_path / "simulations").mkdir()
    summarize(blocks, bank, part, primary, tmp_path)
    parts = pd.read_csv(tmp_path / "score_partition_summary.csv")
    signal = pd.read_csv(tmp_path / "signal_summary.csv")
    keys = ["animal", "regime", "support", "condition", "readout"]
    summed = parts.groupby(keys)[["weighted_fraction", "score_contribution"]].sum()
    np.testing.assert_allclose(summed.weighted_fraction, 1)
    np.testing.assert_allclose(summed.score_contribution, signal.set_index(keys).sort_index().mean_score)
    assert set(parts.partition) == set(PARTITIONS)
    powers = pd.read_csv(tmp_path / "conditional_sensitivity.csv")
    assert (powers.status == "unsupported").all()
