from itertools import pairwise, product

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.metric_finite_recovery import score_batch
from hipporeplayimm.metric_oracle_recovery import lagged_evidence, path_evidence, whole_evidence
from scripts.run_2d_metric_oracle_recovery import MODELS, OBSERVED, aggregate, classify, validate


def fixture():
    rng = np.random.default_rng(705)
    a, b = [rng.dirichlet(np.ones(3) * 2, 3) for _ in range(2)]
    ll = np.log(rng.uniform(0.01, 0.9, (7, 3)))
    return ll, np.array([0, 3, 7]), {"physical": a, "neural": b}


def enumerated(ll, a):
    n = ll.shape[1]
    values = []
    for path in product(range(n), repeat=len(ll)):
        v = -np.log(n) + sum(ll[t, state] for t, state in enumerate(path))
        for i, j in pairwise(path):
            if a[i, j] == 0:
                v = -np.inf
                break
            v += np.log(a[i, j])
        values.append(v)
    return logsumexp(values)


def test_whole_evidence_enumeration_and_hmmlearn():
    from hmmlearn import _hmmc

    ll, offsets, kernels = fixture()
    result = whole_evidence(ll, offsets, kernels)
    controls = kernels | {"stationary": np.eye(3), "iid": np.ones((3, 3)) / 3}
    for model, a in controls.items():
        for i, (start, end) in enumerate(pairwise(offsets)):
            block = ll[start:end]
            assert result[model][i] == pytest.approx(enumerated(block, a), abs=1e-12)
            maximum = block.max(axis=1)
            z, _, _ = _hmmc.forward_scaling(np.ones(3) / 3, a, np.exp(block - maximum[:, None]))
            assert result[model][i] == pytest.approx(z + maximum.sum(), abs=1e-12)


def test_sequence_resets_scaling_and_no_spikes():
    ll, offsets, kernels = fixture()
    together = whole_evidence(ll, offsets, kernels)
    separately = whole_evidence(ll[3:], [0, 4], kernels)
    for m in MODELS:
        assert together[m][1] == pytest.approx(separately[m][0])
        shifted = whole_evidence(ll - 1000, offsets, kernels)[m]
        np.testing.assert_allclose(shifted, together[m] - 1000 * np.diff(offsets), atol=1e-10)
        zero = whole_evidence(np.zeros_like(ll), offsets, kernels)[m]
        np.testing.assert_allclose(zero, 0, atol=1e-12)


def test_latent_path_probabilities():
    _, offsets, kernels = fixture()
    states = np.array([0, 0, 0, 0, 1, 2, 1])
    z = path_evidence(states, offsets, kernels)
    assert z["stationary"][0] == pytest.approx(-np.log(3))
    assert np.isneginf(z["stationary"][1])
    for m in MODELS[:2]:
        expected = -np.log(3) + np.log(kernels[m][0, 1]) + np.log(kernels[m][1, 2]) + np.log(kernels[m][2, 1])
        assert z[m][1] == pytest.approx(expected)
    with pytest.raises(ValueError):
        path_evidence(states + 0.1, offsets, kernels)


def test_full_information_can_distinguish_generators():
    a = np.full((4, 4), 0.01)
    b = a.copy()
    a[np.arange(4), (np.arange(4) + 1) % 4] = 0.97
    b[np.arange(4), (np.arange(4) - 1) % 4] = 0.97
    path = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    ll = np.full((8, 4), -30.0)
    ll[np.arange(8), path] = 0
    scores = whole_evidence(ll, [0, 8], {"physical": a, "neural": b})
    assert scores["physical"][0] - scores["neural"][0] > 20


def test_lagged_reproduces_saved_scorer_without_latent_path_input():
    rng = np.random.default_rng(26)
    rates = rng.gamma(2, 3, (6, 4))
    x, y = rng.poisson(1, (7, 3)), rng.poisson(2, (7, 3))
    kernels = {m: rng.dirichlet(np.ones(4), 4) for m in MODELS[:2]}
    squared = {m: a @ a for m, a in kernels.items()}
    got, info = lagged_evidence(x, y, rates[:3], rates[3:], [0, 3, 7], squared)
    reference = score_batch(x, y, rates[:3], rates[3:], np.zeros(7, int), [0, 3, 7], squared)
    reference = [r for r in reference if r["origin"] == "decoded"]
    assert info.all()
    for m in MODELS:
        np.testing.assert_allclose(got[m], [r["score_" + m] for r in reference], atol=1e-12)


@pytest.mark.parametrize("error", ["offset", "nan", "positive", "matrix", "empty"])
def test_bad_inputs_fail(error):
    ll, offsets, kernels = fixture()
    if error == "offset":
        offsets = [0, 2, 7]
    elif error == "nan":
        ll[0, 0] = np.nan
    elif error == "positive":
        ll[0, 0] = 1
    elif error == "matrix":
        kernels["physical"] *= 2
    else:
        kernels = {}
    with pytest.raises(ValueError):
        whole_evidence(ll, offsets, kernels)


@pytest.fixture(scope="module")
def score_fixture():
    factors = [("latent_path", 0, -1)] + list(product(OBSERVED, (1, 4), range(5)))
    rows = []
    for dataset, g, trial, (method, support, split) in product(("pf", "tanni"), MODELS, range(64), factors):
        scores = dict.fromkeys(MODELS, -10.0)
        if g in MODELS[:2]:
            scores.update(physical=-3.0, neural=-3.0)
            scores[g] = -2.0
        rows.append(
            {
                "dataset": dataset,
                "animal": dataset + "_rat",
                "session": "day1",
                "generator": g,
                "trial": trial,
                "method": method,
                "support": support,
                "split": split,
                "phase": "calibration" if trial < 32 else "evaluation",
                "informative": True,
                "score_kind": "latent_path_log_probability"
                if method == "latent_path"
                else "joint_event_log_probability"
                if method.startswith("whole_")
                else "sum_40ms_predictive_log_scores",
                **{"score_" + m: v for m, v in scores.items()},
            }
        )
    return pd.DataFrame(rows)


def test_readiness_and_nonvacuous_failure(score_fixture):
    output = aggregate(score_fixture)
    assert output[-1].whole_event_oracle_native_operating_pass.item()
    assert not output[-1].new_real_scoring_authorized.item()
    broken = score_fixture.copy()
    broken[["score_" + m for m in MODELS]] = 0
    failed = aggregate(broken)
    assert not failed[-1].whole_event_oracle_native_operating_pass.item()


def test_average_decisions_not_median_likelihood(score_fixture):
    frame = score_fixture.copy()
    mask = frame.method.eq("whole_exact") & frame.generator.eq("neural")
    frame.loc[mask & frame.split.lt(3), "score_neural"] = -2.9
    frame.loc[mask & frame.split.ge(3), "score_neural"] = -20.0
    _, _, trials = classify(frame)
    affected = trials[trials.method.eq("whole_exact") & trials.generator.eq("neural")]
    assert affected.binary_accuracy.eq(0.6).all()
    assert affected.n_realizations.eq(5).all()


def test_evaluation_cannot_set_thresholds(score_fixture):
    _, before, _ = classify(score_fixture)
    changed = score_fixture.copy()
    changed.loc[changed.phase.eq("evaluation"), "score_neural"] = -1
    _, after, _ = classify(changed)
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("change", ["row", "arm", "split", "phase", "nan", "minus_inf", "empty"])
def test_coverage_and_probability_errors(score_fixture, change):
    frame = score_fixture.copy()
    if change == "row":
        frame = frame.iloc[1:]
    elif change == "arm":
        frame = frame[frame.method.ne("whole_exact")]
    elif change == "split":
        frame = frame[frame.split.ne(2)]
    elif change == "phase":
        frame.loc[0, "phase"] = "evaluation"
    elif change in ("nan", "minus_inf"):
        frame.loc[0, "score_neural"] = np.nan if change == "nan" else -np.inf
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        validate(frame)
