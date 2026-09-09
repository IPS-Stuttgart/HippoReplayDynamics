from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from hipporeplayimm.metric_population_recovery import CONDITIONS, LR95, MODELS, SCENARIOS, population_fit, simplex_fit, summarize


def categorical_scores(counts):
    labels = np.repeat(np.arange(4), counts)
    return np.where(labels[:, None] == np.arange(4)[None, :], 0.0, -800.0)


def test_known_mixture_and_profile_binomial_interval():
    result = population_fit(categorical_scores([90, 30, 40, 40]))
    assert result["phi_hat"] == pytest.approx(0.25, abs=1e-5)
    for model, value in zip(MODELS, [0.45, 0.15, 0.2, 0.2], strict=True):
        assert result["weight_" + model] == pytest.approx(value, abs=1e-5)
    maximum = 30 * np.log(0.25) + 90 * np.log(0.75)

    def profile(q):
        return 2 * (maximum - 30 * np.log(q) - 90 * np.log1p(-q)) - LR95

    assert result["phi_low"] == pytest.approx(brentq(profile, 0.001, 0.25), abs=1e-5)
    assert result["phi_high"] == pytest.approx(brentq(profile, 0.25, 0.999), abs=1e-5)
    assert result["direction"] == "physical"


def test_identical_models_cannot_claim_mixture_direction():
    result = population_fit(np.zeros((100, 4)))
    assert result["phi_low"] == 0
    assert result["phi_high"] == 1
    assert result["null_lr"] < 1e-8
    assert result["direction"] == "undetermined"


def test_identical_moving_models_remain_unidentified_with_static_information():
    x = categorical_scores([40, 0, 30, 30])
    x[:, 1] = x[:, 0]
    result = population_fit(x)
    assert result["phi_low"] == 0 and result["phi_high"] == 1


def test_rowwise_log_likelihood_shifts_do_not_change_inference():
    x = categorical_scores([45, 45, 30, 30])
    a = population_fit(x)
    b = population_fit(x - np.arange(len(x))[:, None] * 1000)
    for key in ("phi_hat", "phi_low", "phi_high", "null_lr"):
        assert a[key] == pytest.approx(b[key], abs=1e-7)


@pytest.mark.parametrize("seed", [2, 17, 209])
def test_weak_overlapping_components_have_valid_profile_intervals(seed):
    rng = np.random.default_rng(seed)
    means = np.array([[0, 0], [0.1, 0], [0, 2], [2, 0]])
    labels = rng.choice(4, 256, p=[0.3, 0.3, 0.2, 0.2])
    observations = means[labels] + rng.standard_normal((256, 2))
    ll = -0.5 * ((observations[:, None] - means[None, :]) ** 2).sum(axis=2) - np.log(2 * np.pi)
    fit = population_fit(ll)
    assert 0 <= fit["phi_low"] <= fit["phi_hat"] + 1e-6
    assert fit["phi_hat"] <= fit["phi_high"] + 1e-6 <= 1 + 1e-6
    assert max(fit[k] for k in fit if k.endswith("kkt_error")) <= 2e-5


@pytest.mark.parametrize("x", [np.zeros((0, 4)), np.zeros((4, 3)), np.full((4, 4), np.nan), np.full((4, 4), -np.inf)])
def test_invalid_event_likelihoods_rejected(x):
    with pytest.raises(ValueError):
        population_fit(x)


def test_simplex_is_probability_normalized_and_has_optimality_certificate():
    x = np.array([[0.8, 0.15, 0.05], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    w, value, error = simplex_fit(np.log(x))
    assert w.sum() == pytest.approx(1)
    assert (w > 0).all() and error < 2e-5
    assert value == pytest.approx(np.log(x @ w).sum())
    with pytest.raises(ValueError):
        simplex_fit(np.log(x), [1, 1, 1])


def fake_fits(repeats=50):
    return pd.DataFrame(
        [
            {"dataset": d, "scenario": q, "repeat": r, "events_per_recording": 32, "condition": c, "phi_hat": q, "phi_low": q - 0.02, "phi_high": q + 0.02}
            for d, q, r, c in product(("pfeiffer_foster", "tanni2022"), SCENARIOS, range(repeats), CONDITIONS)
        ]
    )


def test_recovery_summary_and_nonvacuous_failure_cases():
    _, summary, gates = summarize(fake_fits(), sizes=(32,))
    assert gates.practical_pass.all() and summary.covered_fraction.eq(1).all()
    for frame in (fake_fits().iloc[0:0], fake_fits().iloc[:-1], pd.concat([fake_fits(), fake_fits().iloc[:1]])):
        with pytest.raises(ValueError):
            summarize(frame, sizes=(32,))


def test_equal_mixture_false_claims_fail_readiness():
    frame = fake_fits()
    mask = frame.scenario.eq(0.5) & frame.repeat.lt(3)
    frame.loc[mask, ["phi_hat", "phi_low", "phi_high"]] = [0.75, 0.65, 0.85]
    _, _, gates = summarize(frame, sizes=(32,))
    assert gates.null_false_direction_fraction.eq(0.06).all()
    assert not gates.practical_pass.any()


def test_fresh_record_simulation_preserves_native_group_counts(tmp_path):
    from hipporeplayimm.physical_neural_metric import matched_kernel, neural_cost, physical_cost
    from scripts.run_2d_metric_population_recovery import record_task

    source, metric, out = [tmp_path / p for p in ("source", "metric", "output")]
    for p in (source, metric, out):
        p.mkdir()
    tag = "synthetic"
    r = np.random.default_rng(145)
    centers = r.uniform(0, 40, (9, 2))
    rates = r.uniform(0.1, 10, (6, 9))
    train, held = np.arange(4), np.arange(4, 6)
    profile = np.array([[2, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1], [0, 2, 1, 0, 0, 0]])
    np.savez(source / f"{tag}_cache.npz", rates=rates, centers=centers, train_0=train, held_0=held, counts_0=profile, edges_0=np.arange(5) * 0.02)
    pars = {}
    for name, cost in (("physical", physical_cost(centers)), ("full_neural", neural_cost(rates)), ("train_neural_0", neural_cost(rates[train]))):
        k = matched_kernel(cost)
        for field in ("beta", "cost_scale", "log_scale", "entropy", "target_entropy"):
            pars[name + "__" + field] = getattr(k, field)
    np.savez(metric / f"{tag}_kernel_parameters.npz", **pars)
    item = {"dataset": "synthetic", "animal": "rat", "session": "run", "tag": tag}
    detail = record_task(item, source, metric, out, repeats=1, size=3)
    assert detail["rows"] == 27
    scores = pd.read_csv(out / f"{tag}_r000_scores.csv.gz")
    with np.load(out / f"{tag}_r000_observations.npz") as z:
        assert len(z["offsets"]) == 10
        for prefix in ("base", "gain"):
            for group, cells in (("train", train), ("held", held)):
                np.testing.assert_array_equal(z[f"{prefix}_{group}"].sum(axis=1), np.tile(profile[:, cells].sum(axis=1), 9))
    original = scores[scores.condition.eq("exact")].sort_values("simulation_index")
    estimated = scores[scores.condition.eq("train_geometry")].sort_values("simulation_index")
    for m in ("physical", "stationary", "iid"):
        np.testing.assert_allclose(original["score_" + m], estimated["score_" + m], atol=1e-9)
