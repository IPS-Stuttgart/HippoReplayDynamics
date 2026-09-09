from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.metric_population_recovery import population_fit
from hipporeplayimm.unknown_path_clocks import MODELS, SCENARIOS, TEACHERS, exact_bin_average, generate_counts, rng, score_batch, summarize


def test_exact_piecewise_integration_and_regrouping():
    from scripts.verify_unknown_path_clock_populations import integral_bins

    t = np.array([0, 0.1, 0.6, 0.9, 1.0])
    values = np.c_[np.ones(5) * 7, t * 4 + 1]
    fine = exact_bin_average(t, values, 20)
    np.testing.assert_allclose(fine[:, 0], 7)
    np.testing.assert_allclose(fine[:, 1], 1 + (np.arange(20) + 0.5) / 5)
    np.testing.assert_allclose(fine.reshape(5, 4, 2).mean(axis=1), exact_bin_average(t, values, 5))
    irregular = np.array([[1, 2], [3, 1], [5, 9], [2, 6], [4, 2]])
    np.testing.assert_allclose(exact_bin_average(t, irregular, 7), integral_bins(t, irregular, 7))
    for clock in (t[::-1], t + 0.1, t * 0, np.array([0, 0.1, np.nan, 0.9, 1])):
        with pytest.raises(ValueError):
            exact_bin_average(clock, values, 5)


def test_unknown_path_likelihood_matches_exhaustive_latent_sum():
    p = np.array([[[0.8, 0.2], [0.4, 0.6]], [[0.1, 0.9], [0.7, 0.3]]])
    q = p[:, ::-1]
    x = np.array([[[2, 0], [0, 3]]])
    static = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    scores = score_batch(x, np.log(p), np.log(q), np.log(static), (1, 2))
    for h in (1, 2):
        for name, bank in (("physical", p[:h]), ("neural", q[:h])):
            expected = np.mean([np.prod(np.power(row, x[0])) for row in bank])
            assert scores[h][0, MODELS.index(name)] == pytest.approx(np.log(expected))
            reset = np.mean([np.prod(bank[list(indices), np.arange(2)] ** x[0]) for indices in product(range(h), repeat=2)])
            assert scores[h][0, MODELS.index(name + "_reset")] == pytest.approx(np.log(reset))
    expected = np.mean([np.prod(row ** x[0].sum(axis=0)) for row in static])
    assert scores[2][0, 2] == pytest.approx(np.log(expected))
    best = max(np.sum(x[0] * np.log(row)) for row in p)
    assert scores[2][0, 0] < best


def test_likelihood_is_normalized_and_counts_only():
    p = np.array([[[0.8, 0.2], [0.4, 0.6]], [[0.1, 0.9], [0.7, 0.3]]])
    x = np.array([np.eye(2, dtype=int)[list(labels)] for labels in product(range(2), repeat=2)])
    scores = score_batch(x, np.log(p), np.log(p[:, ::-1]), np.log(p[:, 0]), (2,))[2]
    np.testing.assert_allclose(logsumexp(scores, axis=0), 0, atol=1e-12)
    empty = score_batch(x * 0, np.log(p), np.log(p), np.log(p[:, 0]), (2,))[2]
    np.testing.assert_allclose(empty, 0, atol=1e-12)
    for supports in ((), (0,), (3,), (2, 2)):
        with pytest.raises(ValueError):
            score_batch(x, np.log(p), np.log(p), np.log(p[:, 0]), supports)
    with pytest.raises(ValueError):
        score_batch(x, np.log(p) + 1, np.log(p), np.log(p[:, 0]), (2,))


def test_new_nuisance_model_profile_and_old_api_are_consistent():
    labels = np.repeat(np.arange(5), [90, 30, 40, 20, 20])
    ll = np.where(labels[:, None] == np.arange(5), 0.0, -800.0)
    fit = population_fit(ll, MODELS, "coherent")
    assert fit["phi_hat"] == pytest.approx(0.25, abs=1e-5)
    assert fit["coherent_weight"] == pytest.approx(0.6, abs=1e-5)
    for model, w in zip(MODELS, [0.45, 0.15, 0.2, 0.1, 0.1], strict=True):
        assert fit["weight_" + model] == pytest.approx(w, abs=1e-5)
    unidentifiable = population_fit(np.zeros((100, 5)), MODELS, "coherent")
    assert unidentifiable["phi_low"] == 0 and unidentifiable["phi_high"] == 1


def test_generators_preserve_counts_and_reset_breaks_shared_path():
    generator = rng("test")
    p = generator.dirichlet(np.ones(4), size=(12, 30))
    q = generator.dirichlet(np.ones(4), size=(12, 30))
    static = generator.dirichlet(np.ones(4), size=9)
    for model in MODELS:
        counts, latent = generate_counts(np.arange(30), model, p, q, static, rng(model))
        np.testing.assert_array_equal(counts.sum(axis=1), np.arange(30))
        assert len(np.unique(latent)) > 1 if model.endswith("reset") else len(np.unique(latent)) == 1
        again, other = generate_counts(np.arange(30), model, p, q, static, rng(model))
        np.testing.assert_array_equal(counts, again)
        np.testing.assert_array_equal(latent, other)


def test_summary_requires_all_conditions_and_calibrated_false_claims():
    rows = [
        {"dataset": d, "scenario": q, "repeat": r, "teacher": t, "support": 256, "phi_hat": q, "phi_low": q - 0.02, "phi_high": q + 0.02, "coherent_weight": 0.6}
        for d, q, r, t in product(("pfeiffer_foster", "tanni2022"), SCENARIOS, range(50), TEACHERS)
    ]
    frame = pd.DataFrame(rows)
    _, _, gates = summarize(frame, supports=(256,))
    assert gates.practical_pass.all()
    for bad in (frame.iloc[:-1], frame.iloc[:0], pd.concat([frame, frame.iloc[:1]])):
        with pytest.raises(ValueError):
            summarize(bad, supports=(256,))
    frame.loc[frame.scenario.eq(0.5) & frame.repeat.lt(3), ["phi_hat", "phi_low", "phi_high"]] = [0.8, 0.7, 0.9]
    _, _, gates = summarize(frame, supports=(256,))
    assert gates.null_false_direction_fraction.eq(0.06).all()
    assert not gates.practical_pass.any()


def test_small_record_pipeline_has_no_supplied_path_in_scorer(tmp_path):
    from scripts.run_unknown_path_clock_populations import record_task
    from scripts.verify_unknown_path_clock_populations import record_audit

    source, out = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    out.mkdir()
    centers = np.array(list(product(np.arange(0, 81, 8), repeat=2)))
    generator = rng("fixture")
    peaks = generator.uniform(0, 80, (8, 2))
    rates = 0.1 + 12 * np.exp(-np.sum((peaks[:, None] - centers) ** 2, axis=2) / 400)
    counts = np.array([[3, 2, 0, 1, 0, 1, 0, 2], [2, 0, 3, 0, 1, 0, 1, 0], [1, 0, 2, 0, 0, 1, 0, 0], [1, 0, 2, 3, 1, 0, 1, 0]])
    np.savez(source / "test_cache.npz", centers=centers, rates=rates, counts_0=counts, edges_0=np.arange(5) * 0.02)
    item = {"tag": "test", "dataset": "synthetic", "animal": "rat", "session": "run"}
    result = record_task(item, source, out, n_paths=4, repeats=1, events=3, supports=(2, 4))
    assert result["n_observations"] == 18 and result["n_rows"] == 36
    scores = pd.read_csv(out / "test_r000_scores.csv.gz")
    assert not scores.duplicated(["scenario", "teacher", "event_in_population", "support"]).any()
    with np.load(out / "test_r000_observations.npz") as z:
        assert len(z["offsets"]) == 19
        np.testing.assert_array_equal(z["counts"].sum(axis=1), np.tile(counts.sum(axis=1), 18))
    audit = record_audit(item, source, out, n_paths=4, repeats=1, events=3, supports=(2, 4))
    assert audit["observations"] == 18 and audit["likelihoods_checked"] == 180
    assert audit["max_score_error"] < 1e-8
