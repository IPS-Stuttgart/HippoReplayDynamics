import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from hipporeplayimm.literal_replay_clock import (
    SpatialRates,
    bin_average,
    clock_coordinates,
    coarse_grid,
    decode,
    multinomial_samples,
    normalize,
    path_log_score,
    recovery_credit,
    rng,
    sample_geometry,
    time_samples,
)
from scripts.run_literal_replay_clock_recovery import summarize


def fixture():
    x, y = np.meshgrid(np.arange(0, 161, 8), np.arange(0, 161, 8))
    centers = np.c_[x.ravel(), y.ravel()]
    fields = np.array([[20, 20], [50, 30], [80, 80], [140, 140]])
    rates = 0.1 + 10 * np.exp(-np.sum((fields[:, None] - centers) ** 2, axis=-1) / 600)
    return centers, rates


def test_clock_has_same_endpoints_and_different_timing():
    centers, rates = fixture()
    points, pr, _ = sample_geometry(SpatialRates(centers, rates), rng("test"), curved=True)
    s, c, clocks = clock_coordinates(points, pr)
    assert s[-1] > np.linalg.norm(points[-1] - points[0])
    assert c[-1] > 0
    assert not np.allclose(clocks["physical"], clocks["neural"])
    for clock in clocks.values():
        np.testing.assert_allclose(time_samples(clock, points, [0, 1]), points[[0, -1]])
    edges = np.interp(np.linspace(0, 1, 11), clocks["physical"], s)
    np.testing.assert_allclose(np.diff(edges), s[-1] / 10)
    code_edges = np.interp(np.linspace(0, 1, 11), clocks["neural"], c)
    np.testing.assert_allclose(np.diff(code_edges), c[-1] / 10)


def test_bin_average_integrates_not_midpoint():
    t = np.linspace(0, 1, 10001)
    values = np.c_[t, t**2]
    mean = bin_average(t, values, 1)
    np.testing.assert_allclose(mean, [[0.5, 1 / 3]], atol=2e-8)


def test_missing_coverage_not_bridged():
    centers, rates = fixture()
    mask = centers[:, 0] != 80
    spatial = SpatialRates(centers[mask], rates[:, mask])
    with pytest.raises(ValueError, match="supported"):
        spatial([[80, 80]])


def test_constant_population_geometry_is_unresolved():
    with pytest.raises(ValueError, match="degenerate"):
        clock_coordinates(np.c_[np.arange(5), np.zeros(5)], np.ones((5, 3)))


def test_counts_and_likelihood_match_multinomial():
    p = np.array([[0.3, 0.7], [0.9, 0.1]])
    x = multinomial_samples(p, [20, 0], rng("counts"))
    assert np.array_equal(x.sum(axis=1), [20, 0])
    full = np.sum(gammaln(x.sum(axis=1) + 1) - gammaln(x + 1).sum(axis=1) + (x * np.log(p)).sum(axis=1))
    coefficient = np.sum(gammaln(x.sum(axis=1) + 1) - gammaln(x + 1).sum(axis=1))
    assert path_log_score(x, p) == pytest.approx(full - coefficient)
    with pytest.raises(ValueError):
        multinomial_samples(p, [1.5, 0], rng("counts"))


def test_decoder_flat_prior_and_no_temporal_dependency():
    centers, rates = fixture()
    x = rng("obs").integers(0, 3, (5, len(rates)))
    mean, maximum, width = decode(x, rates, centers)
    np.testing.assert_allclose(decode(x[::-1], rates, centers)[0], mean[::-1])
    np.testing.assert_allclose(decode(np.zeros_like(x), rates, centers)[0], np.tile(centers.mean(axis=0), (5, 1)))
    assert np.isfinite(maximum).all() and (width > 0).all()


def test_grid_and_normalization_and_ties():
    centers, rates = fixture()
    coarse, coordinates = coarse_grid(rates, centers)
    assert coarse.shape == (len(rates), len(coordinates))
    assert len(coordinates) < len(centers)
    np.testing.assert_allclose(normalize(coarse.T).sum(axis=1), 1)
    assert recovery_credit(0, "physical") == 0.5
    assert recovery_credit(2, "neural") == 1
    assert recovery_credit(2, "physical") == 0


def test_summary_equal_animal_weight_and_reject_duplicates():
    rows = []
    for animal, sessions, correct in [("a", 3, 1), ("b", 1, 0)]:
        for session in range(sessions):
            for gen in ("physical", "neural"):
                rows.append(
                    {
                        "dataset": "test",
                        "animal": animal,
                        "session": session,
                        "path_id": 0,
                        "condition": "exact",
                        "generator": gen,
                        "repeat": 0,
                        "decoder_grid_cm": 8,
                        "oracle_correct": correct,
                        "delta_neural_minus_physical": 0,
                        "posterior_mean_error_cm": 0,
                        "decoded_mean_step_speed_cm_s": 1,
                        "true_arc_speed_cm_s": 1,
                        "clock_separation_rms_cm": 1,
                    }
                )
    table = pd.DataFrame(rows)
    summary = summarize(table)[-1]
    assert summary.oracle_correct.item() == 0.5
    assert not summary.oracle_practical_pass.item()
    with pytest.raises(ValueError):
        summarize(pd.concat([table, table]))
    with pytest.raises(ValueError):
        summarize(table.iloc[:0])


def test_end_to_end_independent_audit(tmp_path):
    from scripts.run_literal_replay_clock_recovery import record_task
    from scripts.verify_literal_replay_clock_recovery import record_audit

    centers, rates = fixture()
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    tag = "synthetic"
    np.savez_compressed(source / f"{tag}_cache.npz", rates=rates, centers=centers, counts_0=np.array([[1, 2, 3, 4], [3, 1, 2, 4], [5, 1, 1, 3]]), edges_0=np.arange(4) * 0.02)
    item = {"tag": tag, "dataset": "test", "animal": "one", "session": "run"}
    result = record_task(item, source, output, n_paths=2, repeats=1)
    audit = record_audit(item, source, output, n_paths=2, repeats=1)
    assert audit["score_rows_checked"] == result["n_rows"] == 16
    assert audit["quadrature_max_probability_error"] < 1e-4
    with np.load(output / f"{tag}_p000.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["counts_exact_physical_0"][0, 0] += 1
    np.savez_compressed(output / f"{tag}_p000.npz", **arrays)
    with pytest.raises(AssertionError):
        record_audit(item, source, output, n_paths=2, repeats=1)
