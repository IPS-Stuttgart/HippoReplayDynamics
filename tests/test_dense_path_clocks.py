from itertools import product

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.dense_path_clocks import PathAccumulator, RateIntegral, rng, summarize
from hipporeplayimm.unknown_path_clocks import exact_bin_average, score_batch


def test_reused_antiderivative_matches_reference_and_independent_integral():
    from scripts.verify_dense_path_clock_recovery import integral_bins as vectorized_integral
    from scripts.verify_unknown_path_clock_populations import integral_bins

    clock = np.array([0, 0.02, 0.15, 0.78, 1.0])
    values = rng("integration").uniform(0.01, 20, (5, 9))
    integral = RateIntegral(clock, values)
    for n in (1, 3, 7, 23):
        np.testing.assert_allclose(integral.average(n), exact_bin_average(clock, values, n))
        np.testing.assert_allclose(integral.average(n), integral_bins(clock, values, n))
        np.testing.assert_allclose(vectorized_integral(clock, values, n), integral_bins(clock, values, n))
    for invalid in (clock[::-1], clock + 0.1, np.zeros(5)):
        with pytest.raises(ValueError):
            RateIntegral(invalid, values)


def test_vectorized_audit_integral_matches_original_on_dense_and_coincident_knots():
    from scripts.verify_dense_path_clock_recovery import integral_bins as vectorized
    from scripts.verify_unknown_path_clock_populations import integral_bins as reference

    generator = rng("audit-integral")
    for n in (1, 3, 7, 20, 38):
        clock = np.unique(np.r_[0, generator.uniform(size=799), np.linspace(0, 1, n + 1)])
        values = generator.uniform(0.01, 20, (len(clock), 31))
        np.testing.assert_allclose(vectorized(clock, values, n), reference(clock, values, n), atol=1e-10, rtol=1e-12)


def test_streaming_sparse_likelihood_matches_full_dense_and_chunk_order():
    generator = rng("sparse")
    counts = generator.poisson(0.3, (13, 7, 9))
    p = generator.dirichlet(np.ones(9), size=(12, 7))
    q = generator.dirichlet(np.ones(9), size=(12, 7))
    static = generator.dirichlet(np.ones(9), size=20)
    reference = score_batch(counts, np.log(p), np.log(q), np.log(static), (4, 12))
    for mi, bank in enumerate((p, q)):
        acc = PathAccumulator(counts)
        with pytest.raises(ValueError):
            acc.scores()
        acc.add(np.log(bank[:4]))
        a, b = acc.scores()
        np.testing.assert_allclose(a, reference[4][:, mi])
        np.testing.assert_allclose(b, reference[4][:, mi + 3])
        acc.add(np.log(bank[4:9]))
        acc.add(np.log(bank[9:]))
        a, b = acc.scores()
        np.testing.assert_allclose(a, reference[12][:, mi])
        np.testing.assert_allclose(b, reference[12][:, mi + 3])
        reordered = PathAccumulator(counts)
        reordered.add(np.log(bank[::-1]))
        np.testing.assert_allclose(acc.scores(), reordered.scores())


def test_zero_counts_and_invalid_probabilities():
    acc = PathAccumulator(np.zeros((5, 4, 3), dtype=int))
    p = np.full((8, 4, 3), 1 / 3)
    acc.add(np.log(p))
    np.testing.assert_allclose(acc.scores(), 0, atol=1e-12)
    with pytest.raises(ValueError):
        acc.add(np.log(p) + 0.1)
    with pytest.raises(ValueError):
        PathAccumulator(np.full((1, 2, 3), 0.5))


def fake_fits():
    return pd.DataFrame(
        [
            {
                "dataset": d,
                "source_teacher": t,
                "scenario": q,
                "repeat": r,
                "candidate_bank": b,
                "support": h,
                "phi_hat": q,
                "phi_low": q - 0.02,
                "phi_high": q + 0.02,
                "coherent_weight": 0.6,
            }
            for d, t, q, r, b, h in product(("pfeiffer_foster", "tanni2022"), ("original_bank_0", "original_bank_1"), (0.25, 0.5, 0.75), range(50), (0, 1), (1024, 4096, 8192))
        ]
    )


def test_complete_recovery_and_nonvacuous_convergence():
    fits = fake_fits()
    _, _, gates, convergence = summarize(fits)
    assert gates.practical_pass.all() and convergence.integration_stable.all()
    fits.loc[fits.candidate_bank.eq(1) & fits.support.eq(8192), "phi_hat"] += 0.1
    _, _, _, convergence = summarize(fits)
    assert not convergence.integration_stable.any()
    for bad in (fits.iloc[:-1], fits.iloc[:0], fits[fits.candidate_bank.eq(0)]):
        with pytest.raises(ValueError):
            summarize(bad)


def test_small_streamed_record_reuses_parent_counts(tmp_path):
    from scripts.run_dense_path_clock_recovery import record_task
    from scripts.run_unknown_path_clock_populations import record_task as generate_parent
    from scripts.verify_dense_path_clock_recovery import record_audit

    source, parent, out = [tmp_path / p for p in ("source", "parent", "out")]
    for p in (source, parent, out):
        p.mkdir()
    centers = np.array(list(product(np.arange(0, 81, 8), repeat=2)))
    peaks = rng("fixture").uniform(0, 80, (8, 2))
    rates = 0.1 + 12 * np.exp(-np.sum((peaks[:, None] - centers) ** 2, axis=2) / 400)
    counts = np.array([[3, 2, 0, 1, 0, 1, 0, 2], [2, 0, 3, 0, 1, 0, 1, 0], [1, 0, 2, 0, 0, 1, 0, 0], [1, 0, 2, 3, 1, 0, 1, 0]])
    np.savez(source / "test_cache.npz", rates=rates, centers=centers, counts_0=counts, edges_0=np.arange(5) * 0.02)
    item = {"tag": "test", "dataset": "synthetic", "animal": "rat", "session": "run"}
    generate_parent(item, source, parent, n_paths=4, repeats=1, events=3, supports=(2, 4))
    p = parent / "test_r000_scores.csv.gz"
    metadata = pd.read_csv(p)
    metadata.loc[metadata.support.eq(4), "support"] = 256
    metadata.to_csv(p, index=False)
    result = record_task(item, source, parent, out, supports=(2, 4), banks=(0, 1), chunk=2, repeats=1)
    assert result["observations"] == 18 and result["score_rows"] == 72
    ll = np.load(out / "test_scores.npy")
    assert ll.shape == (2, 2, 18, 5) and np.isfinite(ll).all()
    np.testing.assert_allclose(ll[0, 0, :, 2], ll[1, 1, :, 2])
    table = pd.read_csv(out / "test_observations.csv.gz")
    assert set(table.source_teacher) == {"original_bank_0", "original_bank_1"}
    audit = record_audit(item, source, parent, out, supports=(2, 4), banks=(0, 1), repeats=1, chunk=2)
    assert audit["observations_audited"] == 12
    assert audit["paths_regenerated"] == 8
    assert audit["likelihoods_recomputed"] == 240
    assert audit["max_score_error"] < 1e-7
    corrupted = np.load(out / "test_scores.npy", mmap_mode="r+")
    corrupted[0, 0, 0, 0] += 1
    corrupted.flush()
    with pytest.raises(AssertionError):
        record_audit(item, source, parent, out, supports=(2, 4), banks=(0, 1), repeats=1, chunk=2)
