from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.dense_path_clocks import PathAccumulator
from hipporeplayimm.exact_path_clocks import merge_blocks, score_geometry_block
from hipporeplayimm.literal_replay_clock import SpatialRates
from scripts.verify_dense_path_clock_recovery import integral_bins
from scripts.verify_exact_path_clock_recovery import independent_block, independent_geometries


def fixture():
    centers = np.array(list(product(np.arange(0, 49, 8), repeat=2)), dtype=float)
    x, y = centers.T
    rates = np.array([1 + x / 3, 1 + y / 4, 2 + (48 - x) / 5, 2 + (48 - y) / 6])
    generator = np.random.default_rng(72)
    counts = generator.poisson(0.6, (7, 5, 4))
    return centers, rates, {5: (np.arange(7), counts)}


def test_exhaustive_blocks_match_independent_geometry_scores_and_brute_force():
    centers, rates, groups = fixture()
    spatial = SpatialRates(centers, rates)
    a = score_geometry_block(spatial, groups, 0, 2, chunk=7)
    b = score_geometry_block(spatial, groups, 2, 4, chunk=13)
    full = score_geometry_block(spatial, groups, 0, 4, chunk=31)
    reference = independent_block(centers, rates, groups, 0, 4, chunk=11)
    for key, values in reference.items():
        np.testing.assert_allclose(full[key], values, atol=1e-9)
    actual, totals = merge_blocks([a, b], groups, rates, 7)
    combined, _ = merge_blocks([full], groups, rates, 7)
    np.testing.assert_allclose(actual, combined)
    reverse, _ = merge_blocks([b, a], groups, rates, 7)
    np.testing.assert_allclose(actual, reverse)
    weights, ll = [], [[], []]
    for descriptor, path in independent_geometries(centers, rates, 0, 4):
        if path is None:
            continue
        values, clocks = path
        weights.append(0.5 / totals[int(descriptor[2] != 0)])
        for clock in range(2):
            p = integral_bins(clocks[clock], values, 5)
            p /= p.sum(axis=1, keepdims=True)
            ll[clock].append(np.einsum("etc,tc->et", groups[5][1], np.log(p)))
    for clock in range(2):
        per_bin = np.array(ll[clock])
        np.testing.assert_allclose(actual[:, clock], logsumexp(per_bin.sum(axis=2) + np.log(weights)[:, None], axis=0))
        np.testing.assert_allclose(actual[:, clock + 3], logsumexp(per_bin + np.log(weights)[:, None, None], axis=0).sum(axis=1))


def test_family_prior_is_half_each_not_uniform_over_union():
    counts = np.array([[[3, 0], [3, 0]]])
    probs = [np.array([[[0.9, 0.1], [0.9, 0.1]]]), np.repeat(np.array([[[0.1, 0.9], [0.1, 0.9]]]), 3, axis=0)]
    block = {"accepted": np.array([1, 3])}
    for family, p in enumerate(probs):
        acc = PathAccumulator(counts)
        acc.add(np.log(p))
        for clock in range(2):
            block[f"coherent_{family}_{clock}_2"] = acc.coherent
            block[f"reset_{family}_{clock}_2"] = acc.reset
    score, _ = merge_blocks([block], {2: (np.arange(1), counts)}, np.ones((2, 3)), 1)
    np.testing.assert_allclose(score[0, 0], np.log(0.5 * 0.9**6 + 0.5 * 0.1**6))
    np.testing.assert_allclose(score[0, 3], 2 * np.log(0.5 * 0.9**3 + 0.5 * 0.1**3))
    assert not np.isclose(score[0, 0], np.log(0.25 * 0.9**6 + 0.75 * 0.1**6))


def test_empty_geometry_is_not_a_vacuous_success():
    centers, rates, groups = fixture()
    block = score_geometry_block(SpatialRates(centers, rates), groups, 0, 0)
    assert block["descriptors"].shape == (0, 3)
    with pytest.raises(ValueError, match="both complete"):
        merge_blocks([block], groups, rates, 7)


def test_tampered_partial_fails_independent_comparison():
    centers, rates, groups = fixture()
    actual = score_geometry_block(SpatialRates(centers, rates), groups, 0, 1)
    reference = independent_block(centers, rates, groups, 0, 1)
    actual["coherent_0_0_5"][0] += 1
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(actual["coherent_0_0_5"], reference["coherent_0_0_5"])


def test_saved_block_round_trip_and_audit(tmp_path):
    from scripts.run_exact_path_clock_recovery import block_task
    from scripts.verify_exact_path_clock_recovery import audit_block

    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    centers, rates, groups = fixture()
    np.savez(source / "toy_cache.npz", centers=centers, rates=rates)
    np.savez(run / "toy_counts.npz", index_5=groups[5][0], counts_5=groups[5][1])
    pd.DataFrame({"row_index": range(7), "event_in_population": range(7)}).to_csv(run / "toy_observations.csv.gz", index=False)
    block = block_task("toy", 0, 2, source, run)
    result = audit_block(block, source, run)
    assert result["paths_regenerated"] == sum(block["accepted"])
    assert result["max_error"] < 1e-8
    path = run / block["file"]
    with np.load(path) as z:
        altered = {k: z[k] for k in z.files}
    altered["reset_0_0_5"][0, 0] += 1
    np.savez(path, **altered)
    with pytest.raises(AssertionError):
        audit_block(block, source, run)


def test_same_population_fits_for_identical_exact_and_mc_inputs(tmp_path):
    from scripts.run_exact_path_clock_recovery import fit_repeat

    run, dense = tmp_path / "run", tmp_path / "dense"
    run.mkdir()
    dense.mkdir()
    metadata = pd.DataFrame(
        {"dataset": "pfeiffer_foster", "animal": "rat", "session": "toy", "source_teacher": "original_bank_1", "scenario": 0.25, "repeat": 0, "row_index": range(128)}
    )
    metadata.to_csv(run / "toy_observations.csv.gz", index=False)
    scores = np.random.default_rng(917).normal(size=(128, 5))
    np.save(run / "toy_exact_scores.npy", scores)
    np.save(dense / "toy_scores.npy", np.broadcast_to(scores, (2, 3, 128, 5)))
    rows = pd.DataFrame(fit_repeat(0, run, dense, tags=("toy",)))
    assert len(rows) == 7 and rows.n_events.eq(128).all()
    assert rows.phi_hat.nunique() == 1 and rows.phi_low.nunique() == 1 and rows.phi_high.nunique() == 1
