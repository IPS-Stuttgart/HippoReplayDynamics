import itertools
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from hipporeplayimm.replay_coverage_shuffle_baseline import (
    CRITERIA,
    FAMILIES,
    bank_rates,
    screen_batch,
    screen_maps,
    screen_plan,
    shuffle_bank,
    shuffle_test,
    sparse_map,
)
from scripts._provenance import file_sha256
from scripts.analyze_replay_coverage_shuffle_baseline import (
    decision_rows,
    input_arrays,
    order_permutation,
    process_session,
    run,
    summarize,
)
from scripts.audit_replay_coverage_shuffle_baseline import audit_dense_map, audit_session
from scripts.audit_replay_coverage_shuffle_baseline import main as run_audit
from scripts.report_replay_coverage_shuffle_baseline import order_contrast, report


def scalar_screen(path, grid, counts, filtered, min_frames):
    supported = np.flatnonzero(counts.sum(axis=1) >= 2)
    if not len(supported):
        return False
    longest = []
    current = []
    for t in range(supported[0], supported[-1] + 1):
        if filtered and (counts[t].sum() < 3 or np.count_nonzero(counts[t]) < 2):
            current = []
            continue
        if current and np.linalg.norm(grid[path[t]] - grid[path[current[-1]]]) >= 20:
            current = []
        current.append(t)
        if len(current) > len(longest):
            longest = list(current)
    return bool(len(longest) >= min_frames and np.linalg.norm(grid[path[longest[-1]]] - grid[path[longest[0]]]) >= 40)


def test_batched_geometry_matches_independent_scalar_all_events():
    rng = np.random.default_rng(4)
    grid = np.column_stack([np.arange(25) * 8, np.zeros(25)])
    offsets = np.array([0, 0, 1, 15, 15, 50, 60, 60])
    counts = rng.poisson(.8, (60, 3))
    counts[4:7] = 0
    paths = rng.integers(0, 25, (17, 60))
    paths[0] = np.arange(60) % 25
    got = screen_batch(paths, grid, screen_plan(counts, offsets))
    for k in range(len(paths)):
        for j, (a, b) in enumerate(itertools.pairwise(offsets)):
            for c, (filtered, minimum) in enumerate(CRITERIA):
                assert got[k, j, c] == scalar_screen(paths[k, a:b], grid, counts[a:b], filtered, minimum)


def test_edges_internal_support_strict_jumps_and_frame_sensitivity():
    grid = np.column_stack([np.arange(30) * 8, np.zeros(30)])
    counts = np.ones((10, 3), int)
    assert screen_maps(np.arange(10)[None], grid, counts).tolist() == [[True, True, False, False]]
    counts[5] = 0
    assert screen_maps(np.arange(10)[None], grid, counts).tolist() == [[True, False, False, False]]
    counts[0] = 0
    assert not screen_maps(np.arange(10)[None], grid, counts).any()
    grid[:, 0] = np.arange(30) * 20
    assert not screen_maps(np.arange(10)[None], grid, np.ones((10, 3))).any()


def test_earliest_longest_tie_is_not_largest_displacement():
    grid = np.column_stack([np.arange(40) * 8, np.zeros(40)])
    path = np.r_[np.zeros(10, int), np.arange(10, 20)]
    assert not screen_maps(path[None], grid, np.ones((20, 3))).any()


def test_forty_cm_diagonal_at_decimal_grid_offset_is_inclusive():
    from scripts.audit_replay_coverage_shuffle_baseline import independent_screen

    grid = np.column_stack([np.linspace(0, 32, 11), np.linspace(0, 23.999999999999993, 11)])
    counts, path = np.ones((11, 3)), np.arange(11)
    assert screen_maps(path[None], grid, counts).all()
    for filtered, minimum in CRITERIA:
        assert independent_screen(path, grid, counts, filtered, minimum)
    grid[-1] -= .001
    assert not screen_maps(path[None], grid, counts).any(), "tolerance must not admit scientifically shorter paths"


def test_sparse_poisson_map_equals_dense_and_batches():
    rng = np.random.default_rng(8)
    counts = rng.poisson(.1, (83, 13))
    rates = np.exp(rng.normal(size=(7, 13, 21)))
    dense = np.array([(counts @ np.log(rate * .02) - .02 * rate.sum(axis=0)).argmax(axis=1) for rate in rates])
    np.testing.assert_array_equal(sparse_map(counts, rates, 17), dense)
    np.testing.assert_array_equal(sparse_map(counts, rates, 81), dense)
    rates[0, 0, 0] = 0
    with pytest.raises(ValueError, match="positive rates"):
        sparse_map(counts, rates)


@pytest.mark.parametrize("family", FAMILIES)
def test_shuffle_operator_and_prefix_match_explicit_maps(family):
    rng = np.random.default_rng(17)
    rates = rng.uniform(.01, 20, (5, 35))
    support = rng.random(35) > .3
    bank = shuffle_bank(5, [5, 7], 17, 81, family)
    np.testing.assert_array_equal(bank[:3], shuffle_bank(5, [5, 7], 3, 81, family))
    actual = bank_rates(rates, support, [5, 7], bank, family)
    for k in range(len(bank)):
        if family == "cell_identity":
            expected = rates[bank[k]][:, support]
            assert sorted(bank[k]) == list(range(5))
        else:
            expected = np.array([np.roll(rate.reshape(5, 7), tuple(shift), axis=(0, 1)).ravel()[support] for rate, shift in zip(rates, bank[k], strict=True)])
            assert (bank[k].sum(axis=1) > 0).all()
        np.testing.assert_array_equal(actual[k], expected)
    with pytest.raises(ValueError, match="complete rectangular"):
        bank_rates(rates[:, support], np.ones(support.sum(), bool), [5, 7], bank, family)


def test_geometry_failures_never_receive_vacuous_significance():
    grid = np.column_stack([np.arange(16) * 8, np.zeros(16)])
    rates = np.eye(16) * 20 + .01
    counts = np.r_[np.eye(16, dtype=int) * 4, np.zeros((12, 16), int)]
    one = shuffle_test(counts, [0, 16, 28, 28], grid, rates, np.ones(16, bool), [16, 1], 99, 881, 1)
    many = shuffle_test(counts, [0, 16, 28, 28], grid, rates, np.ones(16, bool), [16, 1], 99, 881, 8)
    for key in ["original_path", "geometric_pass", "null_pass", "p_values", "accepted"]:
        np.testing.assert_array_equal(one[key], many[key])
    assert one["geometric_pass"][0, 0]
    assert not one["geometric_pass"][1:].any()
    assert not one["accepted"][1:].any()
    assert np.isnan(one["p_values"][~one["geometric_pass"]]).all()
    assert one["tested_observations"].tolist() == [0]
    p = (one["null_pass"][:, :, 0, 0].sum(axis=1) + 1) / 100
    np.testing.assert_allclose(one["p_values"][0, 0], p)
    assert one["accepted"][0, 0] == (p < .02).all()


def test_zero_observations_are_supported_without_positive_rows():
    result = shuffle_test(np.empty((0, 3)), [0], np.array([[0, 0], [10, 0]]), np.ones((3, 2)), np.ones(2, bool), [2, 1], 5, 8)
    assert result["accepted"].shape == (0, 4)
    assert not result["accepted"].any()


def test_order_shuffle_preserves_partial_bin_and_all_population_vectors():
    durations = np.r_[np.full(40, .005), .002]
    perm = order_permutation(durations, 92)
    assert perm[-1] == 40 and sorted(perm) == list(range(41))
    assert not np.array_equal(perm[:-1], np.arange(40))
    np.testing.assert_array_equal(perm, order_permutation(durations, 92))


def fixture(tmp_path):
    ids = np.arange(8)
    grid = np.column_stack([np.arange(16) * 8 + 4, np.full(16, 4.)])
    rates = .01 + 20 * np.exp(-((grid[:, 0][None] - np.arange(8)[:, None] * 16) / 10) ** 2)
    spikes = np.array([[t, min(7, int(t / .024))] for t in np.arange(.0001, .19, .001)])
    cache = tmp_path / "cache.npz"
    np.savez(cache, cell_ids=ids, spikes=spikes, rates_hz=rates, unit_qc_mask=np.ones(8, bool),
        bin_centers_cm=grid, x_edges_cm=np.arange(17) * 8., y_edges_cm=[0., 8.], valid_spatial_bins=np.ones(16, bool), arena_bounds_cm=[[0., 0.], [128., 8.]])
    windows = pd.DataFrame([{"dataset": "tanni2022", "animal": "R", "session": "S", "window_uid": "event0", "detector": "source_high_mua",
        "eligible": True, "window_variant": "detected_core", "start_s": 0., "end_s": .193}])
    wp = tmp_path / "windows.csv"
    windows.to_csv(wp, index=False)
    return {"dataset": "tanni2022", "animal": "R", "session": "S", "source_cache_path": str(cache), "source_cache_sha256": file_sha256(cache),
        "windows_path": str(wp), "windows_sha256": file_sha256(wp), "ripple_status": "unavailable_insufficient_baseline"}, windows, spikes


def test_input_counts_and_full_session_denominators(tmp_path):
    record, windows, spikes = fixture(tmp_path)
    arrays = input_arrays(record, windows, 20260905)
    for j, (t, duration) in enumerate(zip(arrays["base_starts_s"], arrays["base_durations_s"], strict=True)):
        subset = spikes[(spikes[:, 0] >= t) & (spikes[:, 0] < t + duration), 1]
        np.testing.assert_array_equal(arrays["base_counts"][j], [np.sum(subset == cell) for cell in arrays["cell_ids"]])
    n = arrays["frame_offsets"][1]
    np.testing.assert_array_equal(arrays["frame_counts"][n:], np.array([arrays["base_counts"][arrays["permutation"]][i:i + 4].sum(axis=0) for i in range(n)]))
    out = tmp_path / "out"
    out.mkdir()
    result = process_session(record, out, 20260905, 5, 3)
    assert result["decision_rows"] == 32
    pop = pd.read_csv(result["population_path"])
    ripple = pop[pop.detector == "lfp_ripple_detected"]
    assert ripple.eligible_events.eq(0).all() and ripple.accepted_fraction.isna().all()
    assert not ripple.detector_available.any()
    session, animal, summary = summarize(pop, 20)
    assert len(session) == 48 and len(animal) == 48 and len(summary) == 288
    missing = summary[summary.detector == "lfp_ripple_detected"]
    assert missing.animals_measurable.eq(0).all()
    checked = audit_session(result, 5)
    assert checked["decision_rows"] == 32 and checked["dense_analytic_map_frames"] > 0
    assert pd.read_csv(result["metrics_path"]).geometric_pass.any()
    with np.load(result["shuffle_files"][0]["path"]) as z:
        corrupt = dict(z)
    corrupt["p_values"][:] = 0
    np.savez_compressed(result["shuffle_files"][0]["path"], **corrupt)
    result["shuffle_files"][0]["sha256"] = file_sha256(result["shuffle_files"][0]["path"])
    with pytest.raises(AssertionError):
        audit_session(result, 5)


def test_empty_session_is_explicit_not_positive(tmp_path):
    record, windows, _ = fixture(tmp_path)
    windows.eligible = False
    windows.to_csv(record["windows_path"], index=False)
    record["windows_sha256"] = file_sha256(record["windows_path"])
    out = tmp_path / "empty"
    out.mkdir()
    result = process_session(record, out, 20260905, 3, 2)
    assert result["decision_rows"] == 0
    pop = pd.read_csv(result["population_path"])
    assert pop.eligible_events.eq(0).all() and pop.accepted_fraction.isna().all()
    assert audit_session(result, 3)["decision_rows"] == 0


def test_dense_auditor_only_permits_tied_maxima():
    assert audit_dense_map(np.ones((1, 2)), np.ones((2, 3)), np.array([2])) == 1
    with pytest.raises(AssertionError, match="likelihood maximum"):
        audit_dense_map(np.array([[5, 0]]), np.array([[1., 20], [1., 1]]), np.array([0]))


def test_equal_animal_not_event_session_or_subset_weighting():
    rows = []
    for animal, n_sessions, full, half in [("A", 3, .6, .2), ("B", 1, .8, .6)]:
        for s in range(n_sessions):
            for fraction, replicates, value in [(1., 1, full), (.5, 3, half)]:
                for rep in range(replicates):
                    rows.append({"dataset": "D", "animal": animal, "session": f"s{s}", "detector": "MUA", "observation": "original_order",
                        "bin_filter": "edge_only", "min_frames": 10, "alpha": .02, "cell_fraction": fraction, "population_replicate": rep,
                        "eligible_events": 100 if animal == "A" else 10, "detector_available": True,
                        "geometric_fraction": value, "accepted_fraction": value})
    _, _, summary = summarize(pd.DataFrame(rows), 50)
    row = summary[summary.metric.eq("accepted_fraction_delta")].iloc[0]
    assert row.equal_animal_mean == pytest.approx(-.3) and row.animals_measurable == 2


def test_end_to_end_provenance_smoke_and_report(tmp_path):
    record, _, _ = fixture(tmp_path)
    record["status"] = "complete"
    source = tmp_path / "coverage_event_definition_manifest.json"
    source.write_text(json.dumps({"status": "complete", "results": [record]}))
    (tmp_path / "coverage_event_definition_reconstruction_audit.json").write_text(json.dumps(
        {"status": "pass", "input_file_sha256": {"preparation_manifest": file_sha256(source)}}))
    output = tmp_path / "score"
    run(Namespace(input_dir=tmp_path, output_dir=output, sessions=None, workers=1, shuffles=5, seed=20260905, batch_size=3, bootstraps=20))
    gates = pd.read_csv(output / "coverage_shuffle_baseline_gate_summary.csv").set_index("gate").passed
    assert gates.overall_technical and not gates.published_shuffle_budget and not gates.full_benchmark_complete
    run_audit(Namespace(input_dir=output, workers=1, bootstraps=20))
    report(output, tmp_path / "report")
    text = (tmp_path / "report/coverage_shuffle_baseline_report.md").read_text()
    assert "technical_smoke" in text and "not an empirical false-positive rate" in text
    assert (tmp_path / "report/coverage_shuffle_baseline_primary.png").stat().st_size > 1000
    assert (tmp_path / "report/coverage_shuffle_baseline_order_contrast.csv").exists()


def test_order_contrast_is_paired_and_does_not_confuse_acceptance_with_order_excess():
    rows = []
    for animal, original, shuffled in [("a", [.20, .10], [.18, .01]), ("b", [.40, .20], [.36, .02])]:
        for order, rates in [("original_order", original), ("order_randomized", shuffled)]:
            rows.append({"dataset": "d", "detector": "ripple", "animal": animal, "observation": order,
                "bin_filter": "edge_only", "min_frames": 10, "alpha": .02,
                "accepted_fraction_full": rates[0], "accepted_fraction_half": rates[1]})
    animals = pd.DataFrame(rows)
    result = order_contrast(animals, bootstraps=50).set_index("metric")
    assert result.loc["order_excess_full", "equal_animal_mean_pp"] == pytest.approx(3)
    assert result.loc["order_excess_half", "equal_animal_mean_pp"] == pytest.approx(13.5)
    assert result.loc["half_minus_full_order_excess", "equal_animal_mean_pp"] == pytest.approx(10.5)
    assert result.loc["half_minus_full_order_excess", "animals_positive"] == 2
    with pytest.raises(ValueError, match="both observations"):
        order_contrast(animals.iloc[:-1], bootstraps=50)
    with pytest.raises(pd.errors.MergeError):
        order_contrast(pd.concat([animals, animals.iloc[:1]]), bootstraps=50)


def test_decision_status_keeps_zero_p_uncomputed_not_significant():
    windows = pd.DataFrame([{"dataset": "d", "animal": "a", "session": "s", "window_uid": "e", "detector": "source_high_mua"}])
    result = {"tested_observations": np.array([], int), "null_pass": np.empty((2, 5, 0, 4), bool),
        "geometric_pass": np.zeros((2, 4), bool), "p_values": np.full((2, 4, 2), np.nan)}
    rows = pd.DataFrame(decision_rows(result, windows, {"cell_fraction": 1., "population_replicate": 0, "cell_ids": [1, 2]}, 5))
    assert len(rows) == 8 and rows.n_shuffles.eq(0).all()
    assert rows.test_status.eq("geometric_fail_not_tested").all()
    assert rows.cell_identity_p.isna().all()
    assert not rows["accepted_alpha_0.02"].any()
