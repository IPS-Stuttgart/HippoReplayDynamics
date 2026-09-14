import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_spatially_balanced_content_remedy as remedy
from scripts import audit_spatially_balanced_content_remedy as audit
from scripts import report_spatially_balanced_content_remedy as reporter


def fixture(n=21):
    rng = np.random.default_rng(9)
    grid = np.array([(x, y) for x in np.arange(5)*8 for y in np.arange(5)*8], dtype=float)
    peaks = grid[np.arange(n)//2 % len(grid)]
    rates = .05+12*np.exp(-np.square(peaks[:, None]-grid).sum(axis=2)/100)
    state = rng.integers(len(grid), size=120)
    counts = rng.poisson(.02*rates[:, state].T)
    return rates, grid, counts, grid[state]


def test_equal_disjoint_same_universe_deterministic():
    rates, grid, counts, truth = fixture()
    part, shortlist = remedy.choose_partitions(len(rates), rates, grid, counts, truth, 11)
    assert part == remedy.choose_partitions(len(rates), rates, grid, counts, truth, 11)[0]
    assert len(shortlist) == 8
    for condition in ("random", "balanced"):
        a, b = part[condition]["a"], part[condition]["b"]
        assert len(a) == len(b) == 10
        assert not set(a) & set(b)
        assert set(a+b) == set(part["universe"])
    winner = min(shortlist, key=lambda p: (p["run_loss"], p["profile_loss"], p["candidate_id"]))
    assert part["balanced"] == winner


def test_minimum_cells_and_shape_fail_closed():
    rates, grid, counts, truth = fixture(9)
    with pytest.raises(ValueError, match="five cells"):
        remedy.choose_partitions(9, rates, grid, counts, truth, 11)
    with pytest.raises(ValueError, match="align"):
        remedy.choose_partitions(9, rates, grid, counts[:, :-1], truth, 11)


@pytest.mark.parametrize("source", ["sim_matched", "sim_drift"])
def test_shared_simulation_exact_counts_zero_counts_and_odd_drop(source):
    rates, grid, counts, truth = fixture()
    counts[0] = 0
    part, _ = remedy.choose_partitions(len(rates), rates, grid, counts, truth, 11)
    universe = part["universe"]
    simulated, true_xy = remedy.simulated_case(counts, rates, rates*1.3, universe, grid, 21, source)
    repeat, repeat_xy = remedy.simulated_case(counts, rates, rates*1.3, universe, grid, 21, source)
    assert np.array_equal(simulated, repeat)
    assert np.array_equal(true_xy, repeat_xy)
    assert np.array_equal(simulated.sum(axis=1), counts[:, universe].sum(axis=1))
    assert not simulated[0].any()
    assert not simulated[:, list(set(range(len(rates)))-set(universe))].any()
    totals = []
    for condition in ("random", "balanced"):
        p = part[condition]
        totals.append(simulated[:, p["a"]].sum(axis=1)+simulated[:, p["b"]].sum(axis=1))
    assert np.array_equal(totals[0], totals[1])


def results_fixture(tmp_path, *, diffuse=False, inaccurate=False, animals=5):
    results = []
    for animal in range(animals):
        directory = tmp_path/f"rat{animal}"
        directory.mkdir()
        rows = []
        for source in ("real", "run_test", "sim_matched", "sim_drift"):
            for condition in ("random", "balanced"):
                balanced = condition == "balanced"
                for event in range(3+animal):
                    error = 21 if inaccurate and balanced else 18 if balanced else 20
                    entropy = .9 if diffuse and balanced else .75 if balanced else .8
                    rows.append(dict(dataset="blackstad_moser", animal=f"R{animal}", session="S",
                        source=source, condition=condition, split=0, draw=-1, event_index=event,
                        endpoint_separation_cm=30 if balanced else 40, regional_tv=.3 if balanced else .4,
                        pair_entropy=entropy, a_entropy=entropy, b_entropy=entropy,
                        pair_truth_error_cm=error, a_truth_error_cm=error, b_truth_error_cm=error,
                        map_region_agreement=.5, a_width_cm=20, b_width_cm=20,
                        a_spikes=4, b_spikes=4, a_active=3, b_active=3))
        pd.DataFrame(rows).to_csv(directory/"event_readouts.csv.gz", index=False)
        results.append(dict(dataset="blackstad_moser", animal=f"R{animal}", session="S",
            status="complete", candidates=3+animal, source_candidates=3+animal, artifact_dir=str(directory)))
    return results


def test_joint_outcome_gates_and_equal_animal_weights(tmp_path):
    results = results_fixture(tmp_path)
    gates = remedy.summarize(results, tmp_path)
    assert gates.passed.all()
    summary = pd.read_csv(tmp_path/"summary.csv")
    assert summary.loc[summary.condition.eq("balanced"), "regional_tv"].eq(.3).all()


@pytest.mark.parametrize("failure", ["diffuse", "inaccurate"])
def test_agreement_improvement_is_not_sufficient(tmp_path, failure):
    results = results_fixture(tmp_path, **{failure: True})
    gates = remedy.summarize(results, tmp_path).set_index("gate")
    assert gates.loc["regional_tv_reduction", "passed"]
    assert not gates.loc["statistical_validation", "passed"]


def test_failed_sessions_stay_in_denominator_and_hash_failure_blocks(tmp_path):
    results = results_fixture(tmp_path, animals=4)
    results.append(dict(dataset="blackstad_moser", animal="failed", session="S",
                        status="failed", source_candidates=1000))
    gates = remedy.summarize(results, tmp_path, inputs_unchanged=False).set_index("gate")
    assert not gates.loc["external_coverage", "passed"]
    assert not gates.loc["inputs_unchanged", "passed"]
    assert not gates.loc["statistical_validation", "passed"]


def test_population_freeze_precedes_test_and_replay_access(tmp_path, monkeypatch):
    rates, grid, counts, truth = fixture()
    cache = tmp_path/"cache.npz"
    np.savez(cache, unit_qc_mask=np.ones(len(rates), bool), valid_spatial_bins=np.ones(len(grid), bool),
             occupancy_first_half_s=np.ones(len(grid)), bin_centers_cm=grid,
             rates_first_half_hz=rates, rates_second_half_hz=rates,
             cell_ids=np.arange(len(rates)), supported_run_intervals=np.array([[0., 400.]]))
    out = tmp_path/"out"

    def windows(data, ids, start, end):
        if start == 300:
            frozen = json.loads((out/"selected_before_test.json").read_text())
            assert len(frozen["parts"]) == 3
            raise ValueError("heldout deliberately unavailable")
        return np.column_stack([np.arange(len(truth)), np.arange(len(truth))+.02, truth]), counts

    monkeypatch.setattr(remedy, "run_windows", windows)
    row = SimpleNamespace(artifact_path=str(cache), artifact_sha256=remedy.file_sha256(cache),
                          dataset="synthetic", animal="R", session="S", candidates=10)
    with pytest.raises(ValueError, match="deliberately unavailable"):
        remedy.measure(row, out, 11)
    assert not (out/"event_readouts.csv.gz").exists()


def test_independent_selection_and_decode_reconstruction():
    rates, grid, counts, truth = fixture()
    chosen, _ = remedy.choose_partitions(len(rates), rates, grid, counts, truth, 11)
    random, balanced, universe = audit.reconstruct_selection(rates, grid, counts, truth, 11)
    assert chosen["random"] == random
    assert chosen["universe"] == universe
    assert chosen["balanced"]["a"] == balanced["a"]
    assert chosen["balanced"]["b"] == balanced["b"]
    np.testing.assert_allclose(chosen["balanced"]["run_loss"], balanced["run_loss"])
    direct = audit.posterior(counts, rates, grid)
    original = remedy.decode(counts, rates, grid)
    for key in ("mean", "width", "entropy", "map_region"):
        np.testing.assert_allclose(direct[key], original[key], atol=1e-7)
    for source in ("sim_matched", "sim_drift"):
        x, y = remedy.simulated_case(counts, rates, rates, universe, grid, 13, source)
        xx, yy = audit.reconstruct_generator(counts, rates, rates, universe, grid, 13, source)
        assert np.array_equal(x, xx) and np.array_equal(y, yy)


def test_report_requires_current_audit_and_retains_failed_verdict(tmp_path):
    results = results_fixture(tmp_path, inaccurate=True)
    for record in results:
        record["readouts_sha256"] = reporter.digest(f"{record['artifact_dir']}/event_readouts.csv.gz")
    remedy.summarize(results, tmp_path)
    (tmp_path/"manifest.json").write_text(json.dumps(dict(results=results)))
    details = dict(passed=True, producer_manifest_sha256=reporter.digest(tmp_path/"manifest.json"),
                   reconstructed_readouts=100, sessions=5,
                   verified_table_sha256={name: reporter.digest(tmp_path/f"{name}.csv") for name in ("summary", "by_animal", "by_session", "gates")})
    (tmp_path/"independent_audit.json").write_text(json.dumps(details))
    information, _ = reporter.report(tmp_path)
    assert information.both_silent_fraction.eq(0).all()
    assert "**FAIL**" in (tmp_path/"audited_outcome.md").read_text()
    assert (tmp_path/"external_sampling_validation.png").stat().st_size > 10000
    (tmp_path/"manifest.json").write_text(json.dumps(dict(results=results, changed=True)))
    with pytest.raises(ValueError, match="stale"):
        reporter.report(tmp_path)


def test_independent_summary_and_gate_checks_reject_table_changes(tmp_path):
    results = results_fixture(tmp_path, inaccurate=True)
    remedy.summarize(results, tmp_path)
    data = pd.concat([pd.read_csv(f"{r['artifact_dir']}/event_readouts.csv.gz") for r in results])
    audit.check_aggregate_tables(data, tmp_path)
    audit.check_gates(tmp_path, dict(results=results, inputs_unchanged=True))
    gates = pd.read_csv(tmp_path/"gates.csv")
    gates.loc[gates.gate.eq("statistical_validation"), "passed"] = True
    gates.to_csv(tmp_path/"gates.csv", index=False)
    with pytest.raises(AssertionError):
        audit.check_gates(tmp_path, dict(results=results, inputs_unchanged=True))
    summary = pd.read_csv(tmp_path/"by_animal.csv")
    summary.loc[0, "endpoint_separation_cm"] += 1
    summary.to_csv(tmp_path/"by_animal.csv", index=False)
    with pytest.raises(AssertionError):
        audit.check_aggregate_tables(data, tmp_path)
