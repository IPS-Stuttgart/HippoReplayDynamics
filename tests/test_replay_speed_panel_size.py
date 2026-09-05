"""Nested paths, frozen readout parity, availability and budget-isolated fits."""

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from hipporeplayimm.replay_speed_panel_size import (
    READOUT,
    READOUTS,
    decode_event_moments,
    prefix_panels,
    profile_cycle,
    validate_event_statistics,
)
from scripts._provenance import file_sha256
from scripts.audit_replay_speed_panel_size import audit, verify_moment_panels
from scripts.calibrate_replay_speed_identifiability import decode_panel, draw_schedule, evaluate_panels, validate_panel_contract
from scripts.calibrate_replay_speed_panel_size import ROOT, draw_observations, run, score_session, simulate_nested
from scripts.report_replay_speed_panel_size import aggregate_events, directional_endpoints, paired_budgets, report


def tiny_model():
    x, y = np.arange(0., 161., 20.), np.arange(0., 121., 20.)
    grid = np.stack(np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, indexing="ij"), axis=-1).reshape(-1, 2)
    centers = grid[::3]
    rates = .1 + 20 * np.exp(-np.sum((centers[:, None] - grid[None]) ** 2, axis=2) / (2 * 22. ** 2))
    return {"rates_hz": rates, "training_full_rates_hz": rates, "generator_rates_hz": np.roll(rates, 1, axis=1),
        "grid_cm": grid, "x_edges_cm": x, "y_edges_cm": y, "generator_x_edges_cm": x,
        "generator_y_edges_cm": y, "domain_cm": np.array([[0., 0.], [160., 120.]]), "cell_ids": np.arange(len(rates))}


def profiles():
    return pd.DataFrame({"source_event_index": [10, 20, 30], "n_base_bins": [20, 24, 30]})


def draw(phase="test"):
    return draw_schedule(22, "synthetic:r:s", 1, 1, 1, 1).query("phase == @phase").iloc[0].to_dict()


def test_profile_cycles_balanced_prefix_stable_and_order_invariant():
    plan = profile_cycle(profiles(), 11, 13)
    pd.testing.assert_frame_equal(plan.iloc[:5].reset_index(drop=True), profile_cycle(profiles().iloc[::-1], 5, 13))
    for _, group in plan.groupby("profile_cycle"):
        assert group.source_event_index.nunique() == len(group)
    assert plan.source_event_index.value_counts().max() - plan.source_event_index.value_counts().min() <= 1
    assert not plan.equals(profile_cycle(profiles(), 11, 14))
    with pytest.raises(ValueError, match="unique"):
        profile_cycle(pd.concat([profiles(), profiles().iloc[:1]]), 5, 1)
    bad = profiles()
    bad.loc[0, "n_base_bins"] = 3
    with pytest.raises(ValueError, match="four"):
        profile_cycle(bad, 5, 1)


def test_reused_durations_never_reuse_observations_and_budgets_are_paired():
    model, d = tiny_model(), draw()
    p, chunks, hashes = draw_observations(model, profiles(), d, 10)
    short_plan, short, short_hashes = draw_observations(model, profiles(), d, 5)
    pd.testing.assert_frame_equal(p.iloc[:5], short_plan)
    assert hashes[:5] == short_hashes and len(set(hashes)) == 10
    for key, counts in chunks.items():
        for a, b in zip(counts[:5], short[key], strict=True):
            np.testing.assert_array_equal(a, b)
        assert all(count.shape[0] == 5 * n for count, n in zip(counts, p.n_base_bins, strict=True))
    assert set(chunks) == {("A", "poisson"), ("A", "shared_gain"), ("B", "poisson"), ("B", "shared_gain")}
    _, training, _ = draw_observations(model, profiles(), draw("fit"), 5)
    assert set(training) == {("A", "poisson")}


def test_per_event_moments_exactly_reproduce_frozen_readouts_at_every_prefix():
    model = tiny_model()
    _, chunks, _ = draw_observations(model, profiles(), draw(), 10)
    fine = chunks["A", "poisson"]
    moments, diag = decode_event_moments(fine, model)
    panels = prefix_panels(moments, diag, [5, 10], 45, False)
    for budget in [5, 10]:
        expected = decode_panel(fine[:budget], model, 45, False).set_index(READOUT).sort_index()
        observed = panels[panels.candidate_budget.eq(budget)].set_index(READOUT).sort_index()
        pd.testing.assert_frame_equal(observed[expected.columns], expected, check_dtype=False, rtol=1e-11, atol=1e-11)
    one, _ = decode_event_moments(fine[:1], model)
    np.testing.assert_allclose(one[0], moments[0], rtol=1e-11, atol=1e-11, equal_nan=True)


def test_dropped_bins_are_not_bridged(monkeypatch):
    import hipporeplayimm.replay_speed_panel_size as module

    fine = np.ones((200, 2), int)
    fine[65:110] = 0
    n = (200 - 20) // 5 + 1
    points = np.column_stack([np.arange(n) * 5., np.zeros(n)])
    points[15:19, 0] += 100
    monkeypatch.setattr(module, "decode_independent", lambda *a: {"map": points, "posterior_mean": points, "posterior": None})
    model = {"rates_hz": np.ones((2, 2)), "grid_cm": np.zeros((2, 2)), "domain_cm": np.array([[0., 0.], [300., 200.]])}
    moments, diag = decode_event_moments([fine], model)
    idx = READOUTS.index(("posterior_mean", "at_least_2cells_3spikes", "all"))
    counts = np.array([fine[t:t + 20].sum(axis=0) for t in range(0, 181, 5)])
    valid = (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)
    x, y = [], []
    for t in range(0, n - 4, 4):
        if valid[t:t + 5].all():
            x.append((points[t, 0] + points[t + 4, 0]) / 300. - 1)
            y.append(abs(points[t + 4, 0] - points[t, 0]) / 20.)
    x, y = np.array(x), np.array(y)
    np.testing.assert_allclose(moments[0, idx], [x.mean(), y.mean(), (x*x).mean(), (x*y).mean()])
    assert diag[0, idx, 0] == len(x)


def test_five_event_gate_and_spatial_variance_are_distinct():
    moments = np.full((10, 8, 4), np.nan)
    diag = np.zeros_like(moments, dtype=int)
    for i in range(10):
        x = -.9 + .2 * i
        moments[i, ::2] = [x, 1 + .3*x, x*x + .1, x*(1 + .3*x) + .03]
        diag[i, ::2] = [3, 1, 13, 13]
    moments[6:, 1::2] = moments[6:, ::2]
    diag[6:, 1::2] = diag[6:, ::2]
    result = prefix_panels(moments, diag, [5, 10], 12)
    assert result[result.selection.eq("selected")].statistic_status.eq("fewer_than_five_events").all()
    np.testing.assert_allclose(result[result.selection.eq("all")].statistic, .3)
    flat = np.tile([0., 1., 0., 0.], (10, 8, 1))
    flat_diag = np.tile([3, 1, 13, 13], (10, 8, 1))
    assert prefix_panels(flat, flat_diag, [10], 12).statistic_status.eq("insufficient_spatial_variance").all()
    with pytest.raises(ValueError, match="budgets"):
        prefix_panels(moments, diag, [10, 5], 12)
    broken = moments.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="availability"):
        validate_event_statistics(broken, diag)


def test_nested_runner_separates_budget_fits_and_saves_reconstructable_moments(tmp_path):
    model_path = tmp_path / "model.npz"
    np.savez_compressed(model_path, **tiny_model())
    args = Namespace(output_dir=tmp_path, budgets=[5, 10], seed=12, fit_draws=20, calibration_draws=19, test_draws=1, fixed_draws=1)
    row = {"dataset": "synthetic", "animal": "a", "session": "s", "label": "test"}
    batch, summary = score_session((row, model_path, profiles(), args))
    panels = pd.read_csv(tmp_path / "panels__test.csv.gz", float_precision="round_trip")
    schedule = pd.read_csv(tmp_path / "schedule__test.csv", float_precision="round_trip")
    assert batch["panel_rows"] == (39 + 4 * 6) * 8 * 2
    for budget, part in panels.groupby("candidate_budget"):
        validate_panel_contract(part, schedule, {k: row[k] for k in ["dataset", "animal", "session"]})
        _, expected = evaluate_panels(part)
        fits = pd.read_csv(tmp_path / "local_fits__test.csv.gz", float_precision="round_trip")
        actual = fits[fits.candidate_budget.eq(budget)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)
    with np.load(tmp_path / "events__test.npz", allow_pickle=False) as handle:
        assert handle["moments"].shape == (63, 10, 8, 4)
        first = panels[panels.event_statistics_row.eq(0)]
        recon = prefix_panels(handle["moments"][0], handle["diagnostics"][0], [5, 10], int(first.bootstrap_seed.iloc[0], 16), False)
        np.testing.assert_allclose(first.statistic, recon.statistic, equal_nan=True)
    assert set(summary.candidate_budget) == {5, 10}
    assert summary.panels.gt(0).all()


def test_simulated_condition_metadata_paths_and_prefix_hashes():
    panel, moments, diag, plan = simulate_nested(tiny_model(), profiles(), draw(), [5, 10])
    assert len(panel) == 4 * 8 * 2 and moments.shape == (4, 10, 8, 4) and diag.shape == moments.shape
    assert len(plan) == 10
    assert panel.groupby("candidate_budget").path_sha256.nunique().eq(1).all()
    assert panel.groupby(["generator", "observation", "candidate_budget"]).counts_sha256.nunique().eq(1).all()
    assert panel.groupby(["generator", "observation"]).path_sha256.nunique().eq(2).all()
    assert panel.source_events.eq(panel.candidate_budget).all()


def test_independent_moment_recount_rejects_rehashed_statistic_corruption():
    panel, moments, diag, _ = simulate_nested(tiny_model(), profiles(), draw(), [5, 10])
    part = panel[panel.condition_index.eq(0)].copy()
    seed = int(part.bootstrap_seed.iloc[0], 16)
    verify_moment_panels(part, moments[0], diag[0], [5, 10], seed, True)
    part.loc[part.index[0], "statistic"] = 999.
    with pytest.raises(AssertionError):
        verify_moment_panels(part, moments[0], diag[0], [5, 10], seed, True)


def synthetic_source(root):
    from itertools import product

    from scripts.calibrate_replay_speed_identifiability import TEST_CONDITIONS, summarize_decisions

    root.mkdir()
    batches, summaries, duration_tables = [], [], []
    for animal in ["a", "b"]:
        identity = {"dataset": "synthetic", "animal": animal, "session": f"{animal}/run"}
        label = f"synthetic__{animal}"
        schedule = draw_schedule(20260914, ":".join(identity.values()), 1, 1, 1, 1)
        schedule.to_csv(root / f"schedule__{label}.csv", index=False)
        rows = []
        for d in schedule.to_dict("records"):
            conditions = TEST_CONDITIONS if d["phase"] == "test" else [("A", "poisson")]
            for (generator, observation), readout in product(conditions, READOUTS):
                rows.append({**identity, **d, **dict(zip(READOUT, readout, strict=True)),
                    "generator": generator, "observation": observation, "statistic": d["gradient"],
                    "naive_lower": d["gradient"] - .2, "naive_upper": d["gradient"] + .2})
        panel = pd.DataFrame(rows)
        panel.to_csv(root / f"panels__{label}.csv.gz", index=False)
        dec, _ = evaluate_panels(panel)
        summaries.append(summarize_decisions(dec))
        durations = profiles()
        for name, value in identity.items():
            durations[name] = value
        duration_tables.append(durations)
        batches.append({**identity, "label": label, "status": "scored", "profiles": len(durations), "panel_rows": len(panel)})
        np.savez_compressed(root / f"model__{label}.npz", **tiny_model())
    profile_path = root / "profiles.csv"
    pd.concat(duration_tables, ignore_index=True).to_csv(profile_path, index=False)
    pd.DataFrame(batches).to_csv(root / "speed_identifiability_batches.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(root / "speed_identifiability_session_summary.csv", index=False)
    paths = {"profiles": profile_path, "replay_speed_identifiability": ROOT / "src/hipporeplayimm/replay_speed_identifiability.py",
        "script": ROOT / "scripts/calibrate_replay_speed_identifiability.py"}
    meta = {"status": "complete", "sessions": 2, "parameters": {"max_profiles": None},
        "input_file_paths": {k: str(v) for k, v in paths.items()}, "input_file_sha256": {k: file_sha256(v) for k, v in paths.items()},
        "output_sha256": {p.name: file_sha256(p) for p in root.iterdir()}}
    manifest = root / "speed_identifiability_manifest.json"
    manifest.write_text(json.dumps(meta))
    (root / "speed_identifiability_reconstruction_audit.json").write_text(json.dumps(
        {"status": "pass", "input_file_sha256": {"scoring_manifest": file_sha256(manifest)}}))


def test_full_runner_audit_and_nonvacuous_source_contract(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    synthetic_source(source)
    args = Namespace(input_dir=source, output_dir=out, budgets=[5, 10], seed=12,
        fit_draws=20, calibration_draws=19, test_draws=1, fixed_draws=1, sessions=None, workers=2, allow_dirty=True)
    run(args)
    audit(out, workers=2)
    meta = json.loads((out / "speed_panel_size_manifest.json").read_text())
    assert meta["status"] == "complete" and meta["scope"] == "technical_smoke"
    gates = pd.read_csv(out / "speed_panel_size_gate_summary.csv").set_index("gate").passed
    assert gates.overall_technical and not gates.full_33_session_scope
    checked = json.loads((out / "speed_panel_size_reconstruction_audit.json").read_text())
    assert checked["status"] == "pass" and checked["sessions"] == 2 and checked["transfer_intervals_verified"] == 1536
    report(out, tmp_path / "report", bootstraps=20)
    assert (tmp_path / "report/speed_panel_size_availability.png").stat().st_size > 1000
    assert "CORRECT-SIGN" in (tmp_path / "report/speed_panel_size_report.md").read_text()
    target = out / "panels__synthetic__a.csv.gz"
    changed = pd.read_csv(target, float_precision="round_trip").iloc[:-1]
    changed.to_csv(target, index=False)
    meta["output_sha256"][target.name] = file_sha256(target)
    (out / "speed_panel_size_manifest.json").write_text(json.dumps(meta))
    with pytest.raises((AssertionError, ValueError)):
        audit(out, workers=2)


def test_paired_budget_means_are_equal_animal_not_event_or_session_weighted():
    rows = []
    for animal, sessions, larger in [("a", 3, .8), ("b", 1, .4)]:
        for session in range(sessions):
            for budget, value in [(30, .2), (100, larger)]:
                rows.append({"dataset": "d", "animal": animal, "session": str(session), "candidate_budget": budget,
                    "group": "test", "panels": 100, "coverage": value})
    table = pd.DataFrame(rows)
    paired = paired_budgets(table, ["group"], ["coverage"])
    animals, summary = aggregate_events(paired, ["budget_comparison", "group"], ["coverage"], bootstraps=20)
    assert len(animals) == 2 and summary.equal_animal_mean.item() == pytest.approx(.4)
    table.loc[0, "panels"] = 99
    with pytest.raises(ValueError, match="denominators"):
        paired_budgets(table, ["group"], ["coverage"])


def test_wrong_sign_confidence_is_not_successful_recovery():
    rows = []
    for g, lo, hi in [(.5, .1, .7), (.5, -.7, -.1), (-.5, -.7, -.1), (-.5, .1, .7), (.5, -np.inf, np.inf), (0., .1, .7)]:
        rows.append({"dataset": "d", "animal": "a", "session": "s", "candidate_budget": 30,
            "estimator": "posterior_mean", "bin_filter": "unfiltered", "selection": "all", "generator": "A",
            "observation": "poisson", "stratum": "uniform", "method": "inverse_conformal", "calibration_scope": "within_session",
            "gradient": g, "lower": lo, "upper": hi})
    summary = directional_endpoints(pd.DataFrame(rows)).iloc[0]
    assert summary.correct_sign_nonzero_fraction == pytest.approx(2 / 6)
    assert summary.wrong_sign_nonzero_fraction == pytest.approx(2 / 6)
