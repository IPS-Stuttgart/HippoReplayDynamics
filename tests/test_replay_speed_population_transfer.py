"""Synthetic panels test exclusion, abstention, provenance and reconstruction."""

import hashlib
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from hipporeplayimm.replay_speed_identifiability import interval_decision, inverse_intervals
from hipporeplayimm.replay_speed_population_transfer import (
    FIT_KEY,
    METRICS,
    READOUT,
    apply_model,
    evaluate_transfers,
    fit_transfers,
    paired_comparison,
    summarize_animals,
)
from scripts._provenance import file_sha256
from scripts.audit_replay_speed_population_transfer import audit, independent_fit
from scripts.calibrate_replay_speed_identifiability import draw_schedule, evaluate_panels, summarize_decisions
from scripts.report_replay_speed_population_transfer import report
from scripts.validate_replay_speed_population_transfer import ROOT as REPO_ROOT
from scripts.validate_replay_speed_population_transfer import load_source, run


def synthetic_panels():
    frames, schedules = [], {}
    for a, animal in enumerate(["a", "b", "c"]):
        session = f"{animal}/run"
        schedule = draw_schedule(20260914, f"synthetic:{animal}:{session}", 24, 24, 3, 1)
        schedules[animal] = schedule
        for draw in schedule.to_dict("records"):
            rng = np.random.default_rng(int(draw["draw_seed"], 16))
            conditions = [("A", "poisson"), ("A", "shared_gain"), ("B", "poisson"), ("B", "shared_gain")] if draw["phase"] == "test" else [("A", "poisson")]
            for generator, observation in conditions:
                statistic = .5 * draw["gradient"] + .05 * a + rng.normal(0, .04) + .08 * (generator == "B")
                digest = hashlib.sha256(f"{animal}:{draw['draw_seed']}:{generator}:{observation}".encode()).hexdigest()
                for estimator in ["map", "posterior_mean"]:
                    for support in ["unfiltered", "at_least_2cells_3spikes"]:
                        for selection in ["all", "selected"]:
                            frames.append({"dataset": "synthetic", "animal": animal, "session": session, **draw,
                                "generator": generator, "observation": observation, "statistic": statistic,
                                "estimator": estimator, "bin_filter": support, "selection": selection,
                                "naive_lower": statistic - .15, "naive_upper": statistic + .15,
                                "source_events": 8, "contributing_events": 8, "continuous_events": 8,
                                "valid_windows": 80, "total_windows": 80, "spikes": 200,
                                "path_sha256": digest, "counts_sha256": digest})
    return pd.DataFrame(frames), schedules


def fixture(root):
    panels, schedules = synthetic_panels()
    batches, summaries = [], []
    for animal, data in panels.groupby("animal", sort=True):
        label = f"synthetic__{animal}"
        data.to_csv(root / f"panels__{label}.csv.gz", index=False)
        schedules[animal].to_csv(root / f"schedule__{label}.csv", index=False)
        decisions, _ = evaluate_panels(data)
        summaries.append(summarize_decisions(decisions))
        batches.append({"dataset": "synthetic", "animal": animal, "session": f"{animal}/run",
            "label": label, "status": "scored", "panel_rows": len(data)})
    pd.DataFrame(batches).to_csv(root / "speed_identifiability_batches.csv", index=False)
    pd.concat(summaries).to_csv(root / "speed_identifiability_session_summary.csv", index=False)
    meta = {"status": "complete", "sessions": 3, "parameters": {"max_profiles": 5},
        "output_sha256": {p.name: file_sha256(p) for p in root.iterdir()},
        "input_file_sha256": {"replay_speed_identifiability": file_sha256(REPO_ROOT / "src/hipporeplayimm/replay_speed_identifiability.py"),
            "script": file_sha256(REPO_ROOT / "scripts/calibrate_replay_speed_identifiability.py")}}
    path = root / "speed_identifiability_manifest.json"
    path.write_text(json.dumps(meta))
    (root / "speed_identifiability_reconstruction_audit.json").write_text(json.dumps(
        {"status": "pass", "input_file_sha256": {"scoring_manifest": file_sha256(path)}}))
    return panels


def test_heldout_animal_fit_and_calibration_cannot_change_its_transfer_fit():
    panels, _ = synthetic_panels()
    expected = fit_transfers(panels)
    changed = panels.copy()
    excluded = changed.animal.eq("a") & ~changed.phase.eq("test")
    changed.loc[excluded, "statistic"] = 10000
    changed.loc[excluded, "gradient"] = -9999
    actual = fit_transfers(changed)
    pd.testing.assert_frame_equal(expected[expected.heldout_animal.eq("a")], actual[actual.heldout_animal.eq("a")])
    assert not actual[actual.heldout_animal.eq("a")].source_animals.str.contains('"a"').any()


def test_test_outcomes_do_not_change_any_fit_and_row_order_is_irrelevant():
    panels, _ = synthetic_panels()
    expected = fit_transfers(panels)
    panels.loc[panels.phase.eq("test"), "statistic"] = 999
    panels.loc[panels.phase.eq("test"), "gradient"] = -999
    pd.testing.assert_frame_equal(expected, fit_transfers(panels.sample(frac=1, random_state=4)))


def test_contaminated_duplicate_or_single_animal_panels_fail():
    panels, _ = synthetic_panels()
    bad = panels.copy()
    bad.loc[bad.phase.eq("fit"), "generator"] = "B"
    with pytest.raises(ValueError, match="B/gain"):
        fit_transfers(bad)
    with pytest.raises(ValueError, match="unique"):
        fit_transfers(pd.concat([panels, panels.iloc[:1]]))
    with pytest.raises(ValueError, match="other animals"):
        fit_transfers(panels[panels.animal.eq("a")])


def test_missing_calibration_retains_abstention_not_vacuous_identifiability():
    panels, _ = synthetic_panels()
    panels.loc[~panels.animal.eq("a") & panels.phase.eq("calibration"), "statistic"] = np.nan
    fits = fit_transfers(panels)
    model = fits[fits.heldout_animal.eq("a")].iloc[0].to_dict()
    assert np.isinf(model["calibration_radius"]) and model["status"] == "fitted"
    chosen = panels.animal.eq("a") & panels.phase.eq("test")
    for name in READOUT:
        chosen &= panels[name].eq(model[name])
    scores = apply_model(panels[chosen], model)
    conformal = scores[scores.method.eq("inverse_conformal")]
    assert conformal.covered.all() and not conformal.finite_interval.any()
    assert conformal.decision.eq("abstain").all() and not conformal.equivalence_claim.any()


def test_vector_intervals_match_frozen_scalar_baseline_and_independent_fit():
    panels, _ = synthetic_panels()
    fits = fit_transfers(panels)
    scores = evaluate_transfers(panels, fits)
    lookup = fits.set_index(FIT_KEY)
    for row in scores.sample(80, random_state=2).to_dict("records"):
        model = lookup.loc[(row["dataset"], row["animal"], *(row[k] for k in READOUT))].to_dict()
        expected = inverse_intervals(model, model["calibration_radius"], row["statistic"])[row["method"]]
        np.testing.assert_allclose([row[k] for k in ["estimate", "lower", "upper"]], expected)
        for name, value in interval_decision(expected[1], expected[2], row["gradient"], .25).items():
            assert row[name] == pytest.approx(value) if isinstance(value, float) else row[name] == value
    model = fits.iloc[0].to_dict()
    others = panels[~panels.animal.eq(model["heldout_animal"])]
    for name in READOUT:
        others = others[others[name].eq(model[name])]
    expected = independent_fit(others[others.phase.eq("fit")], others[others.phase.eq("calibration")])
    for name in ["intercept", "slope", "residual_sd", "calibration_radius"]:
        assert model[name] == pytest.approx(expected[name])
    with pytest.raises(ValueError, match="wrong excluded animal"):
        apply_model(panels[panels.phase.eq("test")], model)


def test_insufficient_fits_produce_unbounded_intervals():
    panels, _ = synthetic_panels()
    panels.loc[panels.phase.eq("fit"), "statistic"] = np.nan
    fits = fit_transfers(panels)
    scores = evaluate_transfers(panels, fits)
    assert fits.status.eq("insufficient_fit").all()
    assert scores.decision.eq("abstain").all() and scores.covered.all()


def test_equal_animal_summary_and_reference_pairing():
    rows = []
    for animal, n, local, transfer in [("a", 3, .8, .4), ("b", 1, .6, .4)]:
        for session in range(n):
            for scope, value in [("within_session", local), ("leave_one_animal_out", transfer)]:
                row = {"dataset": "d", "animal": animal, "session": str(session), "estimator": "posterior_mean",
                    "bin_filter": "unfiltered", "selection": "all", "generator": "A", "observation": "poisson",
                    "stratum": "uniform", "method": "inverse_conformal", "calibration_scope": scope,
                    "panels": 100 if animal == "a" else 20, "finite_panels": 10}
                row.update(dict.fromkeys(METRICS, value))
                rows.append(row)
    sessions = pd.DataFrame(rows)
    _, summary = summarize_animals(sessions, bootstraps=40)
    mean = summary[summary.metric.eq("coverage") & summary.calibration_scope.eq("within_session")].equal_animal_mean.item()
    assert mean == pytest.approx(.7)
    paired = paired_comparison(sessions, bootstraps=40)
    assert paired[paired.metric.eq("coverage")].transfer_minus_local.item() == pytest.approx(-.3)
    sessions.loc[0, "panels"] = 17
    with pytest.raises(ValueError, match="denominators"):
        paired_comparison(sessions, bootstraps=40)


def test_source_hash_and_full_cli_audit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    fixture(source)
    out = tmp_path / "transfer"
    run(Namespace(input_dir=source, output_dir=out, seed=20260916, bootstraps=20, allow_dirty=True))
    audit(out)
    gates = pd.read_csv(out / "speed_population_transfer_gate_summary.csv").set_index("gate").passed
    assert gates.overall_technical and not gates.full_33_session_scope
    checked = json.loads((out / "speed_population_transfer_reconstruction_audit.json").read_text())
    assert checked["status"] == "pass" and checked["fits_verified"] == 24
    report(out, tmp_path / "report")
    assert (tmp_path / "report/speed_population_transfer_coverage.png").stat().st_size > 1000
    assert "no asserted finite-sample" in (tmp_path / "report/speed_population_transfer_report.md").read_text()
    changed_path = out / "speed_population_transfer_decisions.csv.gz"
    changed = pd.read_csv(changed_path, float_precision="round_trip")
    index = changed[changed.finite_interval].index[0]
    changed.loc[index, "lower"] -= 1
    changed.to_csv(changed_path, index=False)
    manifest = out / "speed_population_transfer_manifest.json"
    meta = json.loads(manifest.read_text())
    meta["output_sha256"][changed_path.name] = file_sha256(changed_path)
    manifest.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        audit(out)
    path = source / "panels__synthetic__a.csv.gz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash changed"):
        load_source(source)


def test_missing_panel_fails_even_with_updated_source_hashes(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "panels__synthetic__a.csv.gz"
    pd.read_csv(path).iloc[1:].to_csv(path, index=False)
    manifest = tmp_path / "speed_identifiability_manifest.json"
    meta = json.loads(manifest.read_text())
    meta["output_sha256"][path.name] = file_sha256(path)
    manifest.write_text(json.dumps(meta))
    (tmp_path / "speed_identifiability_reconstruction_audit.json").write_text(json.dumps(
        {"status": "pass", "input_file_sha256": {"scoring_manifest": file_sha256(manifest)}}))
    with pytest.raises(ValueError, match="missing or duplicate"):
        load_source(tmp_path)


def test_missing_or_duplicated_fits_cannot_silently_drop_test_panels():
    panels, _ = synthetic_panels()
    fits = fit_transfers(panels)
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_transfers(panels, fits.iloc[:-1])
    with pytest.raises(ValueError, match="unique"):
        evaluate_transfers(panels, pd.concat([fits, fits.iloc[:1]]))
