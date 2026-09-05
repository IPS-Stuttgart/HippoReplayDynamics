#!/usr/bin/env python3
"""Reconstruct nested-panel statistics, sampled observations and all intervals."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_speed_identifiability import bootstrap_slope
from hipporeplayimm.replay_speed_panel_size import READOUTS, validate_event_statistics
from hipporeplayimm.replay_speed_population_transfer import GROUP, READOUT, panel_digest

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_replay_speed_population_transfer import independent_fit, verify_intervals
from scripts.calibrate_replay_speed_identifiability import (
    IDENTITY,
    PANEL_KEY,
    TEST_CONDITIONS,
    decode_panel,
    draw_schedule,
    summarize_decisions,
    validate_panel_contract,
)
from scripts.calibrate_replay_speed_panel_size import draw_observations, prefix_digest, profile_plan, source_inputs
from scripts.simulate_replay_coverage_recovery import seed_parts


def compare_tables(actual, expected, keys):
    actual, expected = [frame.sort_values(keys).reset_index(drop=True) for frame in [actual, expected]]
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, rtol=1e-9, atol=1e-10)


def verify_moment_panels(panels, moments, diagnostics, budgets, seed, bootstrap):
    validate_event_statistics(moments, diagnostics)
    if len(panels) != 8 * len(budgets) or panels.duplicated(["candidate_budget"] + READOUT).any():
        raise AssertionError("incomplete budget/readout statistics")
    lookup = panels.set_index(["candidate_budget"] + READOUT)
    for budget in budgets:
        for column, key in enumerate(READOUTS):
            row = lookup.loc[(budget, *key)]
            values = moments[:budget, column]
            finite = np.isfinite(values).all(axis=1)
            n = int(finite.sum())
            values = values[finite]
            qvar, statistic = np.nan, np.nan
            if n:
                sums = np.sum(values, axis=0) / n
                qvar = sums[2] - sums[0] ** 2
                if n >= 5 and qvar >= .01:
                    statistic = (sums[3] - sums[0] * sums[1]) / qvar
            np.testing.assert_allclose([row.statistic, row.spatial_variance], [statistic, qvar], rtol=1e-9, atol=1e-10, equal_nan=True)
            assert row.source_events == budget and row.contributing_events == n
            np.testing.assert_array_equal(row[["contributing_steps", "continuous_events", "valid_windows", "total_windows"]].to_numpy(int), diagnostics[:budget, column].sum(axis=0))
            expected_status = "available" if np.isfinite(statistic) else "fewer_than_five_events" if n < 5 else "insufficient_spatial_variance"
            assert row.statistic_status == expected_status
            lower, upper = np.nan, np.nan
            if bootstrap:
                _, lower, upper = bootstrap_slope(moments[:budget, column], [seed, budget, column])
            np.testing.assert_allclose([row.naive_lower, row.naive_upper], [lower, upper], rtol=1e-9, atol=1e-10, equal_nan=True)


def verify_raw(rows):
    x, lo, hi, g = [rows[name].to_numpy() for name in ["statistic", "naive_lower", "naive_upper", "gradient"]]
    np.testing.assert_allclose(rows[["estimate", "lower", "upper"]].to_numpy(), np.column_stack([x, lo, hi]), equal_nan=True)
    finite = np.isfinite(lo) & np.isfinite(hi)
    np.testing.assert_array_equal(rows.finite_interval, finite)
    np.testing.assert_array_equal(rows.covered, (lo <= g) & (hi >= g))
    np.testing.assert_allclose(rows.interval_width, hi - lo)
    nonzero = finite & ((lo > 0) | (hi < 0))
    np.testing.assert_array_equal(rows.nonzero_claim, nonzero)
    for bound in [.10, .25, .50]:
        inside, claim = np.abs(g) < bound, finite & (lo > -bound) & (hi < bound)
        np.testing.assert_array_equal(rows[f"inside_{bound:.2f}"], inside)
        np.testing.assert_array_equal(rows[f"equivalence_{bound:.2f}"], claim)
        if bound == .25:
            np.testing.assert_array_equal(rows.equivalence_claim, claim)
            np.testing.assert_array_equal(rows.truth_inside_equivalence, inside)
            np.testing.assert_array_equal(rows.false_equivalence, claim & ~inside)
            np.testing.assert_array_equal(rows.decision, np.select([claim, nonzero, finite], ["equivalent_in_surrogate", "nonzero_in_surrogate", "inconclusive"], default="abstain"))


def check_fits_and_decisions(panels, fits, decisions, transfer=False):
    keys = (["dataset", "heldout_animal"] if transfer else IDENTITY) + READOUT
    assert not fits.duplicated(keys).any()
    assert not decisions.duplicated(PANEL_KEY + ["method"]).any()
    test = panels[panels.phase.eq("test")]
    methods = ["inverse_gaussian", "inverse_conformal"] + ([] if transfer else ["raw_bootstrap"])
    assert set(decisions.method) == set(methods)
    assert decisions.calibration_scope.eq("leave_one_animal_out" if transfer else "within_session").all()
    for method in methods:
        compare_tables(decisions[decisions.method.eq(method)], test, PANEL_KEY)
    verified = 0
    for record in fits.to_dict("records"):
        matching = panels.dataset.eq(record["dataset"])
        for name in READOUT:
            matching &= panels[name].eq(record[name])
        heldout = record["heldout_animal"] if transfer else record["animal"]
        fitting = matching & ~panels.animal.eq(heldout) if transfer else matching & panels.animal.eq(heldout) & panels.session.eq(record["session"])
        source = panels[fitting].sort_values(PANEL_KEY)
        training, calibration = [source[source.phase.eq(phase)] for phase in ["fit", "calibration"]]
        assert len(training) and len(calibration)
        assert source[source.phase.ne("test")].generator.eq("A").all()
        assert source[source.phase.ne("test")].observation.eq("poisson").all()
        if transfer:
            assert heldout not in set(training.animal) | set(calibration.animal)
            assert set(training.animal) == set(json.loads(record["source_animals"]))
            assert set(map(tuple, json.loads(record["source_sessions"]))) == set(zip(training.animal, training.session, strict=True))
            assert record["fit_input_sha256"] == panel_digest(training)
            assert record["calibration_input_sha256"] == panel_digest(calibration)
            assert record["n_fit_scheduled"] == len(training)
        expected = independent_fit(training, calibration)
        assert record["status"] == expected["status"] and record["n_fit"] == expected["n_fit"]
        assert record["n_calibration"] == len(calibration) and record["finite_calibration"] == np.isfinite(calibration.statistic).sum()
        names = [name for name in expected if name not in ["status", "n_fit"]]
        np.testing.assert_allclose([record[name] for name in names], [expected[name] for name in names], rtol=1e-9, atol=1e-9, equal_nan=True)
        choose = decisions.dataset.eq(record["dataset"]) & decisions.animal.eq(heldout)
        if not transfer:
            choose &= decisions.session.eq(record["session"])
        for name in READOUT:
            choose &= decisions[name].eq(record[name])
        verify_intervals(decisions[choose & decisions.method.ne("raw_bootstrap")], expected)
        verified += int(choose.sum())
    assert verified == len(decisions), "missing or extra fit membership"
    if not transfer:
        verify_raw(decisions[decisions.method.eq("raw_bootstrap")])


def audit_session(payload):
    root, meta, row = payload
    label, args = row["label"], meta["parameters"]
    identity = {name: row[name] for name in IDENTITY}
    def read(name):
        return pd.read_csv(root / f"{name}__{label}.csv.gz", float_precision="round_trip")
    panels = read("panels")
    schedule = draw_schedule(args["seed"], ":".join(identity.values()), args["fit_draws"], args["calibration_draws"], args["test_draws"], args["fixed_draws"])
    compare_tables(pd.read_csv(root / f"schedule__{label}.csv", float_precision="round_trip"), schedule, ["phase", "draw_id"])
    for _, part in panels.groupby("candidate_budget"):
        validate_panel_contract(part, schedule, identity)
    assert set(panels.candidate_budget) == set(args["budgets"])
    assert len(panels) == row["panel_rows"]
    profiles = pd.read_csv(root / f"profiles__{label}.csv")
    source_profiles = pd.read_csv(meta["input_file_paths"]["profiles"], usecols=IDENTITY + ["source_event_index", "n_base_bins"]).drop_duplicates()
    selected = np.logical_and.reduce([source_profiles[k].eq(v) for k, v in identity.items()])
    compare_tables(profiles, source_profiles[selected], ["source_event_index"])
    plans = read("duration_plan")
    assert not plans.duplicated(["phase", "draw_id", "synthetic_ordinal"]).any()
    assert len(plans) == len(schedule) * max(args["budgets"])
    with np.load(meta["input_file_paths"][f"model__{label}"], allow_pickle=False) as handle:
        model = dict(handle)
    with np.load(root / f"events__{label}.npz", allow_pickle=False) as handle:
        moments, diagnostics = handle["moments"], handle["diagnostics"]
    assert moments.shape == diagnostics.shape == (row["event_statistic_rows"], max(args["budgets"]), 8, 4)
    samples = set(map(tuple, schedule.groupby(["phase", "stratum"], sort=True).head(1)[["phase", "draw_id"]].to_numpy()))
    counter, sampled_rows = 0, 0
    for draw in schedule.to_dict("records"):
        panel = panels[panels.phase.eq(draw["phase"]) & panels.draw_id.eq(draw["draw_id"])]
        plan = profile_plan(profiles, max(args["budgets"]), draw)
        observed_plan = plans[plans.phase.eq(draw["phase"]) & plans.draw_id.eq(draw["draw_id"])]
        compare_tables(observed_plan, plan, ["synthetic_ordinal"])
        sampled = (draw["phase"], draw["draw_id"]) in samples
        chunks, hashes = None, None
        if sampled:
            _, chunks, hashes = draw_observations(model, profiles, draw, max(args["budgets"]))
            assert len(set(hashes)) == len(hashes), "reused latent paths"
        for i, (generator, observation) in enumerate(TEST_CONDITIONS if draw["phase"] == "test" else [("A", "poisson")]):
            part = panel[panel.generator.eq(generator) & panel.observation.eq(observation)]
            assert part.condition_index.eq(i).all() and part.event_statistics_row.eq(counter).all()
            assert part.bootstrap_seed.nunique() == 1
            expected_seed = int(np.random.SeedSequence(seed_parts(int(draw["draw_seed"], 16), f"{generator}:{observation}", stream=23)).generate_state(1, dtype=np.uint64)[0])
            assert part.bootstrap_seed.eq(hex(expected_seed)).all()
            verify_moment_panels(part, moments[counter], diagnostics[counter], args["budgets"], int(part.bootstrap_seed.iloc[0], 16), draw["phase"] == "test")
            if sampled:
                fine = chunks[generator, observation]
                count_hashes = [array_sha256(c) for c in fine]
                for budget in args["budgets"]:
                    actual = part[part.candidate_budget.eq(budget)]
                    assert actual.path_sha256.eq(prefix_digest(hashes, budget)).all()
                    assert actual.counts_sha256.eq(prefix_digest(count_hashes, budget)).all()
                    assert actual.spikes.eq(sum(int(c.sum()) for c in fine[:budget])).all()
                    old = decode_panel(fine[:budget], model, 1, False).drop(columns=["naive_lower", "naive_upper"])
                    compare_tables(actual, old, READOUT)
                    sampled_rows += len(actual)
            counter += 1
    assert counter == row["event_statistic_rows"]
    decisions, fits = read("local_decisions"), read("local_fits")
    assert len(decisions) == row["local_decision_rows"]
    summaries = []
    for budget, part in panels.groupby("candidate_budget", sort=True):
        dec = decisions[decisions.candidate_budget.eq(budget)]
        check_fits_and_decisions(part, fits[fits.candidate_budget.eq(budget)], dec)
        summary = summarize_decisions(dec)
        summary["candidate_budget"], summary["calibration_scope"] = budget, "within_session"
        summaries.append(summary)
    result = {**identity, "panel_rows_verified": len(panels), "local_intervals_verified": len(decisions),
        "sampled_draws": len(samples), "sampled_decoding_rows": sampled_rows}
    print(json.dumps(result), flush=True)
    return result, pd.concat(summaries, ignore_index=True)


def audit(root, workers=8):
    manifest = root / "speed_panel_size_manifest.json"
    meta = json.loads(manifest.read_text())
    assert meta["status"] == "complete"
    for key, path in meta["input_file_paths"].items():
        assert file_sha256(path) == meta["input_file_sha256"][key], key
    for name, digest in {**meta["output_sha256"], **meta["snapshot_sha256"]}.items():
        assert file_sha256(root / name) == digest, name
    source, _, _, _ = source_inputs(Path(meta["parameters"]["input_dir"]))
    batches = pd.read_csv(root / "speed_panel_size_batches.csv")
    assert len(batches) == meta["sessions"] and not batches.duplicated(IDENTITY).any() and batches.status.eq("scored").all()
    if meta["parameters"]["sessions"]:
        source = source[source.apply(lambda r: ":".join(r[IDENTITY]) in meta["parameters"]["sessions"], axis=1)]
    compare_tables(batches[IDENTITY], source[IDENTITY], IDENTITY)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        checked = list(pool.map(audit_session, [(root, meta, r) for r in batches.to_dict("records")]))
    summaries = [s for _, s in checked]
    panels = pd.concat([pd.read_csv(root / f"panels__{r.label}.csv.gz", float_precision="round_trip") for r in batches.itertuples()], ignore_index=True)
    transferred = 0
    for budget, part in panels.groupby("candidate_budget", sort=True):
        fits = pd.read_csv(root / f"transfer_fits__{budget}.csv", float_precision="round_trip")
        decisions = pd.read_csv(root / f"transfer_decisions__{budget}.csv.gz", float_precision="round_trip")
        assert fits.candidate_budget.eq(budget).all()
        check_fits_and_decisions(part, fits, decisions, transfer=True)
        summary = summarize_decisions(decisions)
        summary["candidate_budget"], summary["calibration_scope"] = budget, "leave_one_animal_out"
        summaries.append(summary)
        transferred += len(decisions)
    expected = pd.concat(summaries, ignore_index=True)
    summary = pd.read_csv(root / "speed_panel_size_session_summary.csv", float_precision="round_trip")
    compare_tables(summary, expected, IDENTITY + GROUP + ["candidate_budget"])
    assert summary.panels.gt(0).all() and (summary.finite_panels <= summary.panels).all()
    records = pd.DataFrame([r for r, _ in checked])
    records.to_csv(root / "speed_panel_size_reconstruction_audit.csv", index=False)
    output = build_script_provenance(input_paths={"manifest": manifest, "auditor": Path(__file__)}, cwd=ROOT)
    output.update(status="pass", sessions=len(records), panel_rows_verified=int(records.panel_rows_verified.sum()),
        local_intervals_verified=int(records.local_intervals_verified.sum()), transfer_intervals_verified=transferred,
        sampled_decoding_rows=int(records.sampled_decoding_rows.sum()),
        audit_scope="all hashes/profile plans; independent equal-event moments and inverse fits/intervals; all raw-bootstrap reconstruction; sampled regenerated paths/counts and old decoder; shared summary reconstruction, not biological ground truth")
    (root / "speed_panel_size_reconstruction_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ["status", "sessions", "panel_rows_verified", "local_intervals_verified", "transfer_intervals_verified"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("positive worker count required")
    audit(args.input_dir.resolve(), args.workers)
