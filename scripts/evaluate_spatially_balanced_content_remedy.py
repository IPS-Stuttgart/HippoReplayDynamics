#!/usr/bin/env python3
"""RUN-only balanced sampling versus random sampling, without event abstention."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_pf_independent_population_content import balanced_partitions, profile
from scripts.measure_population_content_stability import (
    decode, endpoint_counts, readouts, run_windows, seed_for, simulate_counts, tile_ids,
)

SEED = 20260914


def choose_partitions(n_cells, rates, grid, counts, truth, seed):
    """Fix a common universe, then choose balanced halves using RUN only."""
    if len(rates) != n_cells or counts.shape != (len(truth), n_cells):
        raise ValueError("RUN counts, cells and positions must align")
    if len(grid) < 2 or not np.isfinite(grid).all() or not np.isfinite(truth).all():
        raise ValueError("invalid RUN spatial support")
    perm = np.random.default_rng(seed).permutation(n_cells)
    universe = np.sort(perm[:2*(n_cells//2)])
    if len(universe) < 10:
        raise ValueError("fewer than five cells per population")
    half = len(universe)//2
    random = dict(a=np.sort(perm[:half]).tolist(), b=np.sort(perm[half:2*half]).tolist())
    tile = tile_ids(grid)
    candidates = balanced_partitions(universe, rates, grid, seed, n_candidates=128)
    for candidate in candidates:
        pa, pb = [profile(rates[candidate[side]], tile) for side in ("a", "b")]
        candidate["profile_loss"] = float(np.square(pa-pb).sum())
    shortlist = sorted(candidates, key=lambda p: (p["profile_loss"], p["candidate_id"]))[:8]
    truth_tile = np.clip(((truth-grid.min(axis=0))/np.maximum(np.ptp(grid, axis=0), 1)*3).astype(int), 0, 2)
    truth_tile = truth_tile[:, 0]*3+truth_tile[:, 1]
    for candidate in shortlist:
        error = {}
        for side in ("a", "b"):
            ix = candidate[side]
            error[side] = np.linalg.norm(decode(counts[:, ix], rates[ix], grid)["mean"]-truth, axis=1)
        differences = []
        for region in range(9):
            mask = truth_tile == region
            if mask.sum() >= 10:
                differences.append(abs(error["a"][mask].mean()-error["b"][mask].mean()))
        if not differences:
            raise ValueError("no locally supported calibration tile")
        candidate["run_loss"] = float(max(error["a"].mean(), error["b"].mean())+np.mean(differences))
    winner = min(shortlist, key=lambda p: (p["run_loss"], p["profile_loss"], p["candidate_id"]))
    assert set(random["a"]+random["b"]) == set(winner["a"]+winner["b"])
    return dict(random=random, balanced=winner, universe=universe.tolist()), shortlist


def simulated_case(actual, rates, drift, universe, grid, seed, source):
    """One shared observation vector, subsequently split two different ways."""
    if source not in ("sim_matched", "sim_drift"):
        raise ValueError("unknown conditional generator")
    rng = np.random.default_rng(seed)
    state = rng.integers(len(grid), size=len(actual))
    source_rates = rates if source == "sim_matched" else drift*rng.lognormal(0, .4, (len(rates), 1))
    counts = np.zeros_like(actual)
    counts[:, universe] = simulate_counts(actual[:, universe].sum(axis=1), source_rates[universe], state, rng)
    return counts, grid[state]


def measure(row, output, seed):
    if file_sha256(row.artifact_path) != row.artifact_sha256:
        raise ValueError("source cache changed")
    with np.load(row.artifact_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    mask = data["unit_qc_mask"].astype(bool)
    support = data["valid_spatial_bins"] & (data["occupancy_first_half_s"] >= .05)
    grid = data["bin_centers_cm"][support]
    if len(grid) < 2:
        raise ValueError("insufficient first-half spatial support")
    rates = np.maximum(data["rates_first_half_hz"][mask][:, support], 1e-4)
    drift = np.maximum(data["rates_second_half_hz"][mask][:, support], 1e-4)
    ids = data["cell_ids"][mask]
    q = np.linspace(data["supported_run_intervals"][:, 0].min(), data["supported_run_intervals"][:, 1].max(), 5)
    cal_w, cal_c = run_windows(data, ids, q[2], q[3])
    parts, shortlists = [], []
    for split in range(3):
        part, shortlisted = choose_partitions(len(ids), rates, grid, cal_c, cal_w[:, 2:], seed_for(seed, row.dataset, row.animal, row.session, split))
        parts.append(dict(split=split, **part))
        shortlists.extend(dict(split=split, **p) for p in shortlisted)
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(dataset=row.dataset, animal=row.animal, session=row.session, parts=parts,
                  cache_path=row.artifact_path, cache_sha256=row.artifact_sha256,
                  created_at_utc=datetime.now(UTC).isoformat(), cell_ids=ids.tolist(), no_replay_used=True,
                  seed=seed)
    frozen_path = output/"selected_before_test.json"
    frozen_path.write_text(json.dumps(frozen, indent=2)+"\n")
    freeze_hash = file_sha256(frozen_path)
    (output/"shortlist.json").write_text(json.dumps(shortlists, indent=2)+"\n")
    test_w, test_c = run_windows(data, ids, q[3], q[4])
    anchors, actual, excluded = endpoint_counts(data, mask)
    np.savez_compressed(output/"audit_arrays.npz", grid_cm=grid, rates_hz=rates, drift_hz=drift,
        calibration_windows=cal_w, calibration_counts=cal_c, test_windows=test_w, test_counts=test_c,
        endpoint_counts=actual, event_indices=anchors.event_index.to_numpy(), cell_ids=ids,
        endpoint_start_s=anchors.start_s.to_numpy(), endpoint_end_s=anchors.end_s.to_numpy())
    frames = []
    for part in parts:
        universe, split = part["universe"], part["split"]
        cases = [("real", -1, actual, None, anchors),
                 ("run_test", -1, test_c, test_w[:, 2:], pd.DataFrame(dict(
                     event_index=np.arange(len(test_c)), start_s=test_w[:, 0], end_s=test_w[:, 1])))]
        for source in ("sim_matched", "sim_drift"):
            for draw in range(2):
                counts, truth = simulated_case(actual, rates, drift, universe, grid,
                    seed_for(seed, row.dataset, row.animal, row.session, split, source, draw), source)
                cases.append((source, draw, counts, truth, anchors))
        for condition in ("random", "balanced"):
            a, b = part[condition]["a"], part[condition]["b"]
            assert len(a) == len(b) and not set(a) & set(b)
            cal_error = np.linalg.norm(decode(cal_c[:, a], rates[a], grid)["mean"]-cal_w[:, 2:], axis=1)
            for source, draw, counts, truth, identities in cases:
                local = readouts(counts[:, a], counts[:, b], rates[a], rates[b], grid, cal_w[:, 2:], cal_error, truth)
                for key in identities:
                    local[key] = identities[key].to_numpy()
                local["condition"], local["source"], local["draw"], local["split"] = condition, source, draw, split
                local["dataset"], local["animal"], local["session"] = row.dataset, row.animal, row.session
                local["pair_entropy"] = (local.a_entropy+local.b_entropy)/2
                local["pair_truth_error_cm"] = (local.a_truth_error_cm+local.b_truth_error_cm)/2
                frames.append(local)
    joined = pd.concat(frames, ignore_index=True)
    required = ["endpoint_separation_cm", "regional_tv", "pair_entropy", "a_entropy", "b_entropy"]
    if not np.isfinite(joined[required].to_numpy()).all():
        raise ValueError("nonfinite readouts")
    known = joined.loc[joined.source.ne("real"), ["a_truth_error_cm", "b_truth_error_cm"]]
    if not np.isfinite(known.to_numpy()).all():
        raise ValueError("nonfinite known-truth errors")
    joined.to_csv(output/"event_readouts.csv.gz", index=False)
    assert freeze_hash == file_sha256(frozen_path)
    (output/"excluded_endpoints.json").write_text(json.dumps(excluded, indent=2)+"\n")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session, status="complete",
                artifact_dir=str(output), n_cells=len(ids), candidates=len(anchors), source_candidates=int(row.candidates),
                readouts_sha256=file_sha256(output/"event_readouts.csv.gz"), frozen_sha256=freeze_hash)


def summarize(results, out, inputs_unchanged=True):
    good = [pd.read_csv(Path(r["artifact_dir"])/"event_readouts.csv.gz") for r in results if r["status"] == "complete"]
    if not good:
        raise ValueError("no usable recordings")
    data = pd.concat(good, ignore_index=True)
    metrics = ["endpoint_separation_cm", "regional_tv", "pair_entropy", "a_entropy", "b_entropy",
               "pair_truth_error_cm", "a_truth_error_cm", "b_truth_error_cm", "map_region_agreement",
               "a_width_cm", "b_width_cm", "a_spikes", "b_spikes", "a_active", "b_active"]
    cases = ["dataset", "source", "split", "condition"]
    session = data.groupby(cases+["animal", "session", "draw"])[metrics].mean().groupby(cases+["animal", "session"]).mean().reset_index()
    animal = session.groupby(cases+["animal"])[metrics].mean().reset_index()
    total = animal.groupby(cases)[metrics].mean().reset_index()
    for name, table in (("by_session", session), ("by_animal", animal), ("summary", total)):
        table.to_csv(out/f"{name}.csv", index=False)
    external = total.loc[total.dataset.eq("blackstad_moser") & total.split.eq(0)]
    gates = []
    def gate(name, passed, value, criterion):
        gates.append(dict(gate=name, passed=bool(passed), value=value, criterion=criterion))
    gate("inputs_unchanged", inputs_unchanged, inputs_unchanged, "all frozen inputs retain hashes")
    records = [r for r in results if r["dataset"] == "blackstad_moser"]
    retained = [r for r in records if r["status"] == "complete"]
    n_animals = len({r["animal"] for r in retained})
    denominator = sum(r["source_candidates"] for r in records)
    coverage = sum(r["candidates"] for r in retained)/denominator if denominator else 0
    gate("external_coverage", n_animals >= 4 and coverage >= .8, f"{n_animals} animals; {coverage}", ">=4 animals and >=80% source endpoints")
    if external.empty:
        gate("external_results_present", False, 0, "nonempty")
    else:
        case = external.loc[external.source.eq("real")].set_index("condition")
        for metric in ("regional_tv", "endpoint_separation_cm"):
            difference = case.loc["random", metric]-case.loc["balanced", metric]
            reduction = difference/case.loc["random", metric] if case.loc["random", metric] > 0 else float("nan")
            gate(f"{metric}_reduction", reduction >= .1, reduction, ">=10%")
            per_animal = animal.loc[animal.dataset.eq("blackstad_moser") & animal.source.eq("real") & animal.split.eq(0)].pivot(index="animal", columns="condition", values=metric)
            delta = (per_animal.random-per_animal.balanced).to_numpy()
            gate(f"{metric}_animals_positive", (delta > 0).sum() >= 4, int((delta > 0).sum()), ">=4")
            draws = np.random.default_rng(SEED).choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            gate(f"{metric}_bootstrap", low > 0, f"[{low}, {high}]", "descriptive animal interval lower >0")
        for metric, slack in (("pair_entropy", 0), ("a_entropy", .01), ("b_entropy", .01)):
            delta = case.loc["balanced", metric]-case.loc["random", metric]
            gate(f"{metric}_not_worse", delta <= slack, delta, f"balanced minus random <={slack}")
        for source in ("run_test", "sim_matched", "sim_drift"):
            truth = external.loc[external.source.eq(source)].set_index("condition")
            for metric, slack in (("pair_truth_error_cm", 0), ("a_truth_error_cm", 2), ("b_truth_error_cm", 2)):
                delta = truth.loc["balanced", metric]-truth.loc["random", metric]
                gate(f"{source}_{metric}", delta <= slack, delta, f"balanced minus random <={slack}")
    gate("statistical_validation", all(r["passed"] for r in gates), "", "all outcome gates; separate technical audit required")
    pd.DataFrame(gates).to_csv(out/"gates.csv", index=False)
    report = ["# Spatially balanced sampling: external validation", "",
              f"Prespecified outcome gates: **{'PASS' if gates[-1]['passed'] else 'FAIL'}**.",
              "Technical reconstruction audit remains separately required.",
              "All events retained; same total cells and observations; random vs RUN-balanced disjoint halves.",
              "Independent external dataset: Blackstad--Moser. PF is development only.", ""]
    report += [f"- {r['gate']}: {'pass' if r['passed'] else 'FAIL'}; {r['value']}" for r in gates]
    report += ["", "No claim of true replay content, goal preference or unbiased biological replay prevalence.",
               "The previous PF-to-Tanni agreement-selection failure is retained; this is a new frozen sampling remedy."]
    (out/"report.md").write_text("\n".join(report)+"\n")
    print("\n".join(report), flush=True)
    return pd.DataFrame(gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-input-dir", type=Path, required=True)
    parser.add_argument("--external-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = dict(pf=args.pf_input_dir/"coverage_input_sessions.csv", external=args.external_input_dir/"coverage_input_sessions.csv",
                  code=Path(__file__), protocol=ROOT/"docs/spatially_balanced_content_remedy_protocol.md",
                  partition_helper=ROOT/"scripts/analyze_pf_independent_population_content.py",
                  decoding_helper=ROOT/"scripts/measure_population_content_stability.py")
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance.update(created_at_utc=datetime.now(UTC).isoformat(), status="running", seed=SEED,
                      settings=dict(n_splits=3, primary_split=0, simulated_draws=2, n_partitions=128,
                                    n_run_evaluated=8, time_bin_s=.02, drift_log_gain_sd=.4))
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    results = []
    for dataset, input_dir in (("pfeiffer_foster", args.pf_input_dir), ("blackstad_moser", args.external_input_dir)):
        catalog = pd.read_csv(input_dir/"coverage_input_sessions.csv", dtype={"animal": str, "session": str})
        catalog = catalog.loc[catalog.dataset.eq(dataset)]
        if catalog.empty or catalog.duplicated(["animal", "session"]).any():
            raise ValueError("empty/duplicate dataset catalog")
        for row in catalog.itertuples(index=False):
            output = args.output_dir/f"{dataset}__{row.animal}__{row.session.replace('/', '_')}"
            try:
                result = measure(row, output, SEED)
            except (ValueError, OSError, KeyError) as exc:
                result = dict(dataset=dataset, animal=row.animal, session=row.session, status="failed",
                              reason=str(exc), source_candidates=int(row.candidates))
            results.append(result)
            print(json.dumps(result), flush=True)
            pd.DataFrame(results).to_csv(args.output_dir/"sessions.csv", index=False)
        if dataset == "pfeiffer_foster":
            (args.output_dir/"development_finished_before_external.json").write_text(json.dumps(dict(
                created_at_utc=datetime.now(UTC).isoformat(), frozen_code=file_sha256(__file__),
                frozen_protocol=file_sha256(inputs["protocol"])), indent=2)+"\n")
    unchanged = all(file_sha256(path) == provenance["input_file_sha256"][name] for name, path in inputs.items())
    gates = summarize(results, args.output_dir, unchanged)
    provenance.update(status="complete" if unchanged else "failed", results=results,
                      completed_at_utc=datetime.now(UTC).isoformat(), inputs_unchanged=unchanged,
                      outcome_validation_passed=bool(gates.iloc[-1].passed))
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("inputs changed during evaluation")


if __name__ == "__main__":
    main()
