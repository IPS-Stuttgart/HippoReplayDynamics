#!/usr/bin/env python3
"""Reconstruct frozen simulations and independently recount decoder support.

No decoding is performed. Reusing the generator checks artifact reproducibility,
not the biological realism of that generator or an independent ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.simulate_replay_coverage_recovery import build_trials


def support_from_direct_counts(base):
    windows = np.array([base[t:t + 4].sum(axis=0) for t in range(len(base) - 3)])
    if not len(windows):
        return {"windows": 0, "support_fraction": np.nan, "unfiltered_steps": 0, "filtered_steps": 0}
    support = (windows.sum(axis=1) >= 3) & ((windows > 0).sum(axis=1) >= 2)
    nonoverlap = support[::4]
    return {"windows": len(windows), "support_fraction": float(support.mean()),
            "unfiltered_steps": max(0, len(nonoverlap) - 1),
            "filtered_steps": int(np.count_nonzero(nonoverlap[:-1] & nonoverlap[1:]))}


def compare_tables(actual, expected, keys, columns):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise AssertionError("duplicate audit keys")
    joined = actual[keys + columns].merge(expected[keys + columns], on=keys, how="outer", indicator=True, suffixes=("_a", "_e"), validate="one_to_one")
    if not joined["_merge"].eq("both").all():
        raise AssertionError("missing or unexpected audit keys")
    for col in columns:
        a, e = joined[f"{col}_a"], joined[f"{col}_e"]
        if pd.api.types.is_numeric_dtype(e):
            if not np.allclose(a.to_numpy(float), e.to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True):
                raise AssertionError(f"numeric audit mismatch: {col}")
        elif not (a.eq(e) | (a.isna() & e.isna())).all():
            raise AssertionError(f"audit mismatch: {col}")


def run(args):
    out = args.result_dir.resolve()
    manifest_path = out / "coverage_recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parameters = SimpleNamespace(**manifest["parameters"])
    source = Path(parameters.input_dir)
    current_code = {"script": ROOT / "scripts/simulate_replay_coverage_recovery.py",
                    "recovery": ROOT / "src/hipporeplayimm/replay_coverage_recovery.py",
                    "data": ROOT / "src/hipporeplayimm/replay_coverage_data.py",
                    "encoding": ROOT / "src/hipporeplayimm/encoding.py",
                    "subsampling": ROOT / "scripts/analyze_replay_coverage_subsampling.py"}
    for key, path in current_code.items():
        if file_sha256(path) != manifest["input_sha256_at_start"][key]:
            raise ValueError(f"generator dependency changed: {path}")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(out / name) != digest:
            raise ValueError(f"output hash mismatch: {name}")
    for path, digest in manifest["cache_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("source cache changed")
    sessions = pd.read_csv(source / "coverage_input_sessions.csv")
    sessions = sessions[sessions.artifact_path.isin(manifest["cache_sha256"])]
    if len(sessions) != manifest["sessions"]:
        raise ValueError("session count mismatch")
    rows = []
    keys = ["dataset", "animal", "session", "source_event_index", "truth_kind", "gradient"]
    for record in sessions.itertuples(index=False):
        with np.load(record.artifact_path, allow_pickle=False) as cache:
            trials, specs, observations, subsets, _, _, _, _ = build_trials(cache, record, parameters)
        stem = f"{record.dataset}__{record.animal}__{record.session.split('/')[-1]}"
        saved_specs = pd.read_csv(out / f"trials__{stem}.csv")
        expected_specs = pd.DataFrame(specs)
        compare_tables(saved_specs, expected_specs, keys,
                       ["source_duration_s", "simulated_duration_s", "source_spikes", "source_full_bin_spikes",
                        "source_totals_sha256", "n_base_bins", "path_seed", "observation_seed", "status",
                        "path_sha256", "truth_speed_sha256", "truth_window_means_sha256", "truth_supported_fraction", "truth_geometric_eligible"])
        saved_obs = pd.read_csv(out / f"observations__{stem}.csv")
        compare_tables(saved_obs, pd.DataFrame(observations), keys + ["regime", "cell_fraction"],
                       ["counts_sha256", "spikes", "active_units", "fixed_totals_verified", "native_subset_verified"])
        support_rows = []
        for trial in trials:
            if trial["observations"] is None:
                raise ValueError("this audit requires an explicit short-trial extension before claiming complete coverage")
            for (regime, fraction), counts in trial["observations"].items():
                support = support_from_direct_counts(counts)
                for likelihood in ["poisson", "conditional_multinomial"]:
                    for estimator in ["map", "posterior_mean"]:
                        for filtered in [False, True]:
                            support_rows.append({**{k: trial["spec"][k] for k in keys}, "regime": regime,
                                                 "cell_fraction": fraction, "likelihood": likelihood, "estimator": estimator,
                                                 "bin_filter": "at_least_2cells_3spikes" if filtered else "unfiltered",
                                                 "retained_cells": len(subsets[fraction]), "overlapping_frames": support["windows"],
                                                 "supported_frame_fraction": support["support_fraction"],
                                                 "all_steps": support["filtered_steps" if filtered else "unfiltered_steps"],
                                                 "nonoverlapping_valid_steps": support["filtered_steps" if filtered else "unfiltered_steps"]})
        saved_events = pd.read_csv(out / f"events__{stem}.csv")
        compare_tables(saved_events, pd.DataFrame(support_rows), keys + ["regime", "cell_fraction", "likelihood", "estimator", "bin_filter"],
                       ["retained_cells", "overlapping_frames", "supported_frame_fraction", "all_steps", "nonoverlapping_valid_steps"])
        rows.append({"dataset": record.dataset, "animal": record.animal, "session": record.session,
                     "truth_draws_verified": len(specs), "observation_arrays_verified": len(observations),
                     "metric_rows_support_recounted": len(saved_events), "passed": True})
        print(f"Verified reconstruction and direct support counts: {stem}", flush=True)
    table = pd.DataFrame(rows)
    if table.truth_draws_verified.sum() != manifest["truth_trials"] or table.metric_rows_support_recounted.sum() != manifest["metric_rows"]:
        raise AssertionError("aggregate audit denominator mismatch")
    table.to_csv(out / "coverage_recovery_reconstruction_audit.csv", index=False)
    audit = {**build_script_provenance(input_paths={"manifest": manifest_path, "auditor": Path(__file__)}, cwd=ROOT),
             "all_output_hashes_verified": True, "all_cache_hashes_verified": True,
             "truth_trials": int(table.truth_draws_verified.sum()),
             "observation_arrays": int(table.observation_arrays_verified.sum()),
             "metric_rows_support_recounted": int(table.metric_rows_support_recounted.sum()),
             "scope": "generator reconstruction plus independent direct-window support recount; no rescore, no biological validation"}
    (out / "coverage_recovery_reconstruction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
