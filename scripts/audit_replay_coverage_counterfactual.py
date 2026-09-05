#!/usr/bin/env python3
"""Reconstruct paired counts and independently recount support; no rescoring."""

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
from scripts.audit_replay_coverage_recovery import compare_tables, support_from_direct_counts
from scripts.simulate_replay_coverage_counterfactual import (
    SCORE_KEY,
    TRIAL_KEY,
    build_counterfactual_trials,
)


def direct_support_rows(trials, subsets, scale):
    """Exclude aggregate spikes from the sorted-unit support gate explicitly."""
    rows = []
    for trial in trials:
        for (dose, regime, fraction), base in trial["counts"].items():
            if dose != scale:
                continue
            full = trial["counts"][dose, "native", 1.]
            kept = full[:, subsets[fraction]]
            if regime == "pooled_removed":
                if not np.array_equal(base[:, :-1], kept) or not np.array_equal(base[:, -1], full.sum(axis=1) - kept.sum(axis=1)):
                    raise AssertionError("pooled observation is not the deterministic parent sum")
                support_base = base[:, :-1]
            else:
                support_base = base
            support = support_from_direct_counts(support_base)
            for likelihood in ["poisson", "conditional_multinomial"]:
                for estimator in ["map", "posterior_mean"]:
                    for filtered in [False, True]:
                        rows.append({**{k: trial["spec"][k] for k in TRIAL_KEY},
                                     "rate_scale": dose, "regime": regime, "cell_fraction": fraction,
                                     "likelihood": likelihood, "estimator": estimator,
                                     "bin_filter": "at_least_2cells_3spikes" if filtered else "unfiltered",
                                     "retained_cells": len(subsets[fraction]), "observation_channels": base.shape[1],
                                     "overlapping_frames": support["windows"],
                                     "supported_frame_fraction": support["support_fraction"],
                                     "all_steps": support["filtered_steps" if filtered else "unfiltered_steps"],
                                     "nonoverlapping_valid_steps": support["filtered_steps" if filtered else "unfiltered_steps"]})
    return pd.DataFrame(rows)


def run(args):
    out = args.result_dir.resolve()
    manifest_path = out / "coverage_counterfactual_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parameters = SimpleNamespace(**manifest["parameters"])
    for key, digest in manifest["input_sha256_at_start"].items():
        path = Path(manifest["input_file_paths"][key])
        if file_sha256(path) != digest:
            raise ValueError(f"input or dependency changed: {path}")
        if file_sha256(out / "input_snapshot" / f"{key}__{path.name}") != digest:
            raise ValueError(f"snapshot changed: {path.name}")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(out / name) != digest:
            raise ValueError(f"output hash mismatch: {name}")
    for path, digest in manifest["cache_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"source cache changed: {path}")
    reference = json.loads(Path(manifest["input_file_paths"]["reference_manifest"]).read_text())
    sessions = pd.read_csv(manifest["input_file_paths"]["sessions"])
    sessions = sessions[sessions.artifact_path.isin(manifest["cache_sha256"])]
    profiles = pd.read_csv(manifest["input_file_paths"]["profiles"])
    profiles = profiles[profiles.truth_kind.eq("continuous") & profiles.gradient.eq(0)]
    batches = pd.read_csv(out / "coverage_counterfactual_batches.csv")
    saved_specs = pd.read_csv(out / "coverage_counterfactual_trials.csv")
    saved_obs = pd.read_csv(out / "coverage_counterfactual_observations.csv")
    if len(sessions) != manifest["sessions"] or manifest["cache_sha256"].items() - reference["cache_sha256"].items():
        raise ValueError("reference population changed")
    rows, populations = [], []
    for record in sessions.itertuples(index=False):
        local = profiles[profiles.dataset.eq(record.dataset) & profiles.animal.eq(record.animal) & profiles.session.eq(record.session)]
        if parameters.max_profiles is not None:
            local = local.sort_values("source_event_index").head(parameters.max_profiles)
        for replicate in range(parameters.replicates):
            with np.load(record.artifact_path, allow_pickle=False) as cache:
                trials, specs, obs, subsets, ids, _, _, _ = build_counterfactual_trials(cache, local, record, parameters, replicate)
            expected_specs, expected_obs = pd.DataFrame(specs), pd.DataFrame(obs)
            condition = {"dataset": record.dataset, "animal": record.animal, "session": record.session, "replicate": replicate}
            def select(table, condition=condition):
                return table.loc[np.logical_and.reduce([table[key].eq(value).to_numpy() for key, value in condition.items()])]
            compare_tables(select(saved_specs), expected_specs, TRIAL_KEY, [c for c in expected_specs if c not in TRIAL_KEY])
            obs_key = TRIAL_KEY + ["rate_scale", "regime", "cell_fraction"]
            compare_tables(select(saved_obs), expected_obs, obs_key, [c for c in expected_obs if c not in obs_key])
            for fraction, subset in subsets.items():
                populations.append({**condition, "cell_fraction": fraction, "cell_ids": ids[subset].tolist()})
            metric_rows = 0
            selected_batches = select(batches)
            expected_scales = parameters.rate_scales if replicate < parameters.dose_replicates else [parameters.primary_rate_scale]
            if sorted(selected_batches.rate_scale.tolist()) != sorted(expected_scales):
                raise AssertionError("missing or repeated dose batch")
            for batch in selected_batches.itertuples(index=False):
                saved = pd.read_csv(out / batch.file)
                support = direct_support_rows(trials, subsets, batch.rate_scale)
                compare_tables(saved, support, SCORE_KEY, [c for c in support if c not in SCORE_KEY])
                if len(saved) != batch.metric_rows or not saved.status.eq("scored").all():
                    raise AssertionError("scoring status or denominator mismatch")
                metric_rows += len(saved)
            rows.append({**condition, "truth_trials_verified": len(specs), "observation_arrays_verified": len(obs),
                         "metric_rows_support_recounted": metric_rows, "passed": True})
            print(f"Reconstructed {record.dataset}/{record.session}/rep{replicate}; recounted {metric_rows} rows", flush=True)
    table = pd.DataFrame(rows)
    if populations != manifest["populations"]:
        raise AssertionError("population IDs or ordering changed")
    if table.truth_trials_verified.sum() != manifest["truth_trials_including_shuffle"] or table.observation_arrays_verified.sum() != len(saved_obs) or table.metric_rows_support_recounted.sum() != manifest["metric_rows"]:
        raise AssertionError("aggregate audit denominator mismatch")
    table.to_csv(out / "coverage_counterfactual_reconstruction_audit.csv", index=False)
    audit = {**build_script_provenance(input_paths={"manifest": manifest_path, "auditor": Path(__file__),
                                                  "support_auditor": ROOT / "scripts/audit_replay_coverage_recovery.py"}, cwd=ROOT),
             "all_output_hashes_verified": True, "all_cache_hashes_verified": True,
             "all_dependency_and_snapshot_hashes_verified": True,
             "truth_trials": int(table.truth_trials_verified.sum()),
             "observation_arrays": int(table.observation_arrays_verified.sum()),
             "metric_rows_support_recounted": int(table.metric_rows_support_recounted.sum()),
             "scope": "generator reconstruction plus independent parent-pooling and direct-window support recount; no rescore or biological validation"}
    (out / "coverage_counterfactual_reconstruction_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
