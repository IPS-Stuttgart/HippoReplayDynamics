#!/usr/bin/env python3
"""Reconstruct half maps/counts and independently verify sampled decoding."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.replay_coverage_data import CoverageInputConfig
from hipporeplayimm.replay_coverage_map_mismatch import fit_map_pair
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_replay_coverage_geometry import compare_tables, direct_support, direct_truth_eligibility
from scripts.simulate_replay_coverage_map_mismatch import CONDITION, IDENTITY, SCORE_KEY, TRIAL, build_trials, summarize_batch
from scripts.validate_replay_coverage_run_decoder import session_from_cache


def sampled_decode_check(rows, trial, model, subsets, rate_scale):
    checked = 0
    for key, group in rows.groupby(["observation", "cell_fraction", "decoder_map", "likelihood"], sort=False):
        family, fraction, map_name, likelihood = key
        cells = subsets[fraction]
        _, counts = direct_support(trial["fine_counts"][family][:, cells], 20, 5, False)
        rates = model["rates_hz"] if map_name == "independent_RUN_half" else model["generator_at_training_states_hz"]
        rates = rates[cells] * rate_scale
        logs = counts @ np.log(rates)
        if likelihood == "poisson":
            logs -= .02 * rates.sum(axis=0)
        else:
            logs -= counts.sum(axis=1)[:, None] * np.log(rates.sum(axis=0))[None, :]
        posterior = np.exp(logs - logsumexp(logs, axis=1)[:, None])
        positions = {"map": model["grid_cm"][posterior.argmax(axis=1)], "posterior_mean": posterior @ model["grid_cm"]}
        state = trial["states"]
        coverage = np.zeros(len(state), bool)
        for t in np.flatnonzero(state >= 0):
            coverage[t] = posterior[t, posterior[t] > posterior[t, state[t]]].sum() < .95
        for row in group.itertuples(index=False):
            valid = ((counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)) if row.bin_filter != "unfiltered" else np.ones(len(counts), bool)
            points = positions[row.estimator]
            errors = np.linalg.norm(points - trial["truth"]["center_cm"], axis=1)
            np.testing.assert_allclose(row.median_position_error_cm, np.median(errors[valid]) if valid.any() else np.nan, atol=1e-8, rtol=1e-8, equal_nan=True)
            np.testing.assert_allclose(row.hpd95_coverage, np.mean(coverage[valid]) if valid.any() else np.nan, atol=1e-10, equal_nan=True)
            # Independent earliest-longest continuity scan, including support gaps.
            best_start, best_end, start = 0, 0, None
            for t in range(len(points)):
                if not valid[t]:
                    start = None
                    continue
                if start is None or t == 0 or not valid[t-1] or np.linalg.norm(points[t]-points[t-1]) >= 20:
                    start = t
                if t+1-start > best_end-best_start:
                    best_start, best_end = start, t+1
            passed = best_end-best_start >= 10 and np.linalg.norm(points[best_end-1]-points[best_start]) >= 40
            if bool(row.continuity_pass) != bool(passed):
                raise AssertionError("independent continuity scan differs")
            idx = np.arange(0, len(points), 4)
            good = np.array([valid[a:b+1].all() for a, b in pairwise(idx)], dtype=bool)
            speed = np.linalg.norm(np.diff(points[idx], axis=0), axis=1) / .02
            np.testing.assert_allclose(row.all_median_speed_cm_s, np.median(speed[good]) if good.any() else np.nan, atol=1e-8, rtol=1e-8, equal_nan=True)
            checked += 1
    return checked


def audit(root):
    manifest_path = root / "coverage_map_mismatch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise AssertionError("scorer not terminal complete")
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(root / name) != expected:
            raise AssertionError(f"output hash mismatch: {name}")
    for name, path in manifest["input_file_paths"].items():
        expected = manifest["input_file_sha256"][name]
        if file_sha256(path) != expected or file_sha256(root / "inputs" / f"{name}{Path(path).suffix}") != expected:
            raise AssertionError(f"input code/data changed: {name}")
    for path, expected in manifest["cache_sha256"].items():
        if file_sha256(path) != expected:
            raise AssertionError("raw cache changed")
    sessions = pd.read_csv(manifest["input_file_paths"]["sessions"])
    units = pd.read_csv(manifest["input_file_paths"]["source_units"], dtype={"source_cell_type_allowed": "boolean"})
    config = CoverageInputConfig(**manifest["encoding_settings"])
    batches = pd.read_csv(root / "coverage_map_mismatch_batches.csv")
    observations = pd.read_csv(root / "coverage_map_mismatch_observations.csv")
    specs = pd.read_csv(root / "coverage_map_mismatch_trials.csv")
    summaries = pd.read_csv(root / "coverage_map_mismatch_session_summary.csv")
    gradients = pd.read_csv(root / "coverage_map_mismatch_session_gradients.csv")
    profile_file = manifest["input_file_paths"]["profiles"]
    original_profiles = pd.read_csv(profile_file, usecols=["dataset", "animal", "session", "source_event_index", "n_base_bins"]).drop_duplicates()
    rows = []
    for batch in batches.itertuples(index=False):
        def choose(frame, batch=batch):
            mask = frame.dataset.eq(batch.dataset) & frame.animal.eq(batch.animal) & frame.session.eq(batch.session)
            return mask & frame.direction.eq(batch.direction) if "direction" in frame else mask
        record = next(sessions[choose(sessions)].itertuples(index=False))
        full = session_from_cache(record, units[choose(units)])
        if batch.status == "encoding_unavailable":
            try:
                fit_map_pair(full, batch.direction, config)
            except ValueError as exc:
                if f"{type(exc).__name__}: {exc}" != batch.failure_reason:
                    raise AssertionError("availability failure did not reproduce") from exc
            else:
                raise AssertionError("reported unavailable encoding now fits")
            rows.append({**{key: getattr(batch, key) for key in IDENTITY}, "status": "availability_failure_reproduced", "metric_rows": 0, "sampled_decode_rows": 0, "observation_arrays": 0, "passed": True})
            continue
        reconstructed, _, meta = fit_map_pair(full, batch.direction, config)
        with np.load(batch.model_path, allow_pickle=False) as handle:
            model = dict(handle)
        if set(model) != set(reconstructed):
            raise AssertionError("model array keys changed")
        for name, value in model.items():
            np.testing.assert_array_equal(value, reconstructed[name], err_msg=name)
        meta_record = json.loads(Path(batch.model_path).with_suffix(".json").read_text())
        for name in ["training_spikes_sha256", "training_positions_sha256", "generator_spikes_sha256", "generator_positions_sha256"]:
            if meta[name] != meta_record[name]:
                raise AssertionError("half data hashes changed")
        with np.load(record.artifact_path, allow_pickle=False) as handle:
            cache = {name: handle[name] for name in ["cell_ids", "candidate_event_indices", "candidate_offsets", "candidate_base_durations_s", "candidate_base_counts"]}
        profiles = original_profiles[choose(original_profiles)].sort_values("source_event_index")
        cap = manifest["parameters"]["max_profiles"]
        if cap:
            profiles = profiles.head(cap)
        identity = {key: getattr(batch, key) for key in IDENTITY}
        trials, expected_specs, expected_obs, subsets = build_trials(model, profiles, cache, identity,
            manifest["parameters"]["seed"], manifest["parameters"]["rate_scale"], include_fine=True)
        compare_tables(specs[choose(specs)], expected_specs, TRIAL)
        compare_tables(observations[choose(observations)], expected_obs, TRIAL + ["observation", "cell_fraction"])
        events = pd.read_csv(root / batch.file)
        if events.duplicated(SCORE_KEY).any() or len(events) != len(trials)*64:
            raise AssertionError("metric row keys incomplete")
        measured, sampled = 0, 0
        for trial in trials:
            spec = trial["spec"]
            local = events[events.source_event_index.eq(spec["source_event_index"]) & events.truth_kind.eq(spec["truth_kind"]) & events.gradient.eq(spec["gradient"])]
            if len(local) != 64:
                raise AssertionError("missing trial conditions")
            if not local.truth_geometric_eligible.eq(direct_truth_eligibility(trial["path"], 20, 5, "literal_20cm_10frames")).all():
                raise AssertionError("truth eligibility mismatch")
            for (family, fraction, support), group in local.groupby(["observation", "cell_fraction", "bin_filter"], sort=False):
                expected, counts = direct_support(trial["fine_counts"][family][:, subsets[fraction]], 20, 5, support != "unfiltered")
                np.testing.assert_array_equal(counts, trial["counts"][family, fraction])
                for key in ["valid_bins", "valid_adjacent_steps", "all_steps", "n_decoded_bins"]:
                    if not group[key].eq(expected[key]).all():
                        raise AssertionError(f"direct support differs: {key}")
                measured += len(group)
            if spec["source_event_index"] == profiles.source_event_index.iloc[0]:
                sampled += sampled_decode_check(local, trial, model, subsets, manifest["parameters"]["rate_scale"])
        expected_summary, expected_gradients = summarize_batch(events)
        compare_tables(summaries[choose(summaries)], expected_summary, IDENTITY + CONDITION + ["truth_kind", "gradient"])
        compare_tables(gradients[choose(gradients)], expected_gradients, IDENTITY + CONDITION + ["gradient", "selection", "coordinate", "readout"])
        rows.append({**identity, "status": "reconstructed", "metric_rows": measured, "sampled_decode_rows": sampled, "observation_arrays": len(expected_obs), "passed": True})
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) != manifest["planned_directions"] or sum(row["metric_rows"] for row in rows) != manifest["metric_rows"]:
        raise AssertionError("audit scope incomplete")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.input_dir)
    rows.to_csv(args.input_dir / "coverage_map_mismatch_reconstruction_audit.csv", index=False)
    metadata = {**build_script_provenance(input_paths={"scoring_manifest": args.input_dir / "coverage_map_mismatch_manifest.json", "audit_code": __file__}),
        "status": "pass", "directions": len(rows), "metric_rows_recounted": int(rows.metric_rows.sum()),
        "sampled_decode_rows": int(rows.sampled_decode_rows.sum()), "observation_arrays": int(rows.observation_arrays.sum())}
    (args.input_dir / "coverage_map_mismatch_reconstruction_audit.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
