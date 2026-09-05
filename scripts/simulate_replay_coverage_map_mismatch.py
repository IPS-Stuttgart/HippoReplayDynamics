#!/usr/bin/env python3
"""Known-path recovery with independent RUN-half maps and shared-gain stress."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import _positions_to_flat_bins
from hipporeplayimm.replay_coverage_counterfactual import shuffle_base_path
from hipporeplayimm.replay_coverage_data import CoverageInputConfig, array_sha256
from hipporeplayimm.replay_coverage_geometry import TRUTHS, resolution_truth, window_counts
from hipporeplayimm.replay_coverage_map_mismatch import decode_with_coverage, fit_map_pair, mismatch_metrics, mismatch_observations
from hipporeplayimm.replay_coverage_recovery import gradient_from_moments, map_interpolator, simulate_path, true_state_indices
from scripts._provenance import build_script_provenance, file_sha256
from scripts.simulate_replay_coverage_recovery import seed_parts
from scripts.validate_replay_coverage_run_decoder import session_from_cache

IDENTITY = ["dataset", "animal", "session", "direction"]
CONDITION = ["observation", "cell_fraction", "decoder_map", "likelihood", "estimator", "bin_filter"]
TRIAL = IDENTITY + ["source_event_index", "truth_kind", "gradient"]
SCORE_KEY = TRIAL + CONDITION


def build_trials(model, profiles, cache, identity, seed, rate_scale, include_fine=False):
    evaluate = map_interpolator(model["generator_rates_hz"], model["generator_x_edges_cm"], model["generator_y_edges_cm"])
    key = ":".join(str(identity[k]) for k in IDENTITY)
    n_cells = len(model["cell_ids"])
    subset = np.sort(np.random.default_rng(seed_parts(seed, key, stream=2)).permutation(n_cells)[:max(1, n_cells // 2)])
    subsets = {1.: np.arange(n_cells), .5: subset}
    source_cells = np.array([np.flatnonzero(cache["cell_ids"] == cell)[0] for cell in model["cell_ids"]])
    source_lookup = {int(value): i for i, value in enumerate(cache["candidate_event_indices"])}
    trials, specs, observations = [], [], []
    for source in profiles.itertuples(index=False):
        event, n_base = int(source.source_event_index), int(source.n_base_bins)
        index = source_lookup[event]
        lo, hi = cache["candidate_offsets"][index:index+2]
        durations = cache["candidate_base_durations_s"][lo:hi]
        if np.count_nonzero(np.isclose(durations, .005, rtol=0, atol=1e-9)) != n_base or n_base < 4:
            raise ValueError("source duration profile mismatch")
        source_spikes = int(cache["candidate_base_counts"][lo:lo+n_base, :][:, source_cells].sum())
        parent, parent_counts, parent_gain = None, None, None
        permutation = np.random.default_rng(seed_parts(seed, key, event, 3)).permutation(n_base)
        for ordinal, (kind, gradient) in enumerate(TRUTHS):
            if kind == "shuffled_continuous":
                path = shuffle_base_path(parent, permutation)
                counts = {name: value.reshape(n_base, 5, n_cells)[permutation].reshape(-1, n_cells) for name, value in parent_counts.items()}
                gain = parent_gain.reshape(n_base, 5)[permutation].reshape(-1)
                for name, values in counts.items():
                    if not np.array_equal(values.sum(axis=0), parent_counts[name].sum(axis=0)):
                        raise AssertionError("shuffle changed per-cell totals")
            else:
                path = simulate_path(n_base, model["domain_cm"], kind, gradient, 1000., seed_parts(seed, key, event, 4))
                counts, gain = mismatch_observations(evaluate(path["midpoints_cm"]), seed_parts(seed, key, event, 10+ordinal), rate_scale)
                if kind == "continuous" and gradient == 0:
                    parent, parent_counts, parent_gain = path, counts, gain
            truth = resolution_truth(path, 20, 5)
            states = true_state_indices(truth["center_cm"], model["x_edges_cm"], model["y_edges_cm"], model["valid_spatial_bins"])
            b_states = _positions_to_flat_bins(truth["center_cm"], model["generator_x_edges_cm"], model["generator_y_edges_cm"])
            supported_b = np.zeros(len(b_states), bool)
            ok = b_states >= 0
            supported_b[ok] = model["generator_valid_spatial_bins"][b_states[ok]]
            spec = {**identity, "source_event_index": event, "truth_kind": kind, "gradient": gradient,
                "n_base_bins": n_base, "duration_s": n_base * .005, "source_duration_s": float(durations.sum()),
                "source_spikes_training_units": source_spikes, "source_event_spikes_all_sorted": int(cache["candidate_base_counts"][lo:hi].sum()),
                "path_sha256": array_sha256(path["midpoints_cm"]), "truth_speed_sha256": array_sha256(path["speed_cm_s"]),
                "truth_training_supported_fraction": float(np.mean(states >= 0)), "truth_generator_supported_fraction": float(supported_b.mean()),
                "gain_sha256": array_sha256(gain), "mean_gain": float(gain.mean()),
                "shuffle_permutation_sha256": array_sha256(permutation) if kind == "shuffled_continuous" else "not_shuffled"}
            selected = {}
            for observation, full in counts.items():
                for fraction, cells in subsets.items():
                    fine = full[:, cells]
                    selected[observation, fraction] = window_counts(fine, 20, 5)
                    observations.append({**{name: spec[name] for name in TRIAL}, "observation": observation,
                        "cell_fraction": fraction, "fine_counts_sha256": array_sha256(fine),
                        "window_counts_sha256": array_sha256(selected[observation, fraction]),
                        "spikes": int(fine.sum()), "active_units": int(np.count_nonzero(fine.sum(axis=0))),
                        "source_spikes_retained_units": int(cache["candidate_base_counts"][lo:lo+n_base, :][:, source_cells[cells]].sum()),
                        "n_cells": len(cells), "same_observations_for_both_maps": True})
            trials.append({"spec": spec, "path": path, "truth": truth, "states": states, "counts": selected})
            if include_fine:
                trials[-1]["fine_counts"] = counts
            specs.append(spec)
    return trials, pd.DataFrame(specs), pd.DataFrame(observations), subsets


def score_trials(trials, model, subsets, rate_scale):
    rows = []
    for observation in ["poisson", "shared_gain"]:
        for fraction, cells in subsets.items():
            chunks = [trial["counts"][observation, fraction] for trial in trials]
            offsets = np.r_[0, np.cumsum([len(chunk) for chunk in chunks])]
            counts = np.concatenate(chunks)
            states = np.concatenate([trial["states"] for trial in trials])
            for map_name, rates in [("generator_known", model["generator_at_training_states_hz"]), ("independent_RUN_half", model["rates_hz"])]:
                for likelihood in ["poisson", "conditional_multinomial"]:
                    decoded = decode_with_coverage(counts, rates[cells] * rate_scale, model["grid_cm"], states, likelihood)
                    for trial, values, a, b in zip(trials, chunks, offsets[:-1], offsets[1:], strict=True):
                        for estimator in ["map", "posterior_mean"]:
                            for support in [False, True]:
                                rows.append({**trial["spec"], "observation": observation, "cell_fraction": fraction,
                                    "decoder_map": map_name, "likelihood": likelihood, "estimator": estimator,
                                    "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered", "n_cells": len(cells),
                                    "status": "scored", **mismatch_metrics(decoded[estimator][a:b], values, trial["truth"],
                                        model["domain_cm"], support, decoded["posterior_rms_cm"][a:b],
                                        decoded["posterior_entropy_nats"][a:b], decoded["hpd95"][a:b])})
    frame = pd.DataFrame(rows)
    if len(frame) != len(trials) * 64 or frame.duplicated(SCORE_KEY).any():
        raise AssertionError("incomplete or duplicate scoring conditions")
    return frame


def summarize_batch(events):
    summaries, gradients = [], []
    for key, group in events.groupby(IDENTITY + CONDITION + ["truth_kind", "gradient"], sort=True):
        row = dict(zip(IDENTITY + CONDITION + ["truth_kind", "gradient"], key, strict=True))
        eligible = group[group.truth_geometric_eligible]
        row.update(events=len(group), acceptance_fraction=float(group.continuity_pass.mean()),
                   eligible_truth_events=len(eligible), eligible_recovery_fraction=float(eligible.continuity_pass.mean()) if len(eligible) else np.nan)
        for metric in ["valid_bins", "all_steps", "selected_steps", "large_jump_fraction", "median_position_error_cm", "hpd95_coverage",
                       "all_median_speed_cm_s", "selected_median_speed_cm_s", "median_posterior_rms_cm", "truth_training_supported_fraction", "truth_generator_supported_fraction"]:
            row[metric] = float(group[metric].mean())
        summaries.append(row)
        if row["truth_kind"] == "continuous":
            for selection in ["all", "selected"]:
                for coordinate in ["true_coordinate", "decoded_coordinate"]:
                    for readout in ["decoded", "true_arclength", "true_chord"]:
                        prefix = f"{selection}__{coordinate}__{readout}"
                        gradients.append({**{name: row[name] for name in IDENTITY + CONDITION + ["gradient"]},
                            "selection": selection, "coordinate": coordinate, "readout": readout,
                            **gradient_from_moments(group, prefix, 1000.)})
    return pd.DataFrame(summaries), pd.DataFrame(gradients)


def technical_gates(batches, sessions, unchanged):
    planned = len(sessions) * 2
    completed = batches[batches.status.eq("scored")] if len(batches) else pd.DataFrame()
    valid_status = bool(len(batches) and batches.status.isin(["scored", "encoding_unavailable"]).all())
    row_counts = bool(len(completed) and completed.metric_rows.eq(completed.source_profiles * 6 * 64).all())
    gates = [("planned_directions_reported", planned > 0 and len(batches) == planned and not batches.duplicated(IDENTITY).any()),
             ("no_unexpected_scoring_failures", valid_status), ("required_model_rows_complete_for_available_maps", row_counts),
             ("inputs_unchanged", unchanged)]
    gates.append(("overall_technical", all(bool(value) for _, value in gates)))
    gates.append(("all_planned_encoding_directions_available", planned > 0 and len(completed) == planned))
    return pd.DataFrame([{"gate": name, "passed": bool(value)} for name, value in gates])


def run(args):
    start = time.monotonic()
    if args.max_profiles is not None and args.max_profiles < 1:
        raise ValueError("positive smoke profile cap required")
    if not np.isfinite(args.rate_scale) or args.rate_scale <= 0:
        raise ValueError("positive finite rate multiplier required")
    cache_root, reference, out = args.input_dir.resolve(), args.reference_dir.resolve(), args.output_dir.resolve()
    inputs = {"cache_manifest": cache_root / "coverage_input_manifest.json", "sessions": cache_root / "coverage_input_sessions.csv",
              "source_units": cache_root / "coverage_input_units.csv", "profiles": reference / "coverage_recovery_trials.csv",
              "protocol": ROOT / "docs/replay_coverage_map_mismatch_protocol.md", "script": Path(__file__)}
    for name in ["replay_coverage", "replay_coverage_map_mismatch", "replay_coverage_data", "replay_coverage_validation",
                 "replay_coverage_geometry", "replay_coverage_recovery", "replay_coverage_counterfactual", "encoding"]:
        inputs[name] = ROOT / "src/hipporeplayimm" / f"{name}.py"
    inputs.update({name: ROOT / "scripts" / f"{name}.py" for name in ["_provenance", "validate_replay_coverage_run_decoder", "simulate_replay_coverage_recovery"]})
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if provenance["git_dirty"] and not args.allow_dirty:
        raise ValueError("commit the scorer before production (or explicit smoke --allow-dirty)")
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    (out / "inputs").mkdir()
    for name, path in inputs.items():
        shutil.copy2(path, out / "inputs" / f"{name}{path.suffix}")
    config = CoverageInputConfig(**json.loads(inputs["cache_manifest"].read_text())["settings"])
    sessions = pd.read_csv(inputs["sessions"])
    if args.sessions:
        sessions = sessions[sessions.apply(lambda row: f"{row.dataset}:{row.animal}:{row.session}" in args.sessions, axis=1)]
        if len(sessions) != len(set(args.sessions)):
            raise ValueError("requested session mismatch")
    if sessions.empty or not sessions.status.eq("cached").all():
        raise ValueError("complete cache sessions required")
    source_units = pd.read_csv(inputs["source_units"], dtype={"source_cell_type_allowed": "boolean"})
    if source_units.source_cell_type_allowed.isna().any():
        raise ValueError("missing source cell-type eligibility")
    profiles = pd.read_csv(inputs["profiles"], usecols=["dataset", "animal", "session", "source_event_index", "n_base_bins"])
    profiles = profiles.drop_duplicates()
    if profiles.duplicated(["dataset", "animal", "session", "source_event_index"]).any():
        raise ValueError("conflicting frozen duration profiles")
    batches, summaries, gradients, specs, observations, unit_rows, cache_hashes = [], [], [], [], [], [], {}
    for record in sessions.itertuples(index=False):
        if file_sha256(record.artifact_path) != record.artifact_sha256:
            raise ValueError("cache digest mismatch")
        cache_hashes[record.artifact_path] = record.artifact_sha256
        def choose(frame, record=record):
            return frame.dataset.eq(record.dataset) & frame.animal.eq(record.animal) & frame.session.eq(record.session)
        full = session_from_cache(record, source_units[choose(source_units)])
        with np.load(record.artifact_path, allow_pickle=False) as handle:
            cache = {name: handle[name] for name in ["cell_ids", "candidate_event_indices", "candidate_offsets", "candidate_base_durations_s", "candidate_base_counts"]}
        local = profiles[choose(profiles)].sort_values("source_event_index")
        if args.max_profiles:
            local = local.head(args.max_profiles)
        if local.empty:
            raise ValueError("no frozen profiles for selected session")
        for direction in [0, 1]:
            identity = {"dataset": record.dataset, "animal": record.animal, "session": record.session, "direction": direction}
            label = f"{record.dataset}__{record.animal}__{full.name}__dir{direction}"
            batch_start = time.monotonic()
            row = {**identity, "source_profiles": len(local), "status": "failed", "failure_reason": ""}
            print(f"fit/score {label}", flush=True)
            try:
                model, qc, meta = fit_map_pair(full, direction, config)
            except ValueError as exc:
                expected = str(exc).startswith(("insufficient training-only encoding support", "insufficient common synthetic spatial extent",
                    "insufficient occupied RUN bins", "insufficient training position samples", "no training spikes", "no contiguous tracking-supported RUN intervals"))
                row.update(status="encoding_unavailable" if expected else "failed", failure_reason=f"{type(exc).__name__}: {exc}", metric_rows=0)
                batches.append(row)
                pd.DataFrame(batches).to_csv(out / "coverage_map_mismatch_batches.csv", index=False)
                print(row["failure_reason"], flush=True)
                continue
            try:
                model_path = out / "models" / f"{label}.npz"
                np.savez_compressed(model_path, **model)
                trials, spec, obs, subsets = build_trials(model, local, cache, identity, args.seed, args.rate_scale)
                (model_path.with_suffix(".json")).write_text(json.dumps({**identity, **meta,
                    "model_sha256": file_sha256(model_path), "half_cell_indices": subsets[.5].tolist()}, indent=2, allow_nan=False) + "\n")
                for name, value in identity.items():
                    qc[name] = value
                unit_rows.append(qc)
                events = score_trials(trials, model, subsets, args.rate_scale)
                filename = f"events__{label}.csv.gz"
                events.to_csv(out / filename, index=False, compression={"method": "gzip", "mtime": 0})
                summary, gradient = summarize_batch(events)
                summaries.append(summary)
                gradients.append(gradient)
                specs.append(spec)
                observations.append(obs)
                row.update(status="scored", file=filename, sha256=file_sha256(out / filename), metric_rows=len(events),
                           n_training_cells=len(model["cell_ids"]), n_states=len(model["grid_cm"]),
                           model_path=str(model_path), model_sha256=file_sha256(model_path),
                           missing_generator_cells=len(meta["generator_cell_ids_missing"]),
                           median_half_map_correlation=meta["median_half_map_correlation"], shared_support_fraction=meta["shared_support_fraction"])
            except (ValueError, RuntimeError, AssertionError, OSError) as exc:
                row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}", metric_rows=0)
                print(row["failure_reason"], flush=True)
            row["runtime_s"] = time.monotonic() - batch_start
            batches.append(row)
            pd.DataFrame(batches).to_csv(out / "coverage_map_mismatch_batches.csv", index=False)
            print(f"{label}: {row['status']} {row['metric_rows']} rows", flush=True)
    for name, frames in [("session_summary", summaries), ("session_gradients", gradients), ("trials", specs), ("observations", observations), ("training_units", unit_rows)]:
        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_csv(out / f"coverage_map_mismatch_{name}.csv", index=False)
    unchanged = all(file_sha256(path) == hashes[name] for name, path in inputs.items()) and all(file_sha256(path) == digest for path, digest in cache_hashes.items())
    gates = technical_gates(pd.DataFrame(batches), sessions, unchanged)
    gates.to_csv(out / "coverage_map_mismatch_gate_summary.csv", index=False)
    manifest = {**provenance, "created_at_utc": datetime.now(UTC).isoformat(), "parameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "encoding_settings": asdict(config), "cache_sha256": cache_hashes, "input_sha256_at_start": hashes,
        "planned_sessions": len(sessions), "planned_directions": len(sessions)*2,
        "metric_rows": int(pd.DataFrame(batches).metric_rows.sum()), "runtime_s": time.monotonic()-start,
        "status": "complete" if gates.loc[gates.gate.eq("overall_technical"), "passed"].item() else "failed",
        "claim_boundary": "known synthetic paths; independent RUN-half maps are surrogate truth, not actual replay or calibrated biological speed",
        "output_sha256": {str(path.relative_to(out)): file_sha256(path) for path in out.rglob("*") if path.is_file() and "inputs" not in path.relative_to(out).parts}}
    (out / "coverage_map_mismatch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["status"] != "complete":
        raise RuntimeError("map mismatch technical checks failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--max-profiles", type=int, help="deterministic prefix for excluded technical smoke only")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--rate-scale", type=float, default=3.)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
