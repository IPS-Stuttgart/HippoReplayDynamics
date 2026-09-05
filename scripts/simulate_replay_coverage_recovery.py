#!/usr/bin/env python3
"""Known-path recovery in empirical maps; development benchmark, not biology."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage import continuity_metrics
from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_coverage_recovery import (
    BASE_S,
    decode_batches,
    gradient_from_moments,
    map_interpolator,
    paired_speed_moments,
    sample_observations,
    simulate_path,
    true_state_indices,
    truth_windows,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.analyze_replay_coverage_subsampling import (
    decoding_support,
    event_windows,
    observed_metrics,
    population_subsets,
)

TRUTHS = [("continuous", -.5), ("continuous", 0.), ("continuous", .5),
          ("stationary", 0.), ("discontinuous", 0.)]
REGIMES = ["poisson_oracle", "fixed_count", "shared_gain_mismatch"]
STRATA = ["dataset", "truth_kind", "gradient", "regime", "cell_fraction",
          "likelihood", "estimator", "bin_filter"]
KEY = ["animal", "session", "source_event_index"] + STRATA


def seed_parts(seed, session_key, event=0, stream=0):
    token = int.from_bytes(hashlib.sha256(session_key.encode()).digest()[:8], "little")
    return [int(seed), token, int(event), int(stream)]


def select_sources(event_ids, count, seed, key):
    ids = np.asarray(event_ids)
    if count < 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("need distinct candidate IDs and a positive sample size")
    order = np.argsort(ids)
    rng = np.random.default_rng(seed_parts(seed, key, stream=1))
    return np.sort(rng.choice(order, min(count, len(ids)), replace=False))


def simulation_domain(cache):
    bounds = cache["arena_bounds_cm"]
    if np.isfinite(bounds).all():
        return bounds.copy(), "native_arena_bounds"
    if not np.isnan(bounds).all():
        raise ValueError("partially missing arena bounds")
    return np.array([[cache["x_edges_cm"][0], cache["y_edges_cm"][0]],
                     [cache["x_edges_cm"][-1], cache["y_edges_cm"][-1]]]), "encoding_grid_extent_not_verified_walls"


def build_trials(cache, record, args):
    mask = cache["unit_qc_mask"].astype(bool)
    ids = cache["cell_ids"][mask]
    rates = cache["rates_hz"][mask]
    bounds, domain_source = simulation_domain(cache)
    valid = decoding_support(cache["valid_spatial_bins"], cache["bin_centers_cm"], cache["arena_bounds_cm"])
    interpolate = map_interpolator(rates, cache["x_edges_cm"], cache["y_edges_cm"])
    key = f"{record.dataset}:{record.animal}:{record.session}"
    subsets = {f: idx for f, _, idx in population_subsets(ids, [.5, .25], 1, args.seed, key)}
    selected = select_sources(cache["candidate_event_indices"], args.events_per_session, args.seed, key)
    trials, specs, observations = [], [], []
    counts = cache["candidate_base_counts"][:, mask]
    for source in selected:
        start, end = cache["candidate_offsets"][source:source + 2]
        durations = cache["candidate_base_durations_s"][start:end]
        # Validate the source time grid using the same code as the real audit.
        event_windows(counts[start:end], durations)
        n_base = int(np.isclose(durations, BASE_S, rtol=0, atol=1e-9).sum())
        totals = counts[start:start + n_base].sum(axis=1)
        event = int(cache["candidate_event_indices"][source])
        for kind, gradient in TRUTHS:
            identity = {"dataset": record.dataset, "animal": record.animal, "session": record.session,
                        "source_event_index": event, "truth_kind": kind, "gradient": gradient}
            spec = {**identity, "source_duration_s": float(durations.sum()), "simulated_duration_s": n_base * BASE_S,
                    "source_spikes": int(counts[start:end].sum()), "source_full_bin_spikes": int(totals.sum()),
                    "source_totals_sha256": array_sha256(totals), "n_base_bins": n_base,
                    "domain_source": domain_source, "bounds_cm": json.dumps(bounds.tolist()),
                    "path_seed": json.dumps(seed_parts(args.seed, key, event, 2)),
                    "observation_seed": json.dumps(seed_parts(args.seed, key, event, 3)),
                    "truth_supported_fraction": np.nan, "truth_geometric_eligible": False,
                    "status": "insufficient_source_duration" if n_base < 4 else "generated"}
            if n_base < 4:
                trials.append({"spec": spec, "observations": None})
                specs.append(spec)
                continue
            path = simulate_path(n_base, bounds, kind, gradient, args.speed_cm_s,
                                 seed_parts(args.seed, key, event, 2), fine_s=args.fine_s)
            truth = truth_windows(path)
            truth_indices = true_state_indices(truth["center_cm"], cache["x_edges_cm"], cache["y_edges_cm"], valid)
            obs, gains = sample_observations(interpolate(path["midpoints_cm"]), path, totals,
                                            subsets, args.rate_scale, seed_parts(args.seed, key, event, 3))
            spec.update({**gains, "path_sha256": array_sha256(path["midpoints_cm"]),
                         "truth_speed_sha256": array_sha256(path["speed_cm_s"]),
                         "truth_window_means_sha256": array_sha256(truth["window_mean_cm"]),
                         "truth_supported_fraction": float((truth_indices >= 0).mean()),
                         "truth_geometric_eligible": continuity_metrics(truth["window_mean_cm"])["continuity_pass"]})
            for (regime, fraction), values in obs.items():
                if regime == "fixed_count" and not np.array_equal(values.sum(axis=1), totals):
                    raise AssertionError("fixed-count experiment changed source population totals")
                if regime != "fixed_count" and not np.array_equal(values, obs[regime, 1.][:, subsets[fraction]]):
                    raise AssertionError("cell removal changed retained-cell observations")
                observations.append({**identity, "regime": regime, "cell_fraction": fraction,
                                     "counts_sha256": array_sha256(values), "spikes": int(values.sum()),
                                     "active_units": int(np.count_nonzero(values.sum(axis=0))),
                                     "fixed_totals_verified": regime == "fixed_count",
                                     "native_subset_verified": regime != "fixed_count"})
            trials.append({"spec": spec, "observations": obs, "truth": truth, "truth_indices": truth_indices})
            specs.append(spec)
    return trials, specs, observations, subsets, ids, rates[:, valid], cache["bin_centers_cm"][valid], bounds


def score_trials(trials, subsets, rates, grid, bounds, args):
    rows = []
    for regime in REGIMES:
        for fraction, subset in subsets.items():
            windows = [event_windows(trial["observations"][regime, fraction],
                                     np.full(trial["spec"]["n_base_bins"], BASE_S))
                       if trial["observations"] is not None else np.empty((0, len(subset)), dtype=int)
                       for trial in trials]
            offsets = np.r_[0, np.cumsum([len(w) for w in windows])]
            all_counts = np.concatenate(windows)
            indices = np.concatenate([t["truth_indices"] if t["observations"] is not None else np.empty(0, int) for t in trials])
            for likelihood in ["poisson", "conditional_multinomial"]:
                decoded = decode_batches(all_counts, rates[subset] * args.rate_scale, grid, indices, likelihood)
                for trial, values, a, b in zip(trials, windows, offsets[:-1], offsets[1:], strict=True):
                    for estimator in ["map", "posterior_mean"]:
                        for support in [False, True]:
                            row = {**trial["spec"], "regime": regime, "cell_fraction": fraction,
                                   "retained_cells": len(subset), "likelihood": likelihood, "estimator": estimator,
                                   "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered"}
                            if a == b:
                                empty = np.empty((0, 2))
                                truth_empty = {"speed_cm_s": np.empty(0), "true_covariate": np.empty(0),
                                               "window_mean_chord_speed_cm_s": np.empty(0)}
                                rows.append({**row, **observed_metrics(empty, values, np.empty(0), np.empty(0), support),
                                             **paired_speed_moments(empty, values, truth_empty, bounds, support),
                                             "median_position_error_cm": np.nan, "hpd95_coverage": np.nan,
                                             "status": "insufficient_source_duration"})
                                continue
                            keep = (values.sum(axis=1) >= 3) & (np.count_nonzero(values, axis=1) >= 2) if support else np.ones(len(values), bool)
                            path = decoded[estimator][a:b]
                            error = np.linalg.norm(path - trial["truth"]["center_cm"], axis=1)
                            row.update(observed_metrics(path, values, decoded["posterior_rms_cm"][a:b], decoded["posterior_entropy_nats"][a:b], support))
                            row.update(paired_speed_moments(path, values, trial["truth"], bounds, support))
                            row.update({"status": "scored", "median_position_error_cm": float(np.median(error[keep])) if keep.any() else np.nan,
                                        "hpd95_coverage": float(decoded["hpd95"][a:b][keep].mean()) if keep.any() else np.nan,
                                        "hpd95_target": "instantaneous_window_center_not_replay_truth",
                                        "primary_likelihood_for_regime": likelihood == ("conditional_multinomial" if regime == "fixed_count" else "poisson")})
                            rows.append(row)
    return pd.DataFrame(rows)


def summarize_events(events, speed_scale):
    rows, gradients = [], []
    for key, local in events.groupby(STRATA + ["animal", "session"], dropna=False, sort=True):
        base = dict(zip(STRATA + ["animal", "session"], key, strict=True))
        eligible = local.truth_geometric_eligible.fillna(False).astype(bool) if "truth_geometric_eligible" in local else np.zeros(len(local), bool)
        continuous = base["truth_kind"] == "continuous"
        success = local.continuity_pass.fillna(False).astype(bool)
        rows.append({**base, "trials": len(local), "scored_trials": int(local.status.eq("scored").sum()),
                     "truth_eligible_trials": int(eligible.sum()), "geometric_pass_fraction": float(success.mean()),
                     "continuous_eligible_recovery_fraction": float(success[eligible].mean()) if continuous and np.any(eligible) else np.nan,
                     "null_geometric_acceptance_fraction": float(success.mean()) if not continuous else np.nan,
                     "median_speed_cm_s": local.nonoverlapping_event_median_speed_cm_s.median(),
                     "median_selected_speed_cm_s": local.selected_core_median_speed_cm_s.median(),
                     "median_position_error_cm": local.median_position_error_cm.median(),
                     "mean_truth_supported_fraction": local.truth_supported_fraction.mean(),
                     "mean_hpd95_coverage": local.hpd95_coverage.mean()})
        if not continuous:
            continue
        for selection in ["all", "selected"]:
            for axis in ["true_coordinate", "decoded_coordinate"]:
                for readout in ["decoded", "true_arclength", "true_chord"]:
                    prefix = f"{selection}__{axis}__{readout}"
                    result = gradient_from_moments(local, prefix, speed_scale)
                    gradients.append({**base, "selection": selection, "coordinate": axis, "readout": readout, **result})
    return pd.DataFrame(rows), pd.DataFrame(gradients)


def cluster_summary(gradients, seed, bootstraps):
    strata = STRATA + ["selection", "coordinate", "readout"]
    animals = gradients.groupby(strata + ["animal"], as_index=False, dropna=False).agg(
        normalized_slope=("normalized_slope", "mean"), sessions=("session", "size"),
        available_sessions=("status", lambda s: s.eq("available").sum()))
    rows, rng = [], np.random.default_rng(seed)
    for key, local in animals.groupby(strata, dropna=False, sort=True):
        finite = local.normalized_slope.dropna().to_numpy()
        lo, hi = np.nan, np.nan
        if len(finite) >= 2:
            lo, hi = np.quantile(rng.choice(finite, (bootstraps, len(finite)), replace=True).mean(axis=1), [.025, .975])
        rows.append({**dict(zip(strata, key, strict=True)), "animals": len(local), "finite_animals": len(finite),
                     "normalized_slope": float(finite.mean()) if len(finite) else np.nan,
                     "animal_ci95_low": lo, "animal_ci95_high": hi,
                     "uncertainty_scope": "animal_bootstrap_conditional_on_frozen_simulations_not_biological_speed"})
    return animals, pd.DataFrame(rows)


def plot_gradients(summary, output):
    datasets = sorted(summary.dataset.unique())
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 4 * len(datasets)), squeeze=False, constrained_layout=True)
    for row, dataset in enumerate(datasets):
        for col, regime in enumerate(REGIMES):
            ax = axes[row, col]
            local = summary[summary.dataset.eq(dataset) & summary.regime.eq(regime)
                            & summary.estimator.eq("posterior_mean") & summary.bin_filter.eq("unfiltered")
                            & summary.selection.eq("all") & summary.coordinate.eq("true_coordinate")
                            & summary.readout.eq("decoded")
                            & summary.likelihood.eq("conditional_multinomial" if regime == "fixed_count" else "poisson")]
            for fraction, color in [(1., "#187a88"), (.5, "#bf4b42"), (.25, "#656565")]:
                data = local[local.cell_fraction.eq(fraction)].sort_values("gradient")
                ax.plot(data.gradient, data.normalized_slope, "o-", label=f"{fraction:.0%} cells", color=color)
                ax.fill_between(data.gradient, data.animal_ci95_low, data.animal_ci95_high, color=color, alpha=.15)
            ax.plot([-.5, .5], [-.5, .5], "k--", linewidth=1, label="Injected gradient")
            ax.axhline(0, color="gray", linewidth=.6)
            ax.set(xlabel="Injected spatial speed gradient", ylabel="Decoded normalized slope",
                   title=f"{dataset}\n{regime}")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Empirical-map surrogate recovery: posterior mean, known spatial coordinate\nNo continuity selection; not an estimate of biological speed", fontsize=12)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def technical_gates(events, trials, observations, sessions, unchanged):
    expected = len(trials) * len(REGIMES) * 3 * 2 * 2 * 2
    checks = [
        ("all_trial_conditions_present", len(trials) > 0 and len(events) == expected and not events.duplicated(KEY).any()),
        ("all_draws_scored_or_explicitly_short", events.status.isin(["scored", "insufficient_source_duration"]).all() and events.status.eq("scored").any()),
        ("all_sessions_represented", events[["dataset", "animal", "session"]].drop_duplicates().shape[0] == sessions),
        ("fixed_population_counts_verified", len(observations) > 0 and observations.regime.eq("fixed_count").any() and observations.loc[observations.regime.eq("fixed_count"), "fixed_totals_verified"].all()),
        ("native_cell_removal_verified", len(observations) > 0 and observations.regime.ne("fixed_count").any() and observations.loc[observations.regime.ne("fixed_count"), "native_subset_verified"].all()),
        ("inputs_and_code_unchanged", unchanged),
    ]
    checks.append(("overall_technical", all(bool(value) for _, value in checks)))
    return pd.DataFrame([{"gate": name, "passed": bool(value)} for name, value in checks])


def run(args):
    if args.events_per_session < 1 or args.bootstraps < 100:
        raise ValueError("positive event count and >=100 bootstrap replicates required")
    root, out = args.input_dir.resolve(), args.output_dir.resolve()
    inputs = {"input_manifest": root / "coverage_input_manifest.json", "sessions": root / "coverage_input_sessions.csv",
              "candidates": root / "coverage_input_candidates.csv", "input_gates": root / "coverage_input_gate_summary.csv",
              "script": Path(__file__), "protocol": ROOT / "docs/replay_coverage_recovery_protocol.md",
              "decoder": ROOT / "src/hipporeplayimm/replay_coverage.py", "recovery": ROOT / "src/hipporeplayimm/replay_coverage_recovery.py",
              "validation": ROOT / "src/hipporeplayimm/replay_coverage_validation.py", "data": ROOT / "src/hipporeplayimm/replay_coverage_data.py",
              "encoding": ROOT / "src/hipporeplayimm/encoding.py", "subsampling": ROOT / "scripts/analyze_replay_coverage_subsampling.py",
              "provenance": ROOT / "scripts/_provenance.py"}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if not args.allow_dirty and provenance.get("git_dirty") is not False:
        raise ValueError("production requires a verified clean commit; --allow-dirty is smoke-only")
    source_gate = pd.read_csv(inputs["input_gates"])
    if source_gate.loc[source_gate.gate.eq("overall_input_readiness"), "passed"].tolist() != [True]:
        raise ValueError("input readiness must pass")
    sessions = pd.read_csv(inputs["sessions"])
    if args.sessions:
        sessions = sessions[sessions.apply(lambda r: f"{r.dataset}:{r.animal}:{r.session}" in args.sessions, axis=1)]
        if len(sessions) != len(set(args.sessions)):
            raise ValueError("unknown or duplicate session selection")
    if sessions.empty or not sessions.status.eq("cached").all():
        raise ValueError("complete sessions required")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "input_snapshot"
    snapshot.mkdir()
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    for name, path in inputs.items():
        shutil.copyfile(path, snapshot / f"{name}__{path.name}")
    frames, specs, obs_specs, populations, caches = [], [], [], [], {}
    for record in sessions.itertuples(index=False):
        if file_sha256(record.artifact_path) != record.artifact_sha256:
            raise ValueError("cache hash mismatch")
        caches[record.artifact_path] = record.artifact_sha256
        print(f"Generating and scoring {record.dataset}:{record.animal}:{record.session}", flush=True)
        with np.load(record.artifact_path, allow_pickle=False) as cache:
            trials, session_specs, observation_specs, subsets, ids, rates, grid, bounds = build_trials(cache, record, args)
        frame = score_trials(trials, subsets, rates, grid, bounds, args)
        frames.append(frame)
        specs.extend(session_specs)
        obs_specs.extend(observation_specs)
        for fraction, subset in subsets.items():
            populations.append({"dataset": record.dataset, "animal": record.animal, "session": record.session,
                                "cell_fraction": fraction, "cell_ids": ids[subset].tolist(),
                                "cell_indices_in_qc_population": subset.tolist(), "decoder_states": len(grid)})
        stem = f"{record.dataset}__{record.animal}__{record.session.split('/')[-1]}"
        frame.to_csv(out / f"events__{stem}.csv", index=False)
        pd.DataFrame(session_specs).to_csv(out / f"trials__{stem}.csv", index=False)
        pd.DataFrame(observation_specs).to_csv(out / f"observations__{stem}.csv", index=False)
        print(f"Completed {stem}: {len(trials)} truth draws, {len(frame)} metric rows", flush=True)
    events, trial_table, observation_table = pd.concat(frames, ignore_index=True), pd.DataFrame(specs), pd.DataFrame(obs_specs)
    session_summary, gradients = summarize_events(events, args.speed_cm_s)
    animals, summary = cluster_summary(gradients, args.seed + 1, args.bootstraps)
    unchanged = all(file_sha256(path) == hashes[name] for name, path in inputs.items()) and all(file_sha256(path) == sha for path, sha in caches.items())
    gates = technical_gates(events, trial_table, observation_table, len(sessions), unchanged)
    for suffix, table in [("event_metrics", events), ("trials", trial_table), ("observations", observation_table),
                          ("session_summary", session_summary), ("session_gradient_recovery", gradients),
                          ("animal_gradient_recovery", animals), ("gradient_recovery_summary", summary), ("gate_summary", gates)]:
        table.to_csv(out / f"coverage_recovery_{suffix}.csv", index=False)
    plot_gradients(summary, out / "coverage_recovery_gradients.png")
    manifest = {**provenance, "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "input_sha256_at_start": hashes, "cache_sha256": caches, "populations": populations,
                "truth_trials": len(trial_table), "metric_rows": len(events), "sessions": len(sessions),
                "source_candidates": len(trial_table[["dataset", "animal", "session", "source_event_index"]].drop_duplicates()),
                "candidate_draw_rule": "uniform_without_replacement_no_decoded_outcome_redraw",
                "rate_map_truth": "bilinear_empirical_surrogate_not_biological_truth_or_RatInABox",
                "likelihood_limit": "observation_family_matched_baselines_but_moving_position_within_window_is_not_static_likelihood_matched",
                "spatial_gradient": "normalized_horizontal_position_not_wall_distance",
                "heldout_evaluation": False, "claim_boundary": "development_recovery_benchmark_not_biological_uniformity_or_calibrated_replay_test",
                "output_sha256": {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}}
    (out / "coverage_recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "coverage_recovery_report.md").write_text(
        "# Empirical-Map Known-Path Recovery\n\n"
        f"{len(trial_table)} truth draws, {len(events)} repeated metric rows, {len(sessions)} sessions.\n\n"
        "Uniformly sampled candidate durations/counts were frozen before decoding; no trial was redrawn for poor outcomes. "
        "The population maps are empirical surrogates, not known biological tuning. PF uses grid extent, not verified walls.\n\n"
        "Poisson removals lose spikes and cells together. Fixed-count trials preserve every source 5 ms total across subsets, "
        "with count-conditioned decoding as primary. Shared gain is a deliberate fixed-rate-model mismatch. "
        "Even primary observation families retain a moving-within-window/static-position approximation.\n\n"
        "Continuity uses 20 ms overlapping windows at 5 ms strides. Speeds use non-overlapping 20 ms windows. "
        "Arclength truth and window-mean chord truth are distinct, especially at reflections. "
        "Stationary/discontinuous retention is false geometric acceptance, not an empirical replay false-positive rate.\n\n"
        "Gradient slopes use equal-event moments, then sessions within animals and equal animals. "
        "Intervals reflect animal clustering conditional on these simulation draws; independent seed/population evaluation remains required. "
        "Sparse or spatially restricted selected subsets abstain. No real speed-uniformity or new neuronal-mechanism claim follows.\n"
    )
    if not gates.passed.all():
        raise RuntimeError("recovery technical gates failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--events-per-session", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--speed-cm-s", type=float, default=1000.)
    parser.add_argument("--rate-scale", type=float, default=3.)
    parser.add_argument("--fine-s", type=float, default=.001)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--allow-dirty", action="store_true", help="development smoke only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
