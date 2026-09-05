#!/usr/bin/env python3
"""Independent paths with paired spike loss/restoration and information doses."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
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
from hipporeplayimm.replay_coverage_counterfactual import (
    aggregate_fine_counts,
    paired_counts,
    pooled_rates,
    restored_rates,
    shuffle_base_path,
)
from hipporeplayimm.replay_coverage_data import array_sha256
from hipporeplayimm.replay_coverage_recovery import (
    decode_batches,
    map_interpolator,
    paired_speed_moments,
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
from scripts.simulate_replay_coverage_recovery import (
    TRUTHS,
    seed_parts,
    simulation_domain,
    summarize_events,
)

CONDITIONS = [("native", 1.), ("native", .5), ("native", .25),
              ("pooled_removed", .5), ("pooled_removed", .25),
              ("count_restored", .5), ("count_restored", .25)]
TRIAL_KEY = ["dataset", "animal", "session", "source_event_index", "replicate", "truth_kind", "gradient"]
SCORE_KEY = TRIAL_KEY + ["rate_scale", "regime", "cell_fraction", "likelihood", "estimator", "bin_filter"]


def build_counterfactual_trials(cache, profiles, record, args, replicate):
    mask = cache["unit_qc_mask"].astype(bool)
    ids, maps = cache["cell_ids"][mask], cache["rates_hz"][mask]
    domain, domain_source = simulation_domain(cache)
    occupied = decoding_support(cache["valid_spatial_bins"], cache["bin_centers_cm"], cache["arena_bounds_cm"])
    rates, grid = maps[:, occupied], cache["bin_centers_cm"][occupied]
    key = f"{record.dataset}:{record.animal}:{record.session}"
    subsets = {f: idx for f, _, idx in population_subsets(ids, [.5, .25], 1, args.seed, f"{key}:rep{replicate}")}
    evaluate = map_interpolator(maps, cache["x_edges_cm"], cache["y_edges_cm"])
    source_lookup = {int(e): i for i, e in enumerate(cache["candidate_event_indices"])}
    trials, specs, observations = [], [], []
    for source in profiles.itertuples(index=False):
        event = int(source.source_event_index)
        index = source_lookup[event]
        a, b = cache["candidate_offsets"][index:index + 2]
        durations = cache["candidate_base_durations_s"][a:b]
        event_windows(cache["candidate_base_counts"][a:b, mask], durations)
        n_base = int(np.isclose(durations, .005, rtol=0, atol=1e-9).sum())
        if n_base != source.n_base_bins or n_base < 4:
            raise ValueError("frozen source duration changed or requires explicit short-trial support")
        constant_path, constant_counts = None, None
        for kind, gradient in TRUTHS + [("shuffled_continuous", 0.)]:
            identity = {"dataset": record.dataset, "animal": record.animal, "session": record.session,
                        "source_event_index": event, "replicate": replicate, "truth_kind": kind, "gradient": gradient}
            path_seed = seed_parts(args.seed, key, event, 100 * replicate + 2)
            obs_seed = seed_parts(args.seed, key, event, 100 * replicate + 3)
            if kind == "shuffled_continuous":
                if constant_path is None or constant_counts is None:
                    raise AssertionError("shuffle parent was not generated for this source")
                permutation = np.random.default_rng(seed_parts(args.seed, key, event, 100 * replicate + 4)).permutation(n_base)
                path = shuffle_base_path(constant_path, permutation)
                base_counts = {k: values[permutation] for k, values in constant_counts.items()}
                shuffle_hash = array_sha256(permutation)
                for k, values in base_counts.items():
                    if not np.array_equal(values.sum(axis=0), constant_counts[k].sum(axis=0)):
                        raise AssertionError("whole-bin shuffle changed unit spike totals")
            else:
                path = simulate_path(n_base, domain, kind, gradient, args.speed_cm_s, path_seed, fine_s=.001)
                fine = paired_counts(evaluate(path["midpoints_cm"]), subsets, args.rate_scales, .001, obs_seed)
                base_counts = {k: aggregate_fine_counts(v, 5) for k, v in fine.items()}
                shuffle_hash = "not_shuffled"
                if kind == "continuous" and gradient == 0:
                    constant_path, constant_counts = path, base_counts
            truth = truth_windows(path)
            indices = true_state_indices(truth["center_cm"], cache["x_edges_cm"], cache["y_edges_cm"], occupied)
            spec = {**identity, "source_duration_s": float(durations.sum()), "simulated_duration_s": n_base * .005,
                    "source_spikes": int(cache["candidate_base_counts"][a:b, mask].sum()),
                    "n_base_bins": n_base, "domain_source": domain_source,
                    "domain_cm": json.dumps(domain.tolist()), "path_seed": json.dumps(path_seed), "observation_seed": json.dumps(obs_seed),
                    "path_sha256": array_sha256(path["midpoints_cm"]), "truth_speed_sha256": array_sha256(path["speed_cm_s"]),
                    "shuffle_permutation_sha256": shuffle_hash,
                    "truth_supported_fraction": float((indices >= 0).mean()),
                    "truth_geometric_eligible": continuity_metrics(truth["window_mean_cm"])["continuity_pass"]}
            for (scale, family, fraction), counts in base_counts.items():
                full = base_counts[scale, "native", 1.]
                native = base_counts[scale, "native", fraction]
                conserved = np.array_equal(counts.sum(axis=1), full.sum(axis=1)) if family in {"count_restored", "pooled_removed"} else np.array_equal(counts, full[:, subsets[fraction]])
                if not conserved or (family == "count_restored" and np.any(counts < native)):
                    raise AssertionError("paired base-count invariant failed")
                cells_only = counts[:, :-1] if family == "pooled_removed" else counts
                observations.append({**identity, "rate_scale": scale, "regime": family, "cell_fraction": fraction,
                                     "counts_sha256": array_sha256(counts), "full_counts_sha256": array_sha256(full),
                                     "spikes": int(counts.sum()), "active_units": int(np.count_nonzero(cells_only.sum(axis=0))),
                                     "observation_channels": counts.shape[1],
                                     "paired_counts_verified": True})
            trials.append({"spec": spec, "counts": base_counts, "truth": truth, "truth_indices": indices})
            specs.append(spec)
    return trials, specs, observations, subsets, ids, rates, grid, domain


def score_batch(trials, subsets, rates, grid, domain, args, scale):
    rows = []
    for family, fraction in CONDITIONS:
        subset = subsets[fraction]
        chunks = [event_windows(t["counts"][scale, family, fraction], np.full(t["spec"]["n_base_bins"], .005)) for t in trials]
        offsets = np.r_[0, np.cumsum([len(x) for x in chunks])]
        counts = np.concatenate(chunks)
        truth_indices = np.concatenate([t["truth_indices"] for t in trials])
        encoding = rates[subset] if family == "native" else (pooled_rates(rates, subset) if family == "pooled_removed" else restored_rates(rates, subset))
        for likelihood in ["poisson", "conditional_multinomial"]:
            decoded = decode_batches(counts, encoding * scale, grid, truth_indices, likelihood)
            for trial, values, a, b in zip(trials, chunks, offsets[:-1], offsets[1:], strict=True):
                for estimator in ["map", "posterior_mean"]:
                    for support in [False, True]:
                        support_values = values[:, :-1] if family == "pooled_removed" else values
                        keep = (support_values.sum(axis=1) >= 3) & (np.count_nonzero(support_values, axis=1) >= 2) if support else np.ones(len(values), bool)
                        path = decoded[estimator][a:b]
                        errors = np.linalg.norm(path - trial["truth"]["center_cm"], axis=1)
                        rows.append({**trial["spec"], "rate_scale": scale, "regime": family,
                                     "cell_fraction": fraction, "retained_cells": len(subset), "observation_channels": values.shape[1], "likelihood": likelihood,
                                     "estimator": estimator, "bin_filter": "at_least_2cells_3spikes" if support else "unfiltered",
                                     "status": "scored", "median_position_error_cm": float(np.median(errors[keep])) if keep.any() else np.nan,
                                     "hpd95_coverage": float(decoded["hpd95"][a:b][keep].mean()) if keep.any() else np.nan,
                                     **observed_metrics(path, support_values, decoded["posterior_rms_cm"][a:b], decoded["posterior_entropy_nats"][a:b], support),
                                     **paired_speed_moments(path, support_values, trial["truth"], domain, support)})
    frame = pd.DataFrame(rows)
    if len(frame) != len(trials) * len(CONDITIONS) * 8 or frame.duplicated(SCORE_KEY).any():
        raise AssertionError("missing/duplicate simulation scoring conditions")
    return frame


def event_decomposition(events):
    """Algebraic information-loss partition on the SAME truth-eligible events."""
    local = events[events.truth_kind.eq("continuous") & events.gradient.eq(0) & events.truth_geometric_eligible]
    index = TRIAL_KEY + ["rate_scale", "likelihood", "estimator", "bin_filter"]
    pivot = local.pivot(index=index, columns=["regime", "cell_fraction"], values="continuity_pass").astype(float)
    frames = []
    for fraction in [.5, .25]:
        columns = [("native", 1.), ("native", fraction), ("pooled_removed", fraction), ("count_restored", fraction)]
        if len(pivot) and (any(col not in pivot for col in columns) or pivot[columns].isna().any().any()):
            raise ValueError("incomplete paired event conditions")
        if not len(pivot):
            continue
        full, native, pooled, restored = (pivot[col] for col in columns)
        part = pivot.index.to_frame(index=False)
        part["cell_fraction"] = fraction
        part["native_minus_full"] = (native - full).to_numpy()
        part["pooled_minus_full"] = (pooled - full).to_numpy()
        part["native_minus_pooled"] = (native - pooled).to_numpy()
        part["restored_minus_full"] = (restored - full).to_numpy()
        part["native_minus_restored"] = (native - restored).to_numpy()
        if not np.array_equal(part.native_minus_full, part.pooled_minus_full + part.native_minus_pooled):
            raise AssertionError("partition identity failed")
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=index + ["cell_fraction", "native_minus_full", "pooled_minus_full", "native_minus_pooled", "restored_minus_full", "native_minus_restored"])


def analysis_sets(table, primary_scale, dose_replicates):
    return pd.concat([table[table.rate_scale.eq(primary_scale)].assign(analysis_set="primary_replication"),
                      table[table.replicate.lt(dose_replicates)].assign(analysis_set="paired_dose")], ignore_index=True)


def aggregate_decomposition(parts, seed, bootstraps, primary_scale=3., dose_replicates=1):
    parts = analysis_sets(parts, primary_scale, dose_replicates)
    group = ["analysis_set", "dataset", "rate_scale", "cell_fraction", "likelihood", "estimator", "bin_filter"]
    values = ["native_minus_full", "pooled_minus_full", "native_minus_pooled", "restored_minus_full", "native_minus_restored"]
    by_rep = parts.groupby(group + ["animal", "session", "replicate"], as_index=False)[values].mean()
    by_session = by_rep.groupby(group + ["animal", "session"], as_index=False)[values].mean()
    animals = by_session.groupby(group + ["animal"], as_index=False)[values].mean()
    rows, rng = [], np.random.default_rng(seed)
    for key, frame in animals.groupby(group, sort=True):
        for value in values:
            data = frame[value].dropna().to_numpy()
            lo, hi = np.nan, np.nan
            if len(data) > 1:
                lo, hi = np.quantile(rng.choice(data, (bootstraps, len(data))).mean(axis=1), [.025, .975])
            rows.append({**dict(zip(group, key, strict=True)), "contrast": value, "animals": len(data),
                         "effect": float(data.mean()) if len(data) else np.nan, "ci95_low": lo, "ci95_high": hi})
    return by_rep, animals, pd.DataFrame(rows, columns=group + ["contrast", "animals", "effect", "ci95_low", "ci95_high"])


def aggregate_gradients(gradients, seed, bootstraps, primary_scale=3., dose_replicates=1):
    gradients = analysis_sets(gradients, primary_scale, dose_replicates)
    group = ["analysis_set", "dataset", "rate_scale", "regime", "cell_fraction", "likelihood", "estimator", "bin_filter", "selection", "coordinate", "readout"]
    pivot = gradients[gradients.gradient.ne(0)].pivot(index=group + ["animal", "session", "replicate"], columns="gradient", values="normalized_slope")
    response = (pivot[.5] - pivot[-.5]).rename("response").reset_index()
    sessions = response.groupby(group + ["animal", "session"], as_index=False).response.mean()
    animals = sessions.groupby(group + ["animal"], as_index=False).response.mean()
    rows, rng = [], np.random.default_rng(seed)
    for key, local in animals.groupby(group, sort=True):
        data = local.response.dropna().to_numpy()
        lo, hi = np.nan, np.nan
        if len(data) > 1:
            lo, hi = np.quantile(rng.choice(data, (bootstraps, len(data))).mean(axis=1), [.025, .975])
        rows.append({**dict(zip(group, key, strict=True)), "animals": len(local), "finite_animals": len(data),
                     "response": float(data.mean()) if len(data) else np.nan, "ci95_low": lo, "ci95_high": hi,
                     "injected_gradient_difference": 1.})
    return response, pd.DataFrame(rows)


def plot_results(decomposition, gradients, output):
    datasets = sorted(gradients.dataset.unique())
    doses = gradients.rate_scale.unique()
    limits = (float(min(doses)) / 1.25, float(max(doses)) * 1.25)
    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 8), squeeze=False, constrained_layout=True)
    for col, dataset in enumerate(datasets):
        top, bottom = axes[:, col]
        local = decomposition[decomposition.analysis_set.eq("paired_dose") & decomposition.dataset.eq(dataset) & decomposition.cell_fraction.eq(.5)
                              & decomposition.likelihood.eq("conditional_multinomial") & decomposition.estimator.eq("map") & decomposition.bin_filter.eq("unfiltered")]
        for contrast, label, color in [("native_minus_full", "Remove cells + spikes", "#bf4b42"), ("pooled_minus_full", "Keep removed spikes pooled", "#187a88")]:
            part = local[local.contrast.eq(contrast)].sort_values("rate_scale")
            top.plot(part.rate_scale, 100 * part.effect, "o-", color=color, label=label)
            top.fill_between(part.rate_scale, 100 * part.ci95_low, 100 * part.ci95_high, color=color, alpha=.15)
        top.axhline(0, color="gray", linewidth=.7)
        top.set(xscale="log", xlim=limits, xlabel="Firing-rate multiplier", ylabel="Half-minus-full recovery (pp)", title=dataset)
        top.legend(frameon=False, fontsize=9)
        local = gradients[gradients.analysis_set.eq("paired_dose") & gradients.dataset.eq(dataset) & gradients.likelihood.eq("poisson") & gradients.regime.eq("native")
                          & gradients.cell_fraction.eq(1.) & gradients.estimator.eq("posterior_mean") & gradients.bin_filter.eq("unfiltered")
                          & gradients.selection.eq("all") & gradients.coordinate.eq("true_coordinate")]
        for readout, label, color in [("decoded", "Decoded", "#187a88"), ("true_chord", "Window-mean truth", "#777777")]:
            part = local[local.readout.eq(readout)].sort_values("rate_scale")
            bottom.plot(part.rate_scale, part.response, "o-", color=color, label=label)
            bottom.fill_between(part.rate_scale, part.ci95_low, part.ci95_high, color=color, alpha=.15)
        bottom.axhline(1, linestyle="--", color="black", linewidth=.8, label="Injected contrast")
        bottom.set(xscale="log", xlim=limits, xlabel="Firing-rate multiplier", ylabel="Response to opposite speed gradients")
        if not np.isfinite(local.response).any():
            bottom.text(.5, .4, "Insufficient events for gradient estimation", transform=bottom.transAxes, ha="center", fontsize=9)
        bottom.legend(frameon=False, fontsize=9)
        for ax in [top, bottom]:
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Known-path paired counterfactuals: information and recovery\nSynthetic replication on fixed maps; not biological speed estimates", fontsize=12)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args):
    if args.replicates < 1 or not 1 <= args.dose_replicates <= args.replicates or args.primary_rate_scale not in args.rate_scales or args.bootstraps < 100 or (args.max_profiles is not None and args.max_profiles < 1):
        raise ValueError("invalid replication/dose design")
    reference, out = args.reference_dir.resolve(), args.output_dir.resolve()
    original = json.loads((reference / "coverage_recovery_manifest.json").read_text())
    root = Path(original["parameters"]["input_dir"])
    profiles_path = reference / "coverage_recovery_trials.csv"
    if file_sha256(profiles_path) != original["output_sha256"][profiles_path.name]:
        raise ValueError("frozen source profiles changed")
    profiles = pd.read_csv(profiles_path)
    profiles = profiles[profiles.truth_kind.eq("continuous") & profiles.gradient.eq(0)]
    if profiles.duplicated(["dataset", "animal", "session", "source_event_index"]).any():
        raise ValueError("duplicate source profiles")
    sessions = pd.read_csv(root / "coverage_input_sessions.csv")
    sessions = sessions[sessions.artifact_path.isin(original["cache_sha256"])]
    if args.sessions:
        sessions = sessions[sessions.apply(lambda r: f"{r.dataset}:{r.animal}:{r.session}" in args.sessions, axis=1)]
        if len(sessions) != len(set(args.sessions)):
            raise ValueError("unknown/duplicate requested sessions")
    if not len(sessions):
        raise ValueError("no reference sessions")
    inputs = {"reference_manifest": reference / "coverage_recovery_manifest.json", "profiles": profiles_path,
              "sessions": root / "coverage_input_sessions.csv", "script": Path(__file__),
              "protocol": ROOT / "docs/replay_coverage_counterfactual_protocol.md",
              **{name: ROOT / f"src/hipporeplayimm/{name}.py" for name in ["replay_coverage", "replay_coverage_counterfactual", "replay_coverage_recovery", "replay_coverage_data", "replay_coverage_validation", "encoding"]},
              **{name: ROOT / f"scripts/{name}.py" for name in ["_provenance", "simulate_replay_coverage_recovery", "analyze_replay_coverage_subsampling"]}}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if not args.allow_dirty and provenance.get("git_dirty") is not False:
        raise ValueError("production requires a clean code commit")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "input_snapshot"
    snapshot.mkdir()
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    for name, path in inputs.items():
        shutil.copyfile(path, snapshot / f"{name}__{path.name}")
    cache_hashes, specs, obs, populations, batches = {}, [], [], [], []
    summaries, gradients, partitions = [], [], []
    for record in sessions.itertuples(index=False):
        sha = file_sha256(record.artifact_path)
        if sha != original["cache_sha256"][record.artifact_path]:
            raise ValueError("source cache changed")
        cache_hashes[record.artifact_path] = sha
        local = profiles[profiles.dataset.eq(record.dataset) & profiles.animal.eq(record.animal) & profiles.session.eq(record.session)]
        if args.max_profiles:
            local = local.sort_values("source_event_index").head(args.max_profiles)
        if not len(local):
            raise ValueError("session lost from frozen source set")
        for replicate in range(args.replicates):
            label = f"{record.dataset}__{record.animal}__{record.session.split('/')[-1]}__rep{replicate}"
            print(f"Generating {label} profiles={len(local)}", flush=True)
            with np.load(record.artifact_path, allow_pickle=False) as cache:
                trials, trial_specs, observation_specs, subsets, ids, rates, grid, domain = build_counterfactual_trials(cache, local, record, args, replicate)
            specs.extend(trial_specs)
            obs.extend(observation_specs)
            for fraction, subset in subsets.items():
                populations.append({"dataset": record.dataset, "animal": record.animal, "session": record.session,
                                    "replicate": replicate, "cell_fraction": fraction, "cell_ids": ids[subset].tolist()})
            scales = args.rate_scales if replicate < args.dose_replicates else [args.primary_rate_scale]
            for scale in scales:
                frame = score_batch(trials, subsets, rates, grid, domain, args, scale)
                filename = f"events__{label}__scale{scale:g}.csv"
                frame.to_csv(out / filename, index=False)
                summary, gradient = summarize_events(frame, args.speed_cm_s)
                for table in [summary, gradient]:
                    table["replicate"], table["rate_scale"] = replicate, scale
                summaries.append(summary)
                gradients.append(gradient)
                partitions.append(event_decomposition(frame))
                batches.append({"dataset": record.dataset, "animal": record.animal, "session": record.session,
                                "replicate": replicate, "rate_scale": scale, "source_profiles": len(local), "truth_trials": len(trials),
                                "metric_rows": len(frame), "file": filename, "sha256": file_sha256(out / filename)})
                print(f"Completed {label} scale={scale:g}: {len(frame)} rows", flush=True)
    session_summary, gradient_table, partition = pd.concat(summaries, ignore_index=True), pd.concat(gradients, ignore_index=True), pd.concat(partitions, ignore_index=True)
    rep_part, animals, paired = aggregate_decomposition(partition, args.seed + 1, args.bootstraps, args.primary_rate_scale, args.dose_replicates)
    rep_response, response = aggregate_gradients(gradient_table, args.seed + 2, args.bootstraps, args.primary_rate_scale, args.dose_replicates)
    batch_table, spec_table, obs_table = pd.DataFrame(batches), pd.DataFrame(specs), pd.DataFrame(obs)
    unique_sources = len(spec_table[["dataset", "animal", "session", "source_event_index"]].drop_duplicates())
    expected_batches = len(sessions) * (args.dose_replicates * len(args.rate_scales) + args.replicates - args.dose_replicates)
    expected_rows = unique_sources * 6 * (args.dose_replicates * len(args.rate_scales) + args.replicates - args.dose_replicates) * len(CONDITIONS) * 8
    checks = [("all_sources_replicates_and_truths", len(spec_table) == unique_sources * args.replicates * 6 and not spec_table.duplicated(TRIAL_KEY).any()),
              ("all_scoring_batches_present", len(batch_table) == expected_batches and int(batch_table.metric_rows.sum()) == expected_rows and expected_rows > 0),
              ("all_observation_invariants", len(obs_table) == len(spec_table) * len(args.rate_scales) * len(CONDITIONS) and obs_table.paired_counts_verified.all()),
              ("paired_decomposition_available", len(partition) > 0),
              ("inputs_and_code_unchanged", all(file_sha256(path) == hashes[name] for name, path in inputs.items()) and all(file_sha256(path) == digest for path, digest in cache_hashes.items()))]
    checks.append(("overall_technical", all(bool(v) for _, v in checks)))
    gates = pd.DataFrame([{"gate": k, "passed": bool(v)} for k, v in checks])
    for name, table in [("trials", spec_table), ("observations", obs_table), ("batches", batch_table), ("session_summary", session_summary),
                        ("session_gradients", gradient_table), ("event_decomposition", partition), ("replicate_decomposition", rep_part),
                        ("animal_decomposition", animals), ("decomposition_summary", paired), ("replicate_gradient_response", rep_response),
                        ("gradient_response_summary", response), ("gate_summary", gates)]:
        table.to_csv(out / f"coverage_counterfactual_{name}.csv", index=False)
    plot_results(paired, response, out / "coverage_counterfactual.png")
    manifest = {**provenance, "created_at_utc": datetime.now(UTC).isoformat(),
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "input_sha256_at_start": hashes, "cache_sha256": cache_hashes, "populations": populations,
                "sessions": len(sessions), "source_profiles": unique_sources, "source_replicate_units": unique_sources * args.replicates,
                "generated_truth_paths_before_shuffle": unique_sources * args.replicates * 5,
                "truth_trials_including_shuffle": len(spec_table), "metric_rows": int(batch_table.metric_rows.sum()),
                "rates": "native rates times scale; pooled sum-of-removed maps; oracle-restored rates preserve full intensity at every spatial state",
                "pooled_control": "data-only deterministic sum of removed spikes, with sum-of-removed-maps likelihood; no oracle reassignment",
                "support_rule": "only retained sorted-cell spikes/cells count toward support, excluding the pooled channel; pooled/native-subset share masks",
                "coupling": "nested Poisson exposures; pooled removed counts and secondary true-position-based label restoration preserve exact fine-bin population totals and retained spikes",
                "independence_scope": "new random paths/population subsets/spikes on SAME empirical maps and frozen duration profiles, not new animals or held-out empirical maps",
                "claim_boundary": "synthetic replication and sensitivity, not biological uniformity or validated real-data correction",
                "output_sha256": {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()}}
    (out / "coverage_counterfactual_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not gates.passed.all():
        raise RuntimeError("counterfactual technical checks failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--dose-replicates", type=int, default=1)
    parser.add_argument("--rate-scales", nargs="+", type=float, default=[1., 3., 10., 30.])
    parser.add_argument("--primary-rate-scale", type=float, default=3.)
    parser.add_argument("--speed-cm-s", type=float, default=1000.)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--max-profiles", type=int, help="technical smoke only; deterministic prefix of frozen profiles")
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
