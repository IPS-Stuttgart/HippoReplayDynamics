#!/usr/bin/env python3
"""Count-coherent known-generator recovery for frozen hc-11 cell prediction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_hc11_count_conditioned_prediction as frozen
from _provenance import build_script_provenance, file_sha256

from hipporeplayimm.frozen_posterior_prediction import posterior_sha256

GENERATORS = frozen.MODELS
MODELS = frozen.MODELS
VARIANTS = frozen.VARIANTS
REPLICATES = 50
BOOTSTRAPS = 2000
SEED = 20260908
PARENT_DIGEST = "bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07"
CONTRASTS = {
    "imm_minus_iid_position": ("first_order_imm", "iid_position"),
    "imm_minus_static_location": ("first_order_imm", "static_location"),
    "imm_minus_diffusion": ("first_order_imm", "diffusion"),
    "diffusion_minus_iid": ("diffusion", "iid_position"),
    "static_minus_iid": ("static_location", "iid_position"),
    "oracle_minus_iid": ("oracle", "iid_position"),
}


def draw_latent(generator, n_bins, n_time, kernels, rng):
    if generator not in GENERATORS or n_bins < 2 or n_time < 1:
        raise ValueError("invalid generator or dimensions")
    if generator == "iid_position":
        return rng.integers(n_bins, size=n_time), np.full(n_time, -1)
    if generator == "static_location":
        return np.full(n_time, rng.integers(n_bins)), np.full(n_time, -1)
    n_states = n_bins * (3 if generator == "first_order_imm" else 1)
    state = int(rng.integers(n_states))
    states = [state]
    for matrix in kernels[generator]:
        probabilities = np.asarray(matrix[:, state].toarray()).ravel()
        state = int(rng.choice(n_states, p=probabilities))
        states.append(state)
    if len(states) != n_time:
        raise ValueError("transition duration mismatch")
    states = np.asarray(states)
    return states % n_bins, states // n_bins if generator == "first_order_imm" else np.full(n_time, -1)


def draw_population(totals, rates, path, rng):
    totals = np.asarray(totals)
    if totals.shape != np.shape(path) or np.any(totals < 0) or np.any(totals != np.floor(totals)) or not np.isfinite(totals).all():
        raise ValueError("invalid observed totals")
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("invalid generating map")
    probabilities = rates[:, path] / rates[:, path].sum(axis=0, keepdims=True)
    return np.asarray([rng.multinomial(int(n), probabilities[:, t]) for t, n in enumerate(totals)], dtype=np.int64)


def generated_event(data, uid, generator, replicate, session, kernels):
    phase, event_id = uid.split("_", 1)
    seed = frozen.stable_seed(SEED, "hc11_count_recovery", session, replicate, generator, phase, int(event_id))
    rng = np.random.default_rng(seed)
    direction = int(rng.integers(2))
    totals = data[f"counts_{uid}"].sum(axis=1)
    path, modes = draw_latent(generator, len(data["centers"]), len(totals), kernels, rng)
    counts = draw_population(totals, data[f"rates_direction_mixture_{direction}"], path, rng)
    return counts, path, modes, direction, seed


def predict_event(counts, edges, rates_by_variant, train, held, kernels):
    """The prediction API has no generator, latent path or true direction input."""
    results = []
    for variant, rates in rates_by_variant.items():
        train_parts = [frozen.pf.likelihood_parts(counts[:, train], r[train], np.diff(edges)) for r in rates]
        posteriors = {model: frozen.infer_training(train_parts, "count_conditioned", 1.0, model, kernels) for model in MODELS}
        hashes = {model: posterior_sha256(q) for model, q in posteriors.items()}
        held_parts = [frozen.pf.likelihood_parts(counts[:, held], r[held], np.diff(edges)) for r in rates]
        scores = {f"score_{model}": frozen.heldout_score(q, held_parts, "count_conditioned") for model, q in posteriors.items()}
        if any(hashes[m] != posterior_sha256(q) for m, q in posteriors.items()):
            raise ValueError("held-out data changed inference")
        results.append({"encoding_variant": variant, **scores})
    return results


def shard(task):
    source, output, session, selected, start, stop = task
    with np.load(Path(source) / f"{session}_cache.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    rates = {v: [data[f"rates_{v}_{d}"] for d in range(1 if v == "pooled" else 2)] for v in VARIANTS}
    rows, saved = [], {}
    for event in selected.itertuples(index=False):
        uid = f"{event.phase}_{event.event_id}"
        edges = data[f"edges_{uid}"]
        kernels = frozen.transitions(data["centers"], edges, str(data["topology"]), float(data["track_length"]))
        for replicate in range(start, stop):
            for generator in GENERATORS:
                counts, path, modes, direction, seed = generated_event(data, uid, generator, replicate, session, kernels)
                if not np.array_equal(counts.sum(axis=1), data[f"counts_{uid}"].sum(axis=1)):
                    raise ValueError("simulation changed observed count profile")
                name = f"{generator}_{replicate}_{uid}"
                saved[f"counts_{name}"] = counts
                saved[f"path_{name}"] = path
                saved[f"modes_{name}"] = modes
                saved[f"direction_{name}"] = np.array(direction)
                count_hash = frozen.pf._hash_array(counts)
                for split in range(frozen.N_SPLITS):
                    train, held = data[f"train_{split}"], data[f"held_{split}"]
                    if np.intersect1d(train, held).size or not len(train) or not len(held):
                        raise ValueError("invalid frozen split")
                    predictions = predict_event(counts, edges, rates, train, held, kernels)
                    true_parts = frozen.pf.likelihood_parts(counts[:, held], rates["direction_mixture"][direction][held], np.diff(edges))
                    oracle = float(true_parts["count_conditioned"][np.arange(len(path)), path].sum())
                    for prediction in predictions:
                        rows.append(
                            {
                                "session": session,
                                "rat": event.animal,
                                "phase": event.phase,
                                "event_index": int(event.event_id),
                                "generator": generator,
                                "replicate": replicate,
                                "split": split,
                                **prediction,
                                "score_oracle": oracle,
                                "seed": seed,
                                "true_direction": direction,
                                "counts_sha256": count_hash,
                                "n_spikes": int(counts.sum()),
                                "n_train_spikes": int(counts[:, train].sum()),
                                "n_heldout_spikes": int(counts[:, held].sum()),
                                "n_active_units": int((counts.sum(axis=0) > 0).sum()),
                                "n_train_units": len(train),
                                "n_heldout_units": len(held),
                                "n_time_bins": len(counts),
                                "status": "success",
                            }
                        )
    tag = f"{session}_{start:02d}_{stop:02d}"
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output) / f"{tag}_split_scores.csv", index=False)
    np.savez_compressed(Path(output) / f"{tag}_draws.npz", **saved)
    return {"tag": tag, "session": session, "replicate_start": start, "replicate_stop": stop, "rows": len(frame), "draws": len(saved) // 4}


def event_contrasts(scores):
    keys = ["session", "rat", "phase", "event_index", "generator", "replicate", "encoding_variant"]
    if scores.duplicated(keys + ["split"]).any():
        raise ValueError("duplicate simulation scores")
    for name, (a, b) in CONTRASTS.items():
        scores[name] = scores[f"score_{a}"] - scores[f"score_{b}"]
    for model in MODELS:
        scores[f"gain_{model}"] = scores[f"score_{model}"] - scores.score_iid_position
    values = list(CONTRASTS) + [f"gain_{model}" for model in MODELS]
    events = scores.groupby(keys, as_index=False)[values].median()
    gains = events[[f"gain_{model}" for model in MODELS]].to_numpy()
    order = np.argsort(gains, axis=1)
    ambiguous = gains[np.arange(len(gains)), order[:, -1]] - gains[np.arange(len(gains)), order[:, -2]] <= 1e-9
    events["raw_predictive_winner"] = np.asarray(MODELS)[order[:, -1]]
    events.loc[ambiguous, "raw_predictive_winner"] = "ambiguous"
    return events


def bootstrap_weights(cohort, draws=BOOTSTRAPS, seed=SEED):
    cohort = cohort.reset_index(drop=True)
    animals = sorted(cohort.rat.unique())
    groups = {rat: [g.index.to_numpy() for _, g in cohort[cohort.rat == rat].groupby("session")] for rat in animals}
    weights = np.zeros((draws, len(cohort)))
    rng = np.random.default_rng(seed)
    for b in range(draws):
        for rat in rng.choice(animals, len(animals), replace=True):
            sessions = groups[rat]
            chosen = rng.integers(0, len(sessions), len(sessions))
            n_events = sum(len(sessions[s]) for s in chosen)
            for s in chosen:
                picked = rng.choice(sessions[s], len(sessions[s]), replace=True)
                np.add.at(weights[b], picked, 1.0 / (len(animals) * n_events))
    if not np.allclose(weights.sum(axis=1), 1):
        raise ValueError("invalid bootstrap weights")
    return weights


def simulation_panels(events):
    records, animals = [], []
    for phase, phase_events in events.groupby("phase", sort=True):
        cohort = phase_events[["session", "rat", "event_index"]].drop_duplicates().sort_values(["rat", "session", "event_index"]).reset_index(drop=True)
        weights = bootstrap_weights(cohort)
        rat_masks = {rat: cohort.rat.eq(rat).to_numpy() for rat in sorted(cohort.rat.unique())}
        for (generator, replicate, variant), group in phase_events.groupby(["generator", "replicate", "encoding_variant"], sort=True):
            ordered = cohort.merge(group, on=["session", "rat", "event_index"], validate="one_to_one")
            if len(ordered) != len(cohort):
                raise ValueError("missing simulated cohort members")
            values = ordered[list(CONTRASTS)].to_numpy()
            boot = weights @ values
            rat_means = np.stack([values[mask].mean(axis=0) for mask in rat_masks.values()])
            for j, contrast in enumerate(CONTRASTS):
                ci = np.quantile(boot[:, j], [0.025, 0.975])
                records.append(
                    {
                        "phase": phase,
                        "generator": generator,
                        "replicate": replicate,
                        "encoding_variant": variant,
                        "contrast": contrast,
                        "equal_animal_mean": float(rat_means[:, j].mean()),
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                        "positive_animals": int((rat_means[:, j] > 0).sum()),
                        "n_animals": len(rat_masks),
                        "events": len(cohort),
                    }
                )
                for rat, mean in zip(rat_masks, rat_means[:, j], strict=True):
                    animals.append(
                        {
                            "phase": phase,
                            "generator": generator,
                            "replicate": replicate,
                            "encoding_variant": variant,
                            "contrast": contrast,
                            "rat": rat,
                            "mean_event_median_delta": mean,
                        }
                    )
    return pd.DataFrame(records), pd.DataFrame(animals)


def binomial_interval(k, n):
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial denominator")
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def summarize(panels, real_summary):
    summaries, patterns = [], []
    keys = ["phase", "generator", "encoding_variant"]
    for key, group in panels.groupby(keys, sort=True):
        primary = group[group.contrast.isin(["imm_minus_iid_position", "imm_minus_static_location"])].copy()
        passed = (primary.ci_low > 0) & (primary.positive_animals == primary.n_animals) & (primary.equal_animal_mean > 0)
        flag = primary.assign(passed=passed).groupby("replicate").passed.all()
        if len(flag) != REPLICATES or primary.groupby("replicate").size().ne(2).any():
            raise ValueError("incomplete primary patterns")
        lo, hi = binomial_interval(int(flag.sum()), len(flag))
        patterns.append(
            dict(zip(keys, key, strict=True))
            | {"positive_pattern_count": int(flag.sum()), "replicates": len(flag), "positive_pattern_fraction": float(flag.mean()), "binomial_ci_low": lo, "binomial_ci_high": hi}
        )
        for contrast, values in group.groupby("contrast", sort=True):
            real = real_summary[
                (real_summary.phase == key[0]) & (real_summary.encoding_variant == key[2]) & (real_summary.inference_temperature == 1) & (real_summary.contrast == contrast)
            ]
            real_value = float(real.equal_animal_mean_event_median_delta.iloc[0]) if len(real) == 1 else np.nan
            estimate = values.equal_animal_mean
            summaries.append(
                dict(zip(keys, key, strict=True))
                | {
                    "contrast": contrast,
                    "replicates": len(values),
                    "median_simulated_effect": estimate.median(),
                    "p05_simulated_effect": estimate.quantile(0.05),
                    "p95_simulated_effect": estimate.quantile(0.95),
                    "positive_mean_fraction": float((estimate > 0).mean()),
                    "four_positive_animals_fraction": float((values.positive_animals == values.n_animals).mean()),
                    "positive_lower_ci_fraction": float((values.ci_low > 0).mean()),
                    "real_effect": real_value,
                    "real_below_simulation_one_sided_mc_p": (1 + int((estimate <= real_value).sum())) / (1 + len(values)) if np.isfinite(real_value) else np.nan,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(patterns)


def gates(scores, selection):
    keys = ["session", "phase", "event_index", "generator", "replicate", "split", "encoding_variant"]
    expected = 320 * len(GENERATORS) * REPLICATES * frozen.N_SPLITS * len(VARIANTS)
    observed = scores.groupby(["session", "phase", "event_index"]).size()
    scientific = [f"score_{m}" for m in MODELS] + ["score_oracle"]
    checks = {
        "rows_complete_nonempty": len(scores) == expected and expected > 0,
        "unique_conditions": not scores.duplicated(keys).any(),
        "all_templates_complete": len(observed) == 320 and observed.eq(expected // 320).all(),
        "all_expected_factors": set(scores.generator) == set(GENERATORS)
        and set(scores.replicate) == set(range(REPLICATES))
        and set(scores.split) == set(range(5))
        and set(scores.encoding_variant) == set(VARIANTS),
        "finite_normalized_scores": np.isfinite(scores[scientific]).all().all() and scores[scientific].le(1e-8).all().all(),
        "one_population_draw_per_event": scores.groupby(["session", "phase", "event_index", "generator", "replicate"]).counts_sha256.nunique().eq(1).all(),
        "count_partition_conserved": (scores.n_train_spikes + scores.n_heldout_spikes).eq(scores.n_spikes).all(),
        "all_successful": scores.status.eq("success").all(),
        "exact_source_events": set(scores[["session", "phase", "event_index"]].itertuples(index=False, name=None))
        == set(selection[["session", "phase", "event_id"]].itertuples(index=False, name=None))
        and len(selection) == 320,
    }
    checks["overall"] = all(checks.values())
    return pd.DataFrame([{"gate": k, "passed": bool(v)} for k, v in checks.items()])


def run(args):
    source, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    parent_path = source / "hc11_conditional_manifest.json"
    if file_sha256(parent_path) != PARENT_DIGEST:
        raise ValueError("unexpected parent experiment")
    parent = json.loads(parent_path.read_text())
    audit_path = Path(args.source_audit) / "audit_manifest.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["input_file_sha256"]["run_manifest"] != PARENT_DIGEST:
        raise ValueError("parent audit missing or mismatched")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite an experiment")
    output.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(source / "frozen_selection.csv")
    frozen.validate_selection(selection, full_cohort=True)
    for name in ["frozen_selection.csv", "hc11_conditional_summary.csv"] + [f"{s}_cache.npz" for s in selection.session.unique()]:
        if file_sha256(source / name) != parent["output_sha256"][name]:
            raise ValueError(f"parent artifact changed: {name}")
    manifest = build_script_provenance(input_paths={"parent_manifest": parent_path, "parent_audit": audit_path})
    manifest.update(
        experiment="hc11_count_coherent_conditional_recovery",
        created_at_utc=datetime.now(UTC).isoformat(),
        n_replicates=REPLICATES,
        n_bootstraps=BOOTSTRAPS,
        n_splits=5,
        seed=SEED,
        generators=list(GENERATORS),
        source_dir=str(source),
        status="running",
        claim_boundary="known-map conditional observation capability; not biological power or a changed real-data analysis",
    )
    path = output / "recovery_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    completed = []
    try:
        tasks = [(source, output, s, g, first, min(first + 10, REPLICATES)) for s, g in selection.groupby("session") for first in range(0, REPLICATES, 10)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(shard, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps(result), flush=True)
        scores = pd.concat([pd.read_csv(output / f"{item['tag']}_split_scores.csv") for item in completed], ignore_index=True)
        technical = gates(scores, selection)
        technical.to_csv(output / "recovery_gate_summary.csv", index=False)
        if not technical.passed.all():
            raise ValueError("technical recovery gates failed")
        events = event_contrasts(scores)
        events.to_csv(output / "recovery_event_contrasts.csv", index=False)
        panels, by_animal = simulation_panels(events)
        panels.to_csv(output / "recovery_simulation_panels.csv", index=False)
        by_animal.to_csv(output / "recovery_by_animal.csv", index=False)
        summary, patterns = summarize(panels, pd.read_csv(source / "hc11_conditional_summary.csv"))
        summary.to_csv(output / "recovery_effect_summary.csv", index=False)
        patterns.to_csv(output / "recovery_positive_pattern_summary.csv", index=False)
        confusion = events.groupby(["generator", "phase", "encoding_variant", "raw_predictive_winner"], as_index=False).size()
        confusion["fraction"] = confusion["size"] / confusion.groupby(["generator", "phase", "encoding_variant"])["size"].transform("sum")
        confusion.to_csv(output / "recovery_predictive_rank_confusion.csv", index=False)
        information = scores.groupby(["generator", "phase", "replicate"], as_index=False)[["n_spikes", "n_train_spikes", "n_heldout_spikes", "n_active_units"]].median()
        information.to_csv(output / "recovery_information_summary.csv", index=False)
        manifest.update(status="complete", elapsed_s=time.monotonic() - started)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["shards"] = sorted(completed, key=lambda x: x["tag"])
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p != path}
        path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-20260908"))
    parser.add_argument("--source-audit", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-audit-20260908"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    run(parser.parse_args())
