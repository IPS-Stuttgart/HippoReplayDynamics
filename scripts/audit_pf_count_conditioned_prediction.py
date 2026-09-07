#!/usr/bin/env python3
"""Frozen training-cell inference, proper held-out identity/count prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _provenance import build_script_provenance, file_sha256
from extract_replay_commitment_composition_posteriors import decoder_configs
from test_pfeiffer_train_only_map_specific_mode_prediction import (
    _stable_seed,
    population_code_permuted_encoding,
)

from hipporeplayimm.benchmarks import _split_cells
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import build_emissions, fit_place_field_encoding
from hipporeplayimm.frozen_posterior_prediction import (
    frozen_smoothed_marginal_log_score,
    normalized_log_posterior,
    posterior_sha256,
)
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel

OBSERVATIONS = ("full_poisson", "count_conditioned", "total_rate_only")
MODELS = ("iid_position", "static_location", "stationary", "diffusion", "first_order_imm")
MAPS = ("real", "population_code_permuted")


def likelihood_parts(counts, rates, durations, rate_scale=1.0):
    """Proper count, identity and full probabilities, without temperature."""
    k = np.asarray(counts, dtype=float)
    r = np.asarray(rates, dtype=float)
    dt = np.asarray(durations, dtype=float)
    if k.ndim != 2 or r.ndim != 2 or k.shape[1] != r.shape[0]:
        raise ValueError("counts (time, cell) and rates (cell, position) must align")
    if not k.size or not r.size or not np.isfinite(k).all() or np.any(k < 0) or np.any(k != np.floor(k)):
        raise ValueError("counts must be nonempty finite nonnegative integers")
    if not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("rates must be finite and strictly positive")
    if dt.shape != (len(k),) or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("durations must be finite positive per-time exposures")
    if not np.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError("rate_scale must be positive")
    total = k.sum(axis=1)
    log_r = np.log(r)
    log_sum_r = logsumexp(log_r, axis=0)
    combinatorial = gammaln(total + 1) - gammaln(k + 1).sum(axis=1)
    identity = k @ (log_r - log_sum_r) + combinatorial[:, None]
    log_mu = np.log(dt * rate_scale)[:, None] + log_sum_r
    count = total[:, None] * log_mu - np.exp(log_mu) - gammaln(total + 1)[:, None]
    return {"full_poisson": identity + count, "count_conditioned": identity, "total_rate_only": count}


def analytic_posterior(log_emissions, model):
    ll = np.asarray(log_emissions, dtype=float)
    if model == "iid_position":
        return normalized_log_posterior(ll)
    if model == "static_location":
        row = ll.sum(axis=0, keepdims=True)
        return np.repeat(normalized_log_posterior(row), len(ll), axis=0)
    raise ValueError("unknown analytic model")


def _hash_array(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()


def _task(task):
    enc_cfg, emit_cfg, state_cfg = task["configs"]
    session = load_replay_session(Path(task["dataset_root"]) / task["session"])
    enc = fit_place_field_encoding(session, enc_cfg)
    wrong, wrong_hash = population_code_permuted_encoding(enc, seed=_stable_seed(20260805, task["session"]))
    models = {name: SortedSpikeStateSpaceReplayModel(mode=name.replace("_", "-"), config=replace(state_cfg, mode=name.replace("_", "-"))) for name in MODELS[2:]}
    rows = []
    for split in range(task["n_splits"]):
        train, held = _split_cells(enc.cell_ids, 0.30, 20260804 + split)
        if not len(train) or not len(held) or np.intersect1d(train, held).size:
            raise ValueError("invalid cell split")
        for event in task["events"]:
            for map_name, encoding in zip(MAPS, (enc, wrong), strict=True):
                train_enc = encoding.select_cells(train)
                train_e = build_emissions(session, train_enc, event, replace(emit_cfg, likelihood_temperature=1.0))
                train_parts = likelihood_parts(train_e.spike_counts, train_enc.rates_hz, train_e.bin_durations, emit_cfg.spike_rate_scale)
                train_parts_hash = _hash_array(train_e.spike_counts)
                # Freeze every training posterior before accessing held-out replay counts.
                posteriors = {}
                for temperature in task["temperatures"]:
                    for observation in OBSERVATIONS:
                        ll = train_parts[observation] / temperature
                        for model in MODELS:
                            started = time.monotonic()
                            if model in MODELS[:2]:
                                posterior = analytic_posterior(ll, model)
                            else:
                                score = models[model].score(replace(train_e, log_likelihood=ll), encoding.bin_centers)
                                posterior = normalized_log_posterior(score.trajectory_log_posterior)
                            posteriors[(temperature, observation, model)] = (posterior, posterior_sha256(posterior), time.monotonic() - started)
                held_enc = encoding.select_cells(held)
                held_e = build_emissions(session, held_enc, event, replace(emit_cfg, likelihood_temperature=1.0))
                if not np.array_equal(train_e.times, held_e.times):
                    raise ValueError("train/held-out time misalignment")
                held_parts = likelihood_parts(held_e.spike_counts, held_enc.rates_hz, held_e.bin_durations, emit_cfg.spike_rate_scale)
                for (temperature, observation, model), (posterior, before, runtime) in posteriors.items():
                    identity = frozen_smoothed_marginal_log_score(posterior, held_parts["count_conditioned"])
                    poisson = frozen_smoothed_marginal_log_score(posterior, held_parts["full_poisson"])
                    if before != posterior_sha256(posterior):
                        raise ValueError("held-out evaluation mutated posterior")
                    rows.append(
                        {
                            "session": task["session"],
                            "rat": session.rat,
                            "event_index": event,
                            "split": split,
                            "map": map_name,
                            "observation": observation,
                            "model": model,
                            "inference_temperature": temperature,
                            "heldout_temperature": 1.0,
                            "conditional_heldout_log_score": identity.total_log_score,
                            "poisson_heldout_log_score": poisson.total_log_score,
                            "n_train_cells": len(train),
                            "n_heldout_cells": len(held),
                            "train_cell_ids": ",".join(map(str, train)),
                            "heldout_cell_ids": ",".join(map(str, held)),
                            "n_train_spikes": train_e.n_spikes,
                            "n_heldout_spikes": held_e.n_spikes,
                            "n_time_bins": train_e.n_time,
                            "n_heldout_nonzero_bins": int((held_e.spike_counts.sum(axis=1) > 0).sum()),
                            "training_counts_sha256": train_parts_hash,
                            "posterior_sha256": before,
                            "posterior_unchanged": True,
                            "heldout_used_for_inference": False,
                            "cells_disjoint": True,
                            "wrong_map_permutation_sha256": wrong_hash,
                            "n_spatial_bins": encoding.n_bins,
                            "runtime_s": runtime,
                            "status": "success",
                        }
                    )
                del posteriors
    return rows


def contrasts(scores):
    key = ["session", "rat", "event_index", "split", "inference_temperature"]
    if scores.duplicated(key + ["map", "observation", "model"]).any():
        raise ValueError("duplicate model rows")
    wide = scores.pivot(index=key, columns=["map", "observation", "model"], values="conditional_heldout_log_score")
    rows = []

    def add(name, a, b):
        value = wide[a] - wide[b]
        if not np.isfinite(value).all():
            raise ValueError("missing or nonfinite paired scores")
        out = value.rename("delta").reset_index()
        out["contrast"] = name
        rows.append(out)

    for obs in OBSERVATIONS:
        for baseline in MODELS[:4]:
            add(f"{obs}:imm_minus_{baseline}", ("real", obs, "first_order_imm"), ("real", obs, baseline))
        add(f"{obs}:real_minus_wrong_imm", ("real", obs, "first_order_imm"), ("population_code_permuted", obs, "first_order_imm"))
    for model in MODELS:
        add(f"{model}:identity_minus_rate_inference", ("real", "count_conditioned", model), ("real", "total_rate_only", model))
        add(f"{model}:identity_minus_full_inference", ("real", "count_conditioned", model), ("real", "full_poisson", model))
    split = pd.concat(rows, ignore_index=True)
    events = split.groupby(["session", "rat", "event_index", "inference_temperature", "contrast"], as_index=False)["delta"].median()
    return split, events


def summarize(events, replicates=5000, seed=20260908):
    rng = np.random.default_rng(seed)
    rows, animal_rows = [], []
    for (temp, contrast), group in events.groupby(["inference_temperature", "contrast"], sort=True):
        animals = sorted(group.rat.unique())
        by_animal = group.groupby("rat").delta.mean().reindex(animals)
        for rat, value in by_animal.items():
            animal_rows.append({"contrast": contrast, "inference_temperature": temp, "rat": rat, "mean_event_median_delta": value, "events": int((group.rat == rat).sum())})
        cached = {rat: [part.delta.to_numpy() for _, part in group[group.rat == rat].groupby("session")] for rat in animals}
        boots = []
        for _ in range(replicates):
            sampled = []
            for rat in rng.choice(animals, len(animals), replace=True):
                sessions = cached[rat]
                values = []
                for i in rng.integers(0, len(sessions), len(sessions)):
                    v = sessions[i]
                    values.extend(rng.choice(v, len(v), replace=True))
                sampled.append(np.mean(values))
            boots.append(np.mean(sampled))
        signs = 2 * ((np.arange(2 ** len(animals))[:, None] >> np.arange(len(animals))) & 1) - 1
        sign_means = (signs * by_animal.to_numpy()).mean(axis=1)
        estimate = float(by_animal.mean())
        rows.append(
            {
                "contrast": contrast,
                "inference_temperature": temp,
                "events": len(group),
                "animals": len(animals),
                "equal_animal_mean_event_median_delta": estimate,
                "ci_low": np.quantile(boots, 0.025),
                "ci_high": np.quantile(boots, 0.975),
                "positive_animals": int((by_animal > 0).sum()),
                "rat_signflip_one_sided_p": float(np.mean(sign_means >= estimate - 1e-12)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(animal_rows)


def gates(scores, events, n_splits, temperatures):
    expected = len(events) * n_splits * len(temperatures) * len(MAPS) * len(OBSERVATIONS) * len(MODELS)
    key = ["session", "event_index", "split", "inference_temperature", "map", "observation", "model"]
    rows = []

    def add(name, passed, observed):
        rows.append({"gate": name, "passed": bool(passed), "observed": str(observed)})

    add("nonempty_expected_events", len(events) > 0, len(events))
    if scores.empty:
        add("required_rows_complete", False, f"0/{expected}")
        add("overall", False, "empty score table")
        return pd.DataFrame(rows)
    add("required_rows_complete", len(scores) == expected and expected > 0 and not scores.duplicated(key).any(), f"{len(scores)}/{expected}")
    expected_keys = {
        (sid, int(eid), split, float(temp), map_name, obs, model)
        for sid, eid in events[["session", "event_index"]].itertuples(index=False, name=None)
        for split in range(n_splits)
        for temp in temperatures
        for map_name in MAPS
        for obs in OBSERVATIONS
        for model in MODELS
    }
    add("exact_condition_keys", set(scores[key].itertuples(index=False, name=None)) == expected_keys, len(expected_keys))
    source = set(map(tuple, events[["session", "event_index"]].to_numpy()))
    actual = set(map(tuple, scores[["session", "event_index"]].drop_duplicates().to_numpy())) if len(scores) else set()
    add("exact_event_set", source == actual and bool(source), len(actual))
    finite = np.isfinite(scores[["conditional_heldout_log_score", "poisson_heldout_log_score"]]).all().all() if len(scores) else False
    add("finite_proper_scores", finite and scores.heldout_temperature.eq(1).all(), finite)
    add("cells_disjoint", scores.cells_disjoint.all() and len(scores) > 0, scores.cells_disjoint.sum())
    parsed_disjoint = all(set(str(a).split(",")).isdisjoint(str(b).split(",")) for a, b in scores[["train_cell_ids", "heldout_cell_ids"]].itertuples(index=False, name=None))
    add("cell_id_sets_verified_disjoint", parsed_disjoint, parsed_disjoint)
    add("heldout_never_used_for_inference", (~scores.heldout_used_for_inference).all() and len(scores) > 0, scores.heldout_used_for_inference.sum())
    add("posteriors_unchanged", scores.posterior_unchanged.all() and len(scores) > 0, scores.posterior_unchanged.sum())
    analytic = scores[scores.model.isin(MODELS[:2])]
    pivot = analytic.pivot(index=[c for c in key if c != "map"], columns="map", values="conditional_heldout_log_score")
    error = float(np.max(np.abs(pivot[MAPS[0]] - pivot[MAPS[1]]))) if len(pivot) else np.inf
    add("analytic_baselines_map_permutation_invariant", error < 1e-7, error)
    add("overall", all(row["passed"] for row in rows), "technical only")
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--event-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-events-per-session", type=int, default=0)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.3])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--reuse-scores")
    args = parser.parse_args(argv)
    if args.n_splits < 1 or args.workers < 1 or args.bootstrap_replicates < 1 or args.max_events_per_session < 0:
        parser.error("counts must be positive (event cap may be zero)")
    if not all(np.isfinite(x) and x > 0 for x in args.temperatures) or len(set(args.temperatures)) != len(args.temperatures):
        parser.error("temperatures must be distinct finite positive values")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new/empty; preserve existing runs")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    evidence = pd.read_csv(args.event_evidence)
    configs = decoder_configs(evidence)
    enc, emit, state = configs
    if emit.cell_weights is not None or emit.negative_binomial_overdispersion != 0:
        raise ValueError("Poisson factorization requires unweighted independent Poisson counts")
    events = evidence[["session", "event_index"]].drop_duplicates().sort_values(["session", "event_index"])
    if args.max_events_per_session:
        events = events.groupby("session", sort=True).head(args.max_events_per_session)
    if events.empty:
        raise ValueError("empty event set")
    events.to_csv(output / "frozen_event_set.csv", index=False)
    input_paths = {"event_evidence": args.event_evidence, "protocol": ROOT / "docs/pf_count_conditioned_prediction.md"}
    if args.reuse_scores:
        input_paths["reused_scores"] = args.reuse_scores
    provenance = build_script_provenance(input_paths=input_paths, cwd=ROOT)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "arguments": vars(args),
        "encoding_config": asdict(enc),
        "emission_source_config": asdict(emit),
        "state_config": asdict(state),
        "heldout_temperature": 1.0,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": file_sha256(Path(__file__)),
        "events": len(events),
        "status": "running",
    }
    raw_files = sorted({file for sid in events.session for file in (Path(args.dataset_root) / sid).rglob("*.mat")})
    manifest["source_mat_sha256"] = {str(path): file_sha256(path) for path in raw_files}
    if not raw_files:
        raise ValueError("no raw MAT files found for source provenance")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if args.reuse_scores:
        scores = pd.read_csv(args.reuse_scores)
    else:
        tasks = [
            {
                "session": session,
                "events": group.event_index.astype(int).tolist(),
                "dataset_root": args.dataset_root,
                "configs": configs,
                "n_splits": args.n_splits,
                "temperatures": args.temperatures,
            }
            for session, group in events.groupby("session")
        ]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_task, task): task["session"] for task in tasks}
            for future in as_completed(futures):
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(output / "partial_scores.csv", index=False)
                print(f"completed {futures[future]}: {len(rows)} rows", flush=True)
        scores = pd.DataFrame(rows)
    scores = scores.sort_values(["session", "event_index", "split", "map", "observation", "model", "inference_temperature"])
    scores.to_csv(output / "split_scores.csv", index=False)
    gate = gates(scores, events, args.n_splits, args.temperatures)
    gate.to_csv(output / "gate_summary.csv", index=False)
    if not gate.passed.all():
        manifest.update(status="technical_fail", runtime_s=time.monotonic() - started)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        raise ValueError("technical gates failed; do not interpret")
    split, event = contrasts(scores)
    split.to_csv(output / "split_contrasts.csv", index=False)
    event.to_csv(output / "event_medians.csv", index=False)
    summary, animals = summarize(event, args.bootstrap_replicates)
    summary.to_csv(output / "contrast_summary.csv", index=False)
    animals.to_csv(output / "by_animal.csv", index=False)
    selected = summary[
        summary.inference_temperature.eq(1)
        & summary.contrast.isin(
            [
                "count_conditioned:imm_minus_iid_position",
                "count_conditioned:imm_minus_diffusion",
                "count_conditioned:imm_minus_static_location",
                "count_conditioned:real_minus_wrong_imm",
                "first_order_imm:identity_minus_rate_inference",
            ]
        )
    ]
    report = [
        "# PF conditional identity prediction",
        "",
        "Technical pass; exploratory, not external replication.",
        "",
        "Primary target: held-out cell identities conditional on observed population total.",
        "Inference uses training cells only. Held-out probabilities are untempered.",
        "",
        "```text",
        selected.to_string(index=False),
        "```",
        "",
        "Only four animals: the minimum exact one-sided animal sign-flip p is 0.0625.",
        "Conditioning removes the total-rate term, not differences in information from spike count.",
        "The historical all-cell selected event cohort is reused; this is not a new dataset.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    manifest.update(
        status="completed", runtime_s=time.monotonic() - started, outputs_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p != manifest_path}
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(selected.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
