#!/usr/bin/env python3
"""External, non-winner-selected replication of conditional cross-cell prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import _hc11_native_encoding as native
import audit_pf_count_conditioned_prediction as pf
from _provenance import build_script_provenance, file_sha256

import hipporeplayimm.state_space as ss
from hipporeplayimm.benchmarks import _split_cells
from hipporeplayimm.duration_occupancy import _forward_backward_variable
from hipporeplayimm.frozen_posterior_prediction import (
    frozen_smoothed_marginal_log_score,
    posterior_sha256,
)

MODELS = ("iid_position", "static_location", "diffusion", "first_order_imm")
VARIANTS = ("pooled", "direction_mixture")
MAPS = ("real", "population_code_permuted")
TEMPERATURES = (1.0, 0.3)
OBSERVATIONS = pf.OBSERVATIONS
ENCODING_PARAMETERS = {
    "position_bin_size_cm": 4.0,
    "min_run_speed_cm_s": 5.0,
    "min_run_spikes": 20,
    "min_spatial_information": 0.1,
    "min_peak_rate_hz": 1.0,
    "min_encoding_units": 5,
    "smoothing_sigma_bins": 1.5,
}
N_SPLITS = 5
SPLIT_SEED = 20260804
MAP_SEED = 20260908
DEFAULT_SOURCE = Path("/home/florianpfaff/HippoReplayIMM-hc11-pre-post-learning-run/results/hc11-pre-post-paper-events-primary8-p20-pre-rate")
DEFAULT_DATASET = Path("/mnt/lexar4tb/datasets/hc11_grosmark_buzsaki/webshare_processed")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def event_edges(start, end, dt=0.02):
    if not np.isfinite([start, end, dt]).all() or end <= start or dt <= 0:
        raise ValueError("invalid event interval")
    n = int(np.floor((end - start) / dt))
    edges = start + np.arange(n + 1) * dt
    tolerance = 16 * np.finfo(float).eps * max(1.0, abs(start), abs(end))
    if end - edges[-1] > tolerance:
        edges = np.append(edges, end)
    else:
        edges[-1] = end
    if len(edges) == 1:
        edges = np.array([start, end])
    return edges


def count_spikes(spikes, units, edges):
    """Half-open bins including neither an adjacent event nor its end spike."""
    return np.column_stack([np.diff(np.searchsorted(spikes.times_by_unit[int(unit)], edges, side="left")) for unit in units])


def transitions(centers, edges, topology, track_length):
    kwargs = {"topology": topology, "track_length_cm": track_length}
    stationary = native.topology_gaussian_transition(centers, 2.0, 4.0, **kwargs)
    durations = np.diff(0.5 * (edges[:-1] + edges[1:]))
    diff = [native.topology_gaussian_transition(centers, 85.0 * np.sqrt(dt), 4.0, **kwargs) for dt in durations]
    n = len(centers)
    iid = csr_matrix(np.full((n, n), 1.0 / n))
    modes = np.full((3, 3), 0.025)
    np.fill_diagonal(modes, 0.95)
    # Rows are destination (mode, position); columns are source (mode, position).
    imm = [bmat([[modes[src, dst] * kernel for src in range(3)] for dst, kernel in enumerate((stationary, d, iid))], format="csr") for d in diff]
    return {"diffusion": diff, "first_order_imm": imm}


def infer_one(log_emissions, model, kernels):
    ll = np.asarray(log_emissions, dtype=float)
    n = ll.shape[1]
    if model == "iid_position":
        return float(np.sum(logsumexp(ll, axis=1) - np.log(n))), pf.analytic_posterior(ll, model)
    if model == "static_location":
        return float(logsumexp(ll.sum(axis=0)) - np.log(n)), pf.analytic_posterior(ll, model)
    if model == "diffusion":
        return _forward_backward_variable(ss, ll, kernels[model])
    if model != "first_order_imm":
        raise ValueError("unknown model")
    logz, logpost = _forward_backward_variable(ss, np.tile(ll, (1, 3)), kernels[model])
    return logz, logsumexp(logpost.reshape(len(ll), 3, n), axis=1)


def infer_training(train_parts, observation, temperature, model, kernels):
    """Marginalize the global direction using training data alone."""
    inferred = [infer_one(parts[observation] / temperature, model, kernels) for parts in train_parts]
    logz = np.array([value[0] for value in inferred])
    logweights = logz - logsumexp(logz)
    weighted = np.stack([value[1] + weight for value, weight in zip(inferred, logweights, strict=True)])
    return weighted.transpose(1, 0, 2).reshape(weighted.shape[1], -1)


def heldout_score(frozen, parts, observation):
    ll = np.stack([part[observation] for part in parts]).transpose(1, 0, 2)
    return frozen_smoothed_marginal_log_score(frozen, ll.reshape(len(ll), -1)).total_log_score


def validate_selection(selection, full_cohort=False):
    keys = ["session", "phase", "event_id"]
    required = keys + ["animal", "start_time_s", "end_time_s", "match_pair_id", "scoring_time_bin_s", "scoring_event_padding_s"]
    if not set(required).issubset(selection) or selection.empty or selection[required].isna().any().any():
        raise ValueError("missing/nonfinite selection schema or empty cohort")
    if selection.duplicated(keys).any() or selection.duplicated(["session", "phase", "match_pair_id"]).any():
        raise ValueError("duplicate selected events or matches")
    if set(selection.phase) != {"PRE", "POST"}:
        raise ValueError("PRE and POST must both be retained")
    for _, group in selection.groupby("session"):
        if set(group[group.phase == "PRE"].match_pair_id) != set(group[group.phase == "POST"].match_pair_id):
            raise ValueError("PRE/POST pairs not matched")
        if group.animal.nunique() != 1:
            raise ValueError("inconsistent animal mapping")
    values = selection[["start_time_s", "end_time_s", "scoring_time_bin_s", "scoring_event_padding_s"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:, 1] <= values[:, 0]) or not np.allclose(values[:, 2], 0.02) or np.any(values[:, 3] != 0):
        raise ValueError("event times or frozen bins/padding mismatch")
    if full_cohort and (
        len(selection) != 320 or selection.session.nunique() != 8 or selection.animal.nunique() != 4 or not selection.groupby(["session", "phase"]).size().eq(20).all()
    ):
        raise ValueError("frozen all-eight-session, four-animal cohort is incomplete")


def score_session(task):
    session, selected, dataset_root, output = task
    folder = list(Path(dataset_root).glob(f"*/{session}"))
    if len(folder) != 1:
        raise ValueError(f"ambiguous/missing {session}")
    folder = folder[0]
    track = native.load_track_samples(folder)
    spikes = native.load_spikes(folder)
    maps, unit_qc = native.build_session_encodings(track, spikes, **ENCODING_PARAMETERS)
    pooled = maps["pooled"][0]
    units = np.asarray(pooled.unit_ids)
    permutation = np.random.default_rng(stable_seed(MAP_SEED, session)).permutation(len(pooled.bin_centers_cm))
    cache = {
        "unit_ids": units,
        "centers": pooled.bin_centers_cm,
        "edges": pooled.bin_edges_cm,
        "occupancy": pooled.occupancy_s,
        "permutation": permutation,
        "topology": np.array(track.topology),
        "track_length": np.array(track.track_length_cm),
    }
    for variant in VARIANTS:
        for d, encoding in enumerate(maps[variant]):
            if not np.array_equal(units, encoding.unit_ids):
                raise ValueError("direction unit order mismatch")
            cache[f"rates_{variant}_{d}"] = encoding.rates_hz
    source_hashes = {str(p): file_sha256(p) for p in (folder / f"{session}.spikes.cellinfo.mat", folder / f"{session}.position.behavior.mat")}
    rows = []
    for split in range(N_SPLITS):
        train_ids, held_ids = _split_cells(units, 0.30, SPLIT_SEED + split)
        train = np.flatnonzero(np.isin(units, train_ids))
        held = np.flatnonzero(np.isin(units, held_ids))
        if not len(train) or not len(held) or np.intersect1d(train, held).size:
            raise ValueError("invalid neural split")
        cache[f"train_{split}"] = train
        cache[f"held_{split}"] = held
        for event in selected.itertuples(index=False):
            uid = f"{event.phase}_{event.event_id}"
            edges = event_edges(float(event.start_time_s), float(event.end_time_s))
            counts = count_spikes(spikes, tuple(units), edges)
            if split == 0:
                cache[f"counts_{uid}"] = counts
                cache[f"edges_{uid}"] = edges
            kernels = transitions(pooled.bin_centers_cm, edges, track.topology, track.track_length_cm)
            for variant in VARIANTS:
                for map_name in MAPS:
                    order = np.arange(len(permutation)) if map_name == "real" else permutation
                    rates = [encoding.rates_hz[:, order] for encoding in maps[variant]]
                    train_parts = [pf.likelihood_parts(counts[:, train], r[train], np.diff(edges)) for r in rates]
                    frozen = {}
                    for temp in TEMPERATURES:
                        for observation in OBSERVATIONS:
                            for model in MODELS:
                                started = time.monotonic()
                                posterior = infer_training(train_parts, observation, temp, model, kernels)
                                frozen[(temp, observation, model)] = (posterior, posterior_sha256(posterior), time.monotonic() - started)
                    # Held-out likelihoods cannot influence inference or direction weighting.
                    held_parts = [pf.likelihood_parts(counts[:, held], r[held], np.diff(edges)) for r in rates]
                    for (temp, observation, model), (posterior, before, runtime) in frozen.items():
                        identity = heldout_score(posterior, held_parts, "count_conditioned")
                        poisson = heldout_score(posterior, held_parts, "full_poisson")
                        if before != posterior_sha256(posterior):
                            raise ValueError("posterior changed during prediction")
                        rows.append(
                            {
                                "session": session,
                                "rat": folder.parent.name,
                                "phase": event.phase,
                                "event_index": int(event.event_id),
                                "match_pair_id": int(event.match_pair_id),
                                "split": split,
                                "encoding_variant": variant,
                                "map": map_name,
                                "observation": observation,
                                "model": model,
                                "inference_temperature": temp,
                                "heldout_temperature": 1.0,
                                "conditional_heldout_log_score": identity,
                                "poisson_heldout_log_score": poisson,
                                "n_train_spikes": int(counts[:, train].sum()),
                                "n_heldout_spikes": int(counts[:, held].sum()),
                                "n_train_cells": len(train),
                                "n_heldout_cells": len(held),
                                "n_time_bins": len(counts),
                                "duration_s": float(edges[-1] - edges[0]),
                                "geometry": track.topology,
                                "n_spatial_bins": len(pooled.bin_centers_cm),
                                "n_total_spikes": int(counts.sum()),
                                "original_selection_spikes": int(event.n_spikes),
                                "train_cell_ids": ",".join(map(str, units[train])),
                                "heldout_cell_ids": ",".join(map(str, units[held])),
                                "posterior_sha256": before,
                                "posterior_unchanged": True,
                                "heldout_used_for_inference": False,
                                "runtime_s": runtime,
                                "status": "success",
                            }
                        )
    np.savez_compressed(Path(output) / f"{session}_cache.npz", **cache)
    table = pd.DataFrame(rows)
    table.to_csv(Path(output) / f"{session}_scores.csv", index=False)
    unit_qc.to_csv(Path(output) / f"{session}_unit_qc.csv", index=False)
    return {
        "session": session,
        "animal": folder.parent.name,
        "rows": len(table),
        "source_hashes": source_hashes,
        "n_units": len(units),
        "n_bins": len(pooled.bin_centers_cm),
        "topology": track.topology,
    }


def contrasts(scores):
    keys = ["session", "rat", "phase", "event_index", "match_pair_id", "split", "encoding_variant", "inference_temperature"]
    if scores.duplicated(keys + ["map", "observation", "model"]).any():
        raise ValueError("duplicate scores")
    wide = scores.pivot(index=keys, columns=["map", "observation", "model"], values="conditional_heldout_log_score")
    rows = []

    def real(model, obs="count_conditioned"):
        return ("real", obs, model)

    pairs = [(f"imm_minus_{b}", real("first_order_imm"), real(b)) for b in MODELS[:-1]]
    pairs += [
        ("real_minus_wrong_imm", real("first_order_imm"), ("population_code_permuted", "count_conditioned", "first_order_imm")),
        ("identity_minus_rate_imm", real("first_order_imm"), real("first_order_imm", "total_rate_only")),
        ("identity_minus_full_imm", real("first_order_imm"), real("first_order_imm", "full_poisson")),
        ("diffusion_minus_iid", real("diffusion"), real("iid_position")),
        ("static_minus_iid", real("static_location"), real("iid_position")),
    ]
    for name, a, b in pairs:
        delta = wide[a] - wide[b]
        if not np.isfinite(delta).all():
            raise ValueError("missing paired models")
        part = delta.rename("delta").reset_index()
        part["contrast"] = name
        rows.append(part)
    split = pd.concat(rows, ignore_index=True)
    held = scores.groupby(keys, as_index=False).n_heldout_spikes.first()
    split = split.merge(held, on=keys, validate="many_to_one")
    split["delta_per_heldout_spike"] = split.delta / split.n_heldout_spikes.replace(0, np.nan)
    event_keys = [k for k in keys if k != "split"] + ["contrast"]
    events = split.groupby(event_keys, as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    return split, events


def gate_summary(scores, selection):
    rows = []

    def add(name, passed, observed):
        rows.append({"gate": name, "passed": bool(passed), "observed": str(observed)})

    required_count = len(selection) * N_SPLITS * len(VARIANTS) * len(MAPS) * len(OBSERVATIONS) * len(MODELS) * len(TEMPERATURES)
    keys = ["session", "phase", "event_index", "split", "encoding_variant", "map", "observation", "model", "inference_temperature"]
    add("nonempty_selection", len(selection) > 0, len(selection))
    add("required_rows_complete", len(scores) == required_count and required_count > 0 and not scores.duplicated(keys).any(), f"{len(scores)}/{required_count}")
    expected = set(selection[["session", "phase", "event_id"]].itertuples(index=False, name=None))
    actual = set(scores[["session", "phase", "event_index"]].itertuples(index=False, name=None)) if len(scores) else set()
    add("selected_events_exact", actual == expected and bool(expected), len(actual))
    combinations = {(a, b, c, d, e) for a in VARIANTS for b in MAPS for c in OBSERVATIONS for d in MODELS for e in TEMPERATURES}
    complete = True
    for _, group in scores.groupby(["session", "phase", "event_index"]):
        for split in range(N_SPLITS):
            seen = set(group[group.split == split][["encoding_variant", "map", "observation", "model", "inference_temperature"]].itertuples(index=False, name=None))
            complete &= seen == combinations
    add("every_split_condition_complete", complete and bool(actual), len(combinations))
    add(
        "finite_normalized_scores",
        len(scores) > 0
        and np.isfinite(scores[["conditional_heldout_log_score", "poisson_heldout_log_score"]]).all().all()
        and scores.conditional_heldout_log_score.le(1e-9).all()
        and scores.heldout_temperature.eq(1).all(),
        len(scores),
    )
    add(
        "no_failures_or_posterior_updates",
        len(scores) > 0 and scores.status.eq("success").all() and scores.posterior_unchanged.eq(True).all() and scores.heldout_used_for_inference.eq(False).all(),
        len(scores),
    )
    disjoint = all(
        not (set(str(a).split(",")) & set(str(b).split(","))) for a, b in scores[["train_cell_ids", "heldout_cell_ids"]].drop_duplicates().itertuples(index=False, name=None)
    )
    add("cells_disjoint", disjoint and bool(actual), len(scores))
    add("overall", all(r["passed"] for r in rows), "technical only")
    return pd.DataFrame(rows)


def write_summaries(scores, output):
    split, events = contrasts(scores)
    split.to_csv(output / "hc11_conditional_split_contrasts.csv", index=False)
    events.to_csv(output / "hc11_conditional_event_contrasts.csv", index=False)
    paired = events.pivot(index=["session", "rat", "match_pair_id", "encoding_variant", "inference_temperature", "contrast"], columns="phase", values="delta").reset_index()
    paired["delta"] = paired.POST - paired.PRE
    paired["event_index"] = paired.match_pair_id
    paired["phase"] = "POST_minus_PRE"
    summaries, animals = [], []
    for (variant, phase), group in pd.concat([events, paired], ignore_index=True).groupby(["encoding_variant", "phase"]):
        summary, animal = pf.summarize(group, replicates=5000, seed=20260908)
        for table in (summary, animal):
            table["encoding_variant"] = variant
            table["phase"] = phase
        summaries.append(summary)
        animals.append(animal)
    summary = pd.concat(summaries, ignore_index=True)
    pd.concat(animals, ignore_index=True).to_csv(output / "hc11_conditional_by_animal.csv", index=False)
    events.groupby(["session", "rat", "phase", "encoding_variant", "inference_temperature", "contrast"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean().to_csv(
        output / "hc11_conditional_by_session.csv", index=False
    )
    summary.to_csv(output / "hc11_conditional_summary.csv", index=False)
    return summary


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite an existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    selection_path = Path(args.selection_csv).resolve()
    selection = pd.read_csv(selection_path)
    validate_selection(selection, full_cohort=True)
    source_manifest = json.loads(Path(args.source_manifest).read_text())
    if any(source_manifest["parameters"].get(k) != v for k, v in ENCODING_PARAMETERS.items()):
        raise ValueError("source encoding parameters differ from frozen replication")
    if set(selection.session) != {s["session"] for s in source_manifest["sessions"]}:
        raise ValueError("source manifest session mismatch")
    manifest = build_script_provenance(input_paths={"selection": selection_path, "source_manifest": args.source_manifest})
    manifest["created_at_utc"] = datetime.now(UTC).isoformat()
    manifest.update(
        experiment="hc11_conditional_cross_cell_prediction",
        primary="POST; direction_mixture; T=1; held-out identities given totals",
        encoding_parameters=ENCODING_PARAMETERS,
        heldout_fraction=0.30,
        n_splits=N_SPLITS,
        split_seed=SPLIT_SEED,
        map_seed=MAP_SEED,
        time_bin_s=0.02,
        diffusion_sigma_cm_sqrt_s=85.0,
        stationary_sigma_cm=2.0,
        max_step_sigma=4.0,
        mode_stickiness=0.95,
        rate_scale=1.0,
        claim_boundary="previously inspected cohort; external method replication, not a novel mechanism; PRE and pooled maps are declared sensitivities",
        prospective_protocol="docs/hc11_count_conditioned_prediction_protocol.md",
        status="running",
    )
    manifest_path = output / "hc11_conditional_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    selection.to_csv(output / "frozen_selection.csv", index=False)
    started = time.monotonic()
    tasks = [(name, group, args.dataset_root, output) for name, group in selection.groupby("session")]
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(score_session, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps(result), flush=True)
        scores = pd.concat([pd.read_csv(output / f"{name}_scores.csv") for name in sorted(selection.session.unique())], ignore_index=True)
        gates = gate_summary(scores, selection)
        gates.to_csv(output / "hc11_conditional_gate_summary.csv", index=False)
        if not gates.passed.all():
            raise ValueError("technical gates fail; do not interpret contrasts")
        scores.to_csv(output / "hc11_conditional_event_scores.csv", index=False)
        summary = write_summaries(scores, output)
        primary = summary[(summary.phase == "POST") & (summary.encoding_variant == "direction_mixture") & (summary.inference_temperature == 1)]
        (output / "hc11_conditional_summary.md").write_text(
            "# hc-11 conditional cross-cell prediction\n\nTechnical scoring complete. All eight sessions and four animals retained.\n\nPrimary POST / direction-mixture / T=1:\n\n```text\n"
            + primary.to_string(index=False)
            + "\n```\n\nThese are normalized per-bin held-out log scores, not Bayes factors, joint path evidence, replay classes, or future-time prediction. Event medians across five splits precede equal-animal means. Pointwise bootstrap intervals are exploratory; four animals have minimum exact one-sided sign-flip p=0.0625. Compare PRE, pooled maps, and inference temperature without selecting favorable settings. No negative pilot is overwritten.\n"
        )
        manifest.update(status="complete", sessions=completed, elapsed_s=time.monotonic() - started)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), sessions=completed)
        raise
    finally:
        manifest["output_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file() and p != manifest_path}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SOURCE / "hc11_pre_post_event_selection.csv")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE / "hc11_pre_post_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
