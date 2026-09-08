#!/usr/bin/env python3
"""Raw-count and independent predictive verification of hc-11 MAZE transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_hc11_native_maze_transfer as producer
from _provenance import build_script_provenance, file_sha256
from verify_hc11_count_conditioned_prediction import dense_kernels, direct_parts, direct_predict


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def independent_cap(counts, target, seed):
    if counts.sum() <= target:
        return counts.copy()
    return np.random.default_rng(seed).multivariate_hypergeometric(counts.reshape(-1), target).reshape(counts.shape)


def known_scores(counts, edges, rates, pooled, occupancy, held, position, direction, boundaries, permutation):
    bins = np.minimum(np.maximum(np.digitize(position, boundaries) - 1, 0), len(boundaries) - 2)
    dd = (direction > 0).astype(int) if len(rates) == 2 else np.zeros(len(bins), int)
    parts = [direct_parts(counts[:, held], rate[held], np.diff(edges))["count_conditioned"] for rate in rates]
    score = sum(parts[d][t, b] for t, (d, b) in enumerate(zip(dd, bins, strict=True)))
    wrong = sum(parts[d][t, permutation[b]] for t, (d, b) in enumerate(zip(dd, bins, strict=True)))
    marginal = np.average(pooled, axis=1, weights=occupancy)[:, None]
    baseline = direct_parts(counts[:, held], marginal[held], np.diff(edges))["count_conditioned"].sum()
    return {"score_behavior": score, "score_behavior_wrong": wrong, "score_global": baseline}


def iid_position_posterior(parts):
    emissions = [p["count_conditioned"] for p in parts]
    evidence = np.array([(logsumexp(ll, axis=1) - np.log(ll.shape[1])).sum() for ll in emissions])
    weights = evidence - logsumexp(evidence)
    return np.exp(logsumexp(np.stack([ll - logsumexp(ll, axis=1, keepdims=True) + w for ll, w in zip(emissions, weights, strict=True)]), axis=0))


def verify_map_error(q, centers, truth, topology, length, reported):
    distances = np.abs(centers[None, :] - truth[:, None])
    if topology == "circular":
        distances = np.minimum(distances, length - distances)
    tied = np.isclose(q, q.max(axis=1, keepdims=True), rtol=1e-12, atol=1e-14)
    lower = np.median(np.where(tied, distances, np.inf).min(axis=1))
    upper = np.median(np.where(tied, distances, -np.inf).max(axis=1))
    if not lower - 1e-8 <= reported <= upper + 1e-8:
        raise ValueError("MAP error incompatible with independent posterior")
    return int((tied.sum(axis=1) > 1).sum())


def verify_aggregates(scores, root):
    keys = ["session", "rat", "fold", "window_id", "unit_regime", "count_regime", "encoding_variant"]
    parts = []
    for contrast, (a, b) in producer.CONTRASTS.items():
        frame = scores[keys + ["split"]].copy()
        frame["contrast"] = contrast
        frame["delta"] = scores[f"score_{a}"] - scores[f"score_{b}"]
        frame["normalized"] = frame.delta / scores.n_heldout_spikes.replace(0, np.nan)
        parts.append(frame)
    paired = pd.concat(parts, ignore_index=True)
    stored_paired = pd.read_csv(root / "maze_transfer_split_contrasts.csv")
    join = paired.merge(stored_paired, on=keys + ["split", "contrast"], validate="one_to_one", suffixes=("_check", ""))
    if (
        len(join) != len(paired)
        or len(join) != len(stored_paired)
        or not np.allclose(join.delta_check, join.delta)
        or not np.allclose(join.normalized, join.delta_per_heldout_spike, equal_nan=True)
    ):
        raise ValueError("split contrast discrepancy")
    event = paired.groupby(keys + ["contrast"], as_index=False)[["delta", "normalized"]].median()
    stored = pd.read_csv(root / "maze_transfer_window_contrasts.csv")
    join = event.merge(stored, on=keys + ["contrast"], validate="one_to_one", suffixes=("_check", ""))
    if (
        len(join) != len(stored)
        or len(join) != len(event)
        or not np.allclose(join.delta_check, join.delta)
        or not np.allclose(join.normalized, join.delta_per_heldout_spike, equal_nan=True)
    ):
        raise ValueError("window contrast discrepancy")
    summary = pd.read_csv(root / "maze_transfer_summary.csv")
    conditions = ["unit_regime", "count_regime", "encoding_variant", "contrast"]
    rows = []
    max_error = 0.0
    for key, values in event.groupby(conditions):
        rats, normalized = [], []
        for _, animal in values.groupby("rat", sort=True):
            sessions = []
            for _, session in animal.groupby("session", sort=True):
                folds = [group[["delta", "normalized"]].mean().to_numpy() for _, group in session.groupby("fold", sort=True)]
                if len(folds) != 2:
                    raise ValueError("missing cross-time fold")
                sessions.append(np.mean(folds, axis=0))
            average = np.mean(sessions, axis=0)
            rats.append(average[0])
            normalized.append(average[1])
        rats = np.asarray(rats)
        if len(rats) != 4:
            raise ValueError("missing animal")
        rng = np.random.default_rng(20260908)
        boot = np.array([np.mean(rats[rng.integers(0, 4, 4)]) for _ in range(5000)])
        low, high = np.quantile(boot, [0.025, 0.975])
        mask = np.ones(len(summary), bool)
        for name, value in zip(conditions, key, strict=True):
            mask &= summary[name].eq(value)
        record = summary[mask]
        if len(record) != 1:
            raise ValueError("missing/duplicate summary condition")
        record = record.iloc[0]
        err = max(abs(record.equal_animal_mean - rats.mean()), abs(record.ci_low - low), abs(record.ci_high - high), abs(record.mean_per_heldout_spike - np.mean(normalized)))
        max_error = max(max_error, err)
        if err > 1e-8 or record.positive_animals != (rats > 0).sum():
            raise ValueError("independent aggregate/CI mismatch")
        rows.append(dict(zip(conditions, key, strict=True)) | {"equal_animal_mean": rats.mean(), "ci_low": low, "ci_high": high, "error": err})
    if len(rows) != 56 or len(summary) != 56:
        raise ValueError("summary factors incomplete")
    return pd.DataFrame(rows), {"split_contrasts": len(paired), "window_contrasts": len(event), "aggregate_intervals": len(rows), "max_aggregate_error": max_error}


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest_path = root / "maze_transfer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["git_dirty"] or len(manifest["completed_folds"]) != 16:
        raise ValueError("incomplete or untraceable experiment")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed output: {name}")
    for path, digest in manifest["raw_file_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("changed native raw source")
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError("changed parent source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite verification")
    output.mkdir(parents=True, exist_ok=True)
    source = Path(manifest["source_dir"])
    parent = json.loads((source / "hc11_conditional_manifest.json").read_text())
    for name in ["frozen_selection.csv"] + [f"{item['session']}_cache.npz" for item in parent["sessions"]]:
        if file_sha256(source / name) != parent["output_sha256"][name]:
            raise ValueError("parent cache changed")
    all_scores, checks, predictions = [], [], []
    for item in manifest["completed_folds"]:
        session, fold, tag = item["session"], item["fold"], item["tag"]
        spike_path = next(Path(p) for p in manifest["raw_file_sha256"] if Path(p).name == f"{session}.spikes.cellinfo.mat")
        raw = loadmat(spike_path, squeeze_me=True, struct_as_record=False)["spikes"]
        by_unit = {int(uid): np.sort(np.asarray(times, float).ravel()) for uid, times in zip(np.asarray(raw.UID).ravel(), np.asarray(raw.times, dtype=object).ravel(), strict=True)}
        track = producer.native.load_track_samples(spike_path.parent)
        spikes = producer.native.load_spikes(spike_path.parent)
        table = pd.read_csv(root / f"{tag}_scores.csv")
        all_scores.append(table)
        selection = pd.read_csv(root / f"{tag}_selection.csv", float_precision="round_trip")
        inspect = {selection.window_id.min(), selection.window_id.max()}
        train_interval, test_interval = producer.fold_intervals(track, fold)
        selected_again, _ = producer.select_windows(track, test_interval, session, fold)
        if not np.array_equal(selected_again.window_id, selection.window_id):
            raise ValueError("behavior-only selection mismatch")
        with np.load(root / f"{tag}_cache.npz") as data:
            if not np.array_equal(data["train_interval"], train_interval) or not np.array_equal(data["test_interval"], test_interval):
                raise ValueError("time separation cache mismatch")
            parent_units = data["parent_unit_ids"]
            for regime in producer.UNIT_REGIMES:
                maps, _, mask = producer.fit_maps(track, spikes, train_interval, regime, parent_units)
                units = data[f"unit_ids_{regime}"]
                if not np.array_equal(mask, data["train_mask"]) or not np.array_equal(units, maps["pooled"][0].unit_ids):
                    raise ValueError("train-only map/selection mismatch")
                for variant, encodings in maps.items():
                    for d, enc in enumerate(encodings):
                        if not np.allclose(enc.rates_hz, data[f"rates_{regime}_{variant}_{d}"], rtol=1e-12, atol=1e-12):
                            raise ValueError("map refit mismatch")
                for split in range(5):
                    indices = np.arange(len(units))
                    np.random.default_rng(20260804 + split).shuffle(indices)
                    # _split_cells permutes cell IDs, then sorts the returned IDs.
                    held_uids = np.sort(units[indices[: max(1, min(len(units) - 1, round(0.3 * len(units))))]])
                    train_uids = np.sort(units[indices[max(1, min(len(units) - 1, round(0.3 * len(units)))) :]])
                    tr, he = data[f"train_{regime}_{split}"], data[f"held_{regime}_{split}"]
                    if set(units[tr]) != set(train_uids) or set(units[he]) != set(held_uids) or np.intersect1d(tr, he).size:
                        raise ValueError("neural split mismatch")
                for window in selection.itertuples(index=False):
                    key = str(window.window_id)
                    edges = window.start_time_s + np.arange(11) * 0.02
                    if not np.allclose(edges, data[f"edges_{key}"], rtol=0, atol=1e-9):
                        raise ValueError("window times changed")
                    counted = np.column_stack([np.histogram(by_unit[int(uid)][(by_unit[int(uid)] >= edges[0]) & (by_unit[int(uid)] < edges[-1])], bins=edges)[0] for uid in units])
                    cap = independent_cap(counted, int(window.target_sleep_spikes), stable_seed(20260908, "hc11_maze_transfer_cap", session, fold, regime, window.window_id))
                    if np.any(cap > counted):
                        raise ValueError("thinning created spikes")
                    indices = producer.native.nearest_frame_indices(track.times_s, (edges[:-1] + edges[1:]) / 2)
                    if not np.array_equal(track.position_cm[indices], data[f"position_{key}"]) or not np.array_equal(track.direction[indices], data[f"direction_{key}"]):
                        raise ValueError("behavior/time alignment mismatch")
                    for count_regime, counts in zip(producer.COUNT_REGIMES, (counted, cap), strict=True):
                        if not np.array_equal(counts, data[f"counts_{regime}_{count_regime}_{key}"]):
                            raise ValueError("raw timestamp count or thinning mismatch")
                        part = table[(table.window_id == window.window_id) & (table.unit_regime == regime) & (table.count_regime == count_regime)]
                        digest = hashlib.sha256(str(counts.shape).encode() + counts.dtype.str.encode() + np.ascontiguousarray(counts).tobytes()).hexdigest()
                        if len(part) != 10 or not part.counts_sha256.eq(digest).all():
                            raise ValueError("shared count draw mismatch")
                        checks.append(
                            {"session": session, "fold": fold, "unit_regime": regime, "count_regime": count_regime, "window_id": window.window_id, "n_spikes": int(counts.sum())}
                        )
                        for split in range(5):
                            tr, he = data[f"train_{regime}_{split}"], data[f"held_{regime}_{split}"]
                            p = part[part.split == split]
                            if not p.n_heldout_spikes.eq(counts[:, he].sum()).all() or not p.n_train_spikes.eq(counts[:, tr].sum()).all():
                                raise ValueError("split totals wrong")
                            if window.window_id not in inspect or split not in (0, 2, 4):
                                continue
                            kernels = dense_kernels(data["centers"], edges, str(data["topology"]), float(data["track_length"]))
                            for variant in producer.frozen.VARIANTS:
                                rates = [data[f"rates_{regime}_{variant}_{d}"] for d in range(1 if variant == "pooled" else 2)]
                                train_parts = [direct_parts(counts[:, tr], r[tr], np.diff(edges)) for r in rates]
                                held_parts = [direct_parts(counts[:, he], r[he], np.diff(edges)) for r in rates]
                                row = p[p.encoding_variant == variant].iloc[0]
                                expected = {
                                    f"score_{model}": direct_predict(train_parts, held_parts, model, "count_conditioned", 1.0, kernels)["count_conditioned"]
                                    for model in producer.frozen.MODELS
                                }
                                expected.update(
                                    known_scores(
                                        counts,
                                        edges,
                                        rates,
                                        data[f"rates_{regime}_pooled_0"],
                                        data[f"occupancy_{regime}"],
                                        he,
                                        data[f"position_{key}"],
                                        data[f"direction_{key}"],
                                        data["bin_edges"],
                                        data["permutation"],
                                    )
                                )
                                errors = {name: abs(value - row[name]) for name, value in expected.items()}
                                if not np.isfinite(list(errors.values())).all() or max(errors.values()) > 1e-8:
                                    raise ValueError("independent predictive score mismatch")
                                q = iid_position_posterior(train_parts)
                                ties = verify_map_error(
                                    q, data["centers"], data[f"position_{key}"], str(data["topology"]), float(data["track_length"]), row.median_map_error_cm_all
                                )
                                predictions.append(
                                    {
                                        "session": session,
                                        "fold": fold,
                                        "unit_regime": regime,
                                        "count_regime": count_regime,
                                        "window_id": window.window_id,
                                        "split": split,
                                        "encoding_variant": variant,
                                        "maximum_score_error": max(errors.values()),
                                        "scores_checked": len(expected),
                                        "machine_precision_tied_MAP_bins": ties,
                                    }
                                )
        print(f"verified {tag}", flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    windows = pd.read_csv(root / "maze_transfer_windows.csv")
    qc = pd.read_csv(root / "maze_transfer_encoding_qc.csv")
    if not producer.technical_gates(scores, windows, qc).passed.all():
        raise ValueError("technical gates inconsistent")
    aggregate, scope = verify_aggregates(scores, root)
    if len(checks) != 6400 or len(predictions) != 768:
        raise ValueError("audit scope incomplete")
    pd.DataFrame(checks).to_csv(output / "raw_count_audit.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "independent_prediction_audit.csv", index=False)
    aggregate.to_csv(output / "aggregate_audit.csv", index=False)
    audit = build_script_provenance(input_paths={"run_manifest": manifest_path})
    audit.update(
        status="passed",
        count_matrices=len(checks),
        predictive_conditions=len(predictions),
        independent_scores=sum(p["scores_checked"] for p in predictions),
        maximum_score_error=max(p["maximum_score_error"] for p in predictions),
        **scope,
        verification_scope="Raw native/capped counts and alignment for every selected window/population; all guarded map refits (shared frozen fitting helper); 768 sampled conditions with separate dense log-domain predictions, known-position/nonspatial scores and MAP errors; all aggregate point estimates and cluster bootstrap CIs. Mean-error/coverage summaries not independently recalculated.",
    )
    (output / "audit_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.run_dir, args.output_dir)
