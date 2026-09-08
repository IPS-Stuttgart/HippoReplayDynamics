#!/usr/bin/env python3
"""Check raw counts, cross-event fits and predictions with a separate dense solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from _provenance import build_script_provenance, file_sha256

from hipporeplayimm.assembly_predictive_control import fit_assembly

SEED = 20260908
KEYS = ["dataset", "phase", "session", "rat", "event_id", "split"]
CONDITIONS = ["dataset", "phase", "encoding_variant", "n_components", "contrast"]
MODELS = ("first_order_imm", "diffusion", "iid_position", "static_location")


def same(actual, expected, name):
    if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, atol=1e-9, rtol=1e-10, equal_nan=True):
        raise ValueError(f"{name} mismatch")


def compare_table(expected, actual, keys, columns, name):
    merged = expected.merge(actual, on=keys, suffixes=("_check", ""), validate="one_to_one")
    if len(expected) != len(actual) or len(merged) != len(expected) or not len(merged):
        raise ValueError(f"{name} missing rows")
    for c in columns:
        same(merged[c + "_check"], merged[c], name + ": " + c)


def likelihood(counts, probability):
    p = probability / probability.sum(axis=0, keepdims=True)
    constant = gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)
    return counts @ np.log(p) + constant[:, None]


def dense_posterior(ll, weights, centers, mode):
    logw = np.log(weights / weights.sum())
    if mode == "independent":
        q = ll + logw
    elif mode == "static":
        q = np.tile(ll.sum(axis=0) + logw, (len(ll), 1))
    elif mode == "persistent":
        forward = np.empty_like(ll)
        forward[0] = logw + ll[0]
        forward[0] -= logsumexp(forward[0])
        matrices = []
        for t, dt in enumerate(np.diff(centers), 1):
            rho = np.exp(-dt / 0.06)
            # This solver uses source-by-destination matrices.
            matrix = np.log(rho * np.eye(len(weights)) + (1 - rho) * weights[None, :])
            matrices.append(matrix)
            forward[t] = ll[t] + logsumexp(forward[t - 1, :, None] + matrix, axis=0)
            forward[t] -= logsumexp(forward[t])
        backward = np.zeros_like(ll)
        for t in range(len(ll) - 2, -1, -1):
            backward[t] = logsumexp(matrices[t] + ll[t + 1, None, :] + backward[t + 1, None, :], axis=1)
            backward[t] -= logsumexp(backward[t])
        q = forward + backward
    else:
        raise ValueError("unknown mode")
    return q - logsumexp(q, axis=1, keepdims=True)


def direct_scores(counts, centers, train, held, p, weights, global_p):
    tr = likelihood(counts[:, train], p[train])
    posterior = {m: dense_posterior(tr, weights, centers, m) for m in ("independent", "static", "persistent")}
    he = likelihood(counts[:, held], p[held])
    result = {m: float(logsumexp(q + he, axis=1).sum()) for m, q in posterior.items()}
    result["global"] = float(likelihood(counts[:, held], global_p[held, None]).sum())
    return result


def raw_binned(times, edges, end):
    times = np.sort(np.asarray(times, float).ravel())
    boundaries = np.minimum(edges, end)
    return np.diff(np.searchsorted(times, boundaries, side="left"))


def read_pf_spikes(path):
    import h5py

    if h5py.is_hdf5(path):
        with h5py.File(path) as handle:
            raw = np.asarray(handle["Spike_Data"]).T
    else:
        raw = np.asarray(loadmat(path, squeeze_me=True)["Spike_Data"])
    if raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
        raise ValueError("invalid native PF time/unit matrix")
    return raw


def audit_fit(training, p, w, global_p, trace, row, parts):
    reference = training.sum(axis=0) + 100 / training.shape[1]
    reference /= reference.sum()
    same(global_p, reference, "global probability")
    if not row.converged or len(trace) < 2 or (np.diff(trace) < -1e-7).any():
        raise ValueError("unconverged/nonmonotonic fit")
    objective = logsumexp(likelihood(training, p) + np.log(w), axis=1).sum()
    objective += 10 * (reference[:, None] * np.log(p)).sum() + np.log(w).sum() / len(w)
    same(objective, row.objective, "independent calibration objective")
    same(objective, trace[-1], "final objective trace")
    if abs(trace[-1] - trace[-2]) >= 1e-8 * (1 + abs(trace[-2])):
        raise ValueError("stopping criterion not satisfied")
    seed = int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).digest()[:4], "little")
    refit = fit_assembly(training, int(row.n_components), seed)
    same(refit.probabilities, p, "deterministic calibration refit")
    same(refit.weights, w, "deterministic weights refit")
    same(refit.objective_trace, trace, "deterministic restart selection")


def aggregates(scores, spatial, root):
    real = spatial[spatial["map"].eq("real")].set_index(KEYS + ["encoding_variant", "model"])
    parts = []
    for variant in ("pooled", "direction_mixture"):
        available = spatial[spatial.encoding_variant.eq(variant)].dataset.unique()
        joined = scores[scores.dataset.isin(available)].copy()
        joined["encoding_variant"] = variant
        index = pd.MultiIndex.from_frame(joined[KEYS + ["encoding_variant"]])
        for model in MODELS:
            observed = real.xs(model, level="model").conditional_heldout_log_score.reindex(index).to_numpy()
            if not np.isfinite(observed).all():
                raise ValueError("missing spatial comparator")
            for comparator in ("independent", "static", "persistent", "global"):
                parts.append(
                    joined[KEYS + ["encoding_variant", "n_components", "n_heldout_spikes"]].assign(
                        contrast=f"spatial_{model}_minus_assembly_{comparator}", delta=observed - joined[f"score_{comparator}"].to_numpy()
                    )
                )
        for comparator in ("independent", "static", "global"):
            parts.append(
                joined[KEYS + ["encoding_variant", "n_components", "n_heldout_spikes"]].assign(
                    contrast=f"assembly_persistent_minus_{comparator}", delta=joined.score_persistent - joined[f"score_{comparator}"]
                )
            )
    split = pd.concat(parts, ignore_index=True)
    split["delta_per_heldout_spike"] = split.delta / split.n_heldout_spikes.replace(0, np.nan)
    keys = KEYS + ["encoding_variant", "n_components", "contrast"]
    columns = ["delta", "delta_per_heldout_spike"]
    compare_table(split, pd.read_csv(root / "assembly_prediction_split_contrasts.csv"), keys, columns, "split contrast")
    event = split.groupby([k for k in keys if k != "split"], as_index=False)[columns].median()
    compare_table(event, pd.read_csv(root / "assembly_prediction_event_contrasts.csv"), [k for k in keys if k != "split"], columns, "event median")
    sessions = event.groupby(CONDITIONS + ["rat", "session"], as_index=False)[columns].mean()
    animals = sessions.groupby(CONDITIONS + ["rat"], as_index=False)[columns].mean()
    compare_table(sessions, pd.read_csv(root / "assembly_prediction_by_session.csv"), CONDITIONS + ["rat", "session"], columns, "session mean")
    compare_table(animals, pd.read_csv(root / "assembly_prediction_by_animal.csv"), CONDITIONS + ["rat"], columns, "animal mean")
    rows = []
    for key, group in animals.groupby(CONDITIONS):
        values = group.sort_values("rat").delta.to_numpy()
        if len(values) != 4:
            raise ValueError("incomplete animal coverage")
        draw = np.random.default_rng(SEED).integers(0, 4, (5000, 4))
        low, high = np.quantile(values[draw].mean(axis=1), [0.025, 0.975])
        rows.append(
            dict(zip(CONDITIONS, key, strict=True))
            | {
                "mean": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "positive_animals": int((values > 0).sum()),
                "animals": 4,
                "mean_per_heldout_spike": group.delta_per_heldout_spike.mean(),
            }
        )
    summary = pd.DataFrame(rows)
    compare_table(
        summary,
        pd.read_csv(root / "assembly_prediction_summary.csv"),
        CONDITIONS,
        ["mean", "ci_low", "ci_high", "positive_animals", "animals", "mean_per_heldout_spike"],
        "summary",
    )
    return {"split_contrasts": len(split), "event_contrasts": len(event), "aggregate_panels": len(summary)}


def ids(value):
    return [] if pd.isna(value) else list(map(int, str(value).split(",")))


def verify_outputs(root, manifest):
    if manifest.get("status") != "complete" or not manifest.get("output_sha256"):
        raise ValueError("incomplete experiment")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed artifact: {name}")
    gates = pd.read_csv(root / "assembly_prediction_gates.csv")
    if gates.empty or not gates.passed.eq(True).all() or not gates.gate.eq("overall").any():
        raise ValueError("technical gates failed")


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    manifest_path = root / "assembly_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_outputs(root, manifest)
    parents = {}
    for dataset, key, filename in (("PF", "source_pf", "manifest.json"), ("hc11", "source_hc11", "hc11_conditional_manifest.json")):
        parent_path = Path(manifest[key]) / filename
        alias = "pf_manifest" if dataset == "PF" else "hc_manifest"
        if file_sha256(parent_path) != manifest["input_file_sha256"][alias]:
            raise ValueError("parent manifest changed")
        parents[dataset] = json.loads(parent_path.read_text())
    pf_parent, hc_parent = parents["PF"], parents["hc11"]
    parent_scores = {}
    for dataset, path, filename in (("PF", manifest["source_pf"], "split_scores.csv"), ("hc11", manifest["source_hc11"], "hc11_conditional_event_scores.csv")):
        digest_key = "outputs_sha256" if dataset == "PF" else "output_sha256"
        if file_sha256(Path(path) / filename) != parents[dataset][digest_key][filename]:
            raise ValueError("parent evidence changed")
        s = pd.read_csv(Path(path) / filename).rename(columns={"event_index": "event_id"})
        s = s[s.observation.eq("count_conditioned") & s.inference_temperature.eq(1) & s.model.isin(MODELS)].assign(dataset=dataset)
        if dataset == "PF":
            s = s.assign(phase="RUN", encoding_variant="pooled")
        parent_scores[dataset] = s
    all_scores, all_spatial, score_checks = [], [], []
    raw_events, fitted, count_checks = 0, 0, 0
    for item in sorted(manifest["completed"], key=lambda x: x["tag"]):
        tag, dataset, session = item["tag"], item["dataset"], item["session"]
        scores = pd.read_csv(root / f"{tag}_scores.csv")
        selected = pd.read_csv(root / f"{tag}_selection.csv")
        meta = pd.read_csv(root / f"{tag}_folds.csv")
        fits = pd.read_csv(root / f"{tag}_fits.csv")
        spatial = pd.read_csv(root / f"{tag}_spatial.csv")
        source = parent_scores[dataset][parent_scores[dataset].session.eq(session)]
        compare_table(source, spatial, KEYS + ["map", "encoding_variant", "model"], ["conditional_heldout_log_score", "n_train_spikes", "n_heldout_spikes"], "parent scores")
        all_scores.append(scores)
        all_spatial.append(spatial)
        with np.load(root / f"{tag}_cache.npz") as cache:
            z = {k: cache[k] for k in cache.files}
        if dataset == "PF":
            rawpath = Path(pf_parent["arguments"]["dataset_root"]) / session / "Spike_Data.mat"
            if file_sha256(rawpath) != pf_parent["source_mat_sha256"][str(rawpath)]:
                raise ValueError("PF raw spikes changed")
            raw = read_pf_spikes(rawpath)
            by_unit = {int(u): raw[raw[:, 1] == u, 0] for u in z["unit_ids"]}
        else:
            parent_session = next(v for v in hc_parent["sessions"] if v["session"] == session)
            rawpath = next(Path(p) for p in parent_session["source_hashes"] if p.endswith("spikes.cellinfo.mat"))
            if file_sha256(rawpath) != parent_session["source_hashes"][str(rawpath)]:
                raise ValueError("hc11 raw spikes changed")
            raw = loadmat(rawpath, squeeze_me=True, struct_as_record=False)["spikes"]
            by_unit = {int(u): np.asarray(t).ravel() for u, t in zip(np.asarray(raw.UID).ravel(), np.asarray(raw.times, object).ravel(), strict=True)}
        for event in selected.itertuples(index=False):
            key = f"{event.phase}_{event.event_id}"
            counts, edges = z[f"counts_{key}"], z[f"edges_{key}"]
            same(edges[0], event.start_time_s, "event start")
            if edges[-1] < event.end_time_s - 1e-10:
                raise ValueError("event end outside bins")
            rebuilt = np.column_stack([raw_binned(by_unit[int(u)], edges, event.end_time_s) for u in z["unit_ids"]])
            same(rebuilt, counts, "independent raw spike counts")
            raw_events += 1
            for split in range(5):
                tr, he = z[f"train_{split}"], z[f"held_{split}"]
                if set(tr) & set(he) or sorted([*tr, *he]) != list(range(len(z["unit_ids"]))):
                    raise ValueError("invalid neuron partition")
                local = scores[scores.phase.eq(event.phase) & scores.event_id.eq(event.event_id) & scores.split.eq(split)]
                if set(local.n_components) != {3, 8} or len(local) != 2:
                    raise ValueError("incomplete event factors")
                for row in local.itertuples(index=False):
                    if ids(row.train_cell_ids) != list(z["unit_ids"][tr]) or ids(row.heldout_cell_ids) != list(z["unit_ids"][he]):
                        raise ValueError("neuron IDs do not match cache")
                    same([row.n_train_spikes, row.n_heldout_spikes, row.n_spikes], [counts[:, tr].sum(), counts[:, he].sum(), counts.sum()], "spike support")
                    count_checks += 1
        for phase, events in selected.groupby("phase"):
            events = events.sort_values(["start_time_s", "event_id"])
            if len(events) != 20 or events.event_id.duplicated().any():
                raise ValueError("incomplete frozen event cohort")
            for fold in (0, 1):
                test = events.iloc[fold * 10 : (fold + 1) * 10]
                candidates = events.iloc[(1 - fold) * 10 : (2 - fold) * 10]
                allowed, excluded = [], []
                for c in candidates.itertuples(index=False):
                    separated = all(c.end_time_s + 1 <= t.start_time_s or t.end_time_s + 1 <= c.start_time_s for t in test.itertuples(index=False))
                    (allowed if separated else excluded).append(c.event_id)
                if not allowed:
                    raise ValueError("empty calibration")
                saved = meta[meta.phase.eq(phase) & meta.fold.eq(fold)]
                if (
                    len(saved) != 1
                    or ids(saved.iloc[0].test_ids) != list(test.event_id)
                    or ids(saved.iloc[0].calibration_ids) != allowed
                    or ids(saved.iloc[0].excluded_ids) != excluded
                ):
                    raise ValueError("cross-event fold mismatch")
                width = 5 if dataset == "PF" else 1
                pooled = []
                for i in allowed:
                    c = z[f"counts_{phase}_{i}"]
                    pooled.extend([c[j : j + width].sum(axis=0) for j in range(0, len(c), width)])
                training = np.asarray(pooled)
                same(training, z[f"calibration_{phase}_{fold}"], "calibration input")
                same([len(training), training.sum()], [saved.iloc[0].n_calibration_bins, saved.iloc[0].n_calibration_spikes], "calibration support")
                for k in (3, 8):
                    key = f"fit_{phase}_{fold}_{k}"
                    p, w, global_p = z[key + "_probabilities"], z[key + "_weights"], z[key + "_global"]
                    fitrow = fits[fits.phase.eq(phase) & fits.fold.eq(fold) & fits.n_components.eq(k)]
                    if len(fitrow) != 1:
                        raise ValueError("missing fitted model")
                    audit_fit(training, p, w, global_p, z[key + "_trace"], fitrow.iloc[0], (dataset, session, phase, fold, k))
                    fitted += 1
                    for e in test.itertuples(index=False):
                        counts, edges = z[f"counts_{phase}_{e.event_id}"], z[f"edges_{phase}_{e.event_id}"]
                        local = scores[scores.phase.eq(phase) & scores.event_id.eq(e.event_id) & scores.n_components.eq(k)]
                        for row in local.itertuples(index=False):
                            if row.fold != fold:
                                raise ValueError("test event scored with wrong fold")
                            tr, he = z[f"train_{row.split}"], z[f"held_{row.split}"]
                            predicted = direct_scores(counts, (edges[1:] + edges[:-1]) / 2, tr, he, p, w, global_p)
                            for mode, value in predicted.items():
                                observed = getattr(row, f"score_{mode}")
                                same(value, observed, "independent held-out prediction")
                                score_checks.append(
                                    {
                                        "dataset": dataset,
                                        "session": session,
                                        "phase": phase,
                                        "event_id": e.event_id,
                                        "split": row.split,
                                        "n_components": k,
                                        "mode": mode,
                                        "absolute_error": abs(value - observed),
                                    }
                                )
        print(f"audited {tag}", flush=True)
    scores, spatial = pd.concat(all_scores), pd.concat(all_spatial)
    if len(scores) != 4800 or scores.heldout_used_for_inference.any() or not scores.status.eq("success").all() or raw_events != 480 or fitted != 96:
        raise ValueError("incomplete or leaking study")
    metrics = aggregates(scores, spatial, root)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite audit")
    out.mkdir(parents=True, exist_ok=True)
    checks = pd.DataFrame(score_checks)
    checks.to_csv(out / "independent_prediction_checks.csv", index=False)
    result = build_script_provenance(input_paths={"run_manifest": manifest_path})
    result.update(
        status="pass",
        raw_event_counts=raw_events,
        count_partition_rows=count_checks,
        calibration_fits=fitted,
        independent_scores=len(checks),
        max_prediction_error=float(checks.absolute_error.max()),
        **metrics,
        scope="Separate raw spike binning, calibration objective, dense assignment posterior, proper predictive score and aggregates. Deterministic EM refit uses shared fitting code. Parent RUN spatial maps/posteriors are reused, not independently refitted here.",
    )
    result["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "assembly_prediction_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
