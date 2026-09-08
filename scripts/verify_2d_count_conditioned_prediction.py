#!/usr/bin/env python3
"""Independent raw-count, prediction and aggregate audit of the 2D experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from verify_position_free_assembly_prediction import compare_table, raw_binned, read_pf_spikes, same

IDENTITY = ["dataset", "animal", "session"]
KEYS = IDENTITY + ["event_id", "split"]


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode("ascii") + str(a.shape).encode("ascii") + a.tobytes()).hexdigest()


def native_spikes(metadata):
    path = Path(metadata["source_path"])
    if metadata["dataset"] == "pfeiffer_foster":
        filename = path / "Spike_Data.mat"
        if file_sha256(filename) != metadata["consumed_files"][str(filename)]["sha256"]:
            raise ValueError("native PF spikes changed")
        values = read_pf_spikes(filename)
        return {int(u): np.sort(values[values[:, 1] == u, 0]) for u in np.unique(values[:, 1])}
    result = {}
    prefix = "/acquisition/timeseries/recording1/spikes"
    with h5py.File(path) as f:

        def read(key):
            a = np.asarray(f[key])
            if array_hash(a) != metadata["consumed_datasets"][key]["sha256"]:
                raise ValueError("native Tanni spike array changed")
            return a.ravel()

        for name in sorted(f[prefix]):
            match = re.fullmatch(r"electrode(\d+)", name)
            if match is None:
                raise ValueError("unexpected native electrode")
            group = f"{prefix}/{name}"
            times, labels = read(group + "/timestamps"), read(group + "/clustering/manual_1")
            if group + "/idx_keep" in f:
                keep = read(group + "/idx_keep").astype(bool)
                if len(keep) != len(times):
                    raise ValueError("invalid native keep mask")
                if len(labels) == len(times):
                    labels = labels[keep]
                elif len(labels) != keep.sum():
                    raise ValueError("native label alignment mismatch")
                times = times[keep]
            if len(labels) != len(times):
                raise ValueError("native label/time mismatch")
            for label in np.unique(labels):
                if label > 1:
                    uid = 1000 * int(match[1]) + int(label)
                    if uid in result:
                        raise ValueError("duplicate native unit")
                    result[uid] = np.sort(times[(labels == label) & np.isfinite(times)])
    return result


def ll(counts, rates):
    p = rates / rates.sum(axis=0, keepdims=True)
    constant = gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)
    return counts @ np.log(p) + constant[:, None]


def kernel(x, sigma):
    d2 = ((x[:, None] - x[None, :]) ** 2).sum(axis=2)
    a = np.exp(-d2 / (2 * sigma**2)) * (d2 <= (3 * sigma) ** 2)
    return csr_matrix(a / a.sum(axis=0, keepdims=True))


def reference_posterior(emissions, centers, times, imm):
    """Separate factorized forward/backward recursion, with dense-built kernels."""
    e = np.exp(emissions - emissions.max(axis=1, keepdims=True))
    dt = np.round(np.diff(times), 9)
    transitions = {float(t): kernel(centers, 60 * np.sqrt(t)) for t in np.unique(dt)}
    fixed = kernel(centers, 2.0)
    ntime, npos = e.shape
    nmodes = 3 if imm else 1
    forward = np.zeros((ntime, nmodes, npos))
    forward[0] = e[0] / (nmodes * e[0].sum())
    scales = np.ones(ntime)
    matrices = []
    for t in range(1, ntime):
        if imm:
            rho = np.exp(-dt[t - 1] / 0.06)
            modes = np.full((3, 3), (1 - rho) / 2)
            np.fill_diagonal(modes, rho)
            matrices.append(modes)
            mixed = modes.T @ forward[t - 1]
            predicted = np.array([fixed @ mixed[0], transitions[float(dt[t - 1])] @ mixed[1], np.full(npos, mixed[2].sum() / npos)])
        else:
            predicted = (transitions[float(dt[t - 1])] @ forward[t - 1, 0])[None, :]
        forward[t] = predicted * e[t]
        scales[t] = forward[t].sum()
        forward[t] /= scales[t]
    beta = np.ones((nmodes, npos))
    post = np.zeros_like(forward)
    post[-1] = forward[-1]
    for t in range(ntime - 2, -1, -1):
        weighted = e[t + 1] * beta
        if imm:
            back = np.array([fixed.T @ weighted[0], transitions[float(dt[t])].T @ weighted[1], np.full(npos, weighted[2].sum() / npos)])
            beta = matrices[t] @ back / scales[t + 1]
        else:
            beta = (transitions[float(dt[t])].T @ weighted[0])[None, :] / scales[t + 1]
        post[t] = forward[t] * beta
        post[t] /= post[t].sum()
    spatial = post.sum(axis=1)
    with np.errstate(divide="ignore"):
        return np.log(spatial)


def check_aggregate(scores, root):
    keyed = scores.set_index(KEYS + ["map"])
    real = keyed.xs("real", level="map")
    wrong = keyed.xs("population_code_permuted", level="map").reindex(real.index)
    contrasts = {f"imm_minus_{m}": real.score_first_order_imm - real[f"score_{m}"] for m in ("iid_position", "static_location", "diffusion", "event_global", "run_global")}
    contrasts["real_minus_wrong_imm"] = real.score_first_order_imm - wrong.score_first_order_imm
    contrasts["iid_minus_event_global"] = real.score_iid_position - real.score_event_global
    contrasts["diffusion_minus_iid"] = real.score_diffusion - real.score_iid_position
    contrasts["event_global_minus_run_global"] = real.score_event_global - real.score_run_global
    parts = []
    for name, values in contrasts.items():
        part = values.rename("delta").to_frame()
        part["delta_per_heldout_spike"] = values / real.n_heldout_spikes.replace(0, np.nan)
        parts.append(part.reset_index().assign(contrast=name))
    paired = pd.concat(parts, ignore_index=True)
    columns = ["delta", "delta_per_heldout_spike"]
    compare_table(paired, pd.read_csv(root / "conditional_2d_split_contrasts.csv"), KEYS + ["contrast"], columns, "split contrasts")
    event = paired.groupby(IDENTITY + ["event_id", "contrast"], as_index=False)[columns].median()
    compare_table(event, pd.read_csv(root / "conditional_2d_event_contrasts.csv"), IDENTITY + ["event_id", "contrast"], columns, "event medians")
    sessions = event.groupby(IDENTITY + ["contrast"], as_index=False)[columns].mean()
    animals = sessions.groupby(["dataset", "animal", "contrast"], as_index=False)[columns].mean()
    compare_table(sessions, pd.read_csv(root / "conditional_2d_by_session.csv"), IDENTITY + ["contrast"], columns, "session means")
    compare_table(animals, pd.read_csv(root / "conditional_2d_by_animal.csv"), ["dataset", "animal", "contrast"], columns, "animal means")
    observed = pd.read_csv(root / "conditional_2d_summary.csv").set_index(["dataset", "contrast"])
    n_panels = 0
    for (dataset, contrast), frame in event.groupby(["dataset", "contrast"]):
        rat_names = sorted(frame.animal.unique())
        arrays = [[g[columns].to_numpy() for _, g in frame[frame.animal.eq(r)].groupby("session")] for r in rat_names]
        random = np.random.default_rng(20260908)
        samples = np.zeros((5000, 2))
        for b in range(5000):
            rat_results = []
            for rat_index in random.integers(len(arrays), size=len(arrays)):
                local = arrays[rat_index]
                chosen = random.integers(len(local), size=len(local))
                session_results = []
                for i in chosen:
                    indices = random.integers(len(local[i]), size=len(local[i]))
                    session_results.append(np.nanmean(local[i][indices], axis=0))
                rat_results.append(np.nanmean(session_results, axis=0))
            samples[b] = np.nanmean(rat_results, axis=0)
        interval = np.nanquantile(samples, [0.025, 0.975], axis=0)
        a = animals[animals.dataset.eq(dataset) & animals.contrast.eq(contrast)]
        expected = [a.delta.mean(), *interval[:, 0], a.delta_per_heldout_spike.mean(), *interval[:, 1], a.delta.gt(0).sum(), len(a), frame.session.nunique(), len(frame)]
        names = ["mean", "ci_low", "ci_high", "mean_per_heldout_spike", "per_spike_ci_low", "per_spike_ci_high", "positive_animals", "animals", "sessions", "events"]
        same(observed.loc[(dataset, contrast), names].to_numpy(float), expected, "hierarchical bootstrap")
        n_panels += 1
    if n_panels != len(observed):
        raise ValueError("incomplete summary")
    return {"split_contrasts": len(paired), "event_contrasts": len(event), "bootstrap_panels": n_panels}


def parse_ids(value):
    return [] if pd.isna(value) else list(map(int, str(value).split(",")))


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    mpath = root / "conditional_2d_manifest.json"
    manifest = json.loads(mpath.read_text())
    if manifest["status"] != "complete" or len(manifest["completed"]) != 33:
        raise ValueError("incomplete production run")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed output {name}")
    source = Path(manifest["source_event_dir"])
    source_manifest = source / "coverage_event_definition_manifest.json"
    if file_sha256(source_manifest) != manifest["input_file_sha256"]["event_manifest"]:
        raise ValueError("changed source event manifest")
    original = json.loads(source_manifest.read_text())
    scores_all, checked, sessions_checked = [], [], []
    raw_events, analytic_scores, global_scores, dynamic_scores = 0, 0, 0, 0
    for item in sorted(manifest["completed"], key=lambda x: x["tag"]):
        tag = item["tag"]
        parent = next(v for v in original["results"] if all(v[k] == item[k] for k in IDENTITY))
        for a, b in (("source_cache_path", "source_cache_sha256"), ("windows_path", "windows_sha256")):
            if file_sha256(parent[a]) != parent[b]:
                raise ValueError("changed source cache/window")
        source_windows = pd.read_csv(parent["windows_path"])
        expected = source_windows[source_windows.detector.eq("source_high_mua") & source_windows.window_variant.eq("detected_core") & source_windows.eligible.eq(True)].rename(
            columns={"source_event_id": "event_id"}
        )
        selected = pd.read_csv(root / f"{tag}_selection.csv").sort_values(["start_s", "event_id"])
        compare_table(expected, selected, IDENTITY + ["event_id"], ["start_s", "end_s", "n_spikes_qc_units", "n_active_qc_units"], "frozen selection")
        scores = pd.read_csv(root / f"{tag}_scores.csv")
        meta = pd.read_csv(root / f"{tag}_folds.csv")
        scores_all.append(scores)
        with np.load(root / f"{tag}_cache.npz") as z:
            cache = {k: z[k] for k in z.files}
        metadata_path = Path(parent["source_cache_path"]).with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        native = native_spikes(metadata)
        with np.load(parent["source_cache_path"]) as a:
            units = a["cell_ids"][a["unit_qc_mask"]]
            same(units, cache["unit_ids"], "source units")
            same(cache["rates"], a["rates_hz"][a["unit_qc_mask"]][:, a["valid_spatial_bins"]], "source RUN maps")
            same(cache["centers"], a["bin_centers_cm"][a["valid_spatial_bins"]], "source spatial bins")
        for event in selected.itertuples(index=False):
            edges = cache[f"edges_{event.event_id}"]
            raw = np.column_stack([raw_binned(native[int(u)], edges, event.end_s) for u in units])
            same(raw, cache[f"counts_{event.event_id}"], "native raw counts")
            same([edges[0], edges[-1]], [event.start_s, event.end_s], "raw bin limits")
            same(cache[f"times_{event.event_id}"], (edges[1:] + edges[:-1]) / 2, "bin centers")
            raw_events += 1
        reference = cache["rates"].mean(axis=1)
        reference /= reference.sum()
        for fold, index in enumerate(np.array_split(np.arange(len(selected)), 5)):
            test = selected.iloc[index]
            possible = selected[~selected.event_id.isin(test.event_id)]
            good, excluded = [], []
            for c in possible.itertuples(index=False):
                separate = all(c.end_s + 1 <= t.start_s or t.end_s + 1 <= c.start_s for t in test.itertuples(index=False))
                (good if separate else excluded).append(c.event_id)
            row = meta[meta.fold.eq(fold)]
            if (
                len(row) != 1
                or not good
                or parse_ids(row.iloc[0].test_ids) != list(test.event_id)
                or parse_ids(row.iloc[0].calibration_ids) != good
                or parse_ids(row.iloc[0].excluded_ids) != excluded
            ):
                raise ValueError("calibration exclusion mismatch")
            count = np.sum([cache[f"counts_{i}"].sum(axis=0) for i in good], axis=0)
            probability = (count + 100 * reference) / (count.sum() + 100)
            same(probability, cache[f"global_{fold}"], "cross-event baseline")
        dynamic_ids = set(selected.iloc[[0, len(selected) // 2, len(selected) - 1]].event_id)
        maximum_error = 0.0
        for event in selected.itertuples(index=False):
            counts, times = cache[f"counts_{event.event_id}"], cache[f"times_{event.event_id}"]
            local = scores[scores.event_id.eq(event.event_id)]
            if len(local) != 10 or local.duplicated(["split", "map"]).any():
                raise ValueError("incomplete model factors")
            for split in range(5):
                tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                if sorted([*tr, *he]) != list(range(len(units))):
                    raise ValueError("invalid neuron split")
                rows = local[local.split.eq(split)]
                if set(rows["map"]) != {"real", "population_code_permuted"}:
                    raise ValueError("missing map condition")
                tll, hll = ll(counts[:, tr], cache["rates"][tr]), ll(counts[:, he], cache["rates"][he])
                q = tll - logsumexp(tll, axis=1, keepdims=True)
                fixed = tll.sum(axis=0)
                fixed -= logsumexp(fixed)
                analytic = {"iid_position": float(logsumexp(q + hll, axis=1).sum()), "static_location": float(logsumexp(fixed[None, :] + hll, axis=1).sum())}
                fold = int(rows.iloc[0].fold)
                test_ids = parse_ids(meta[meta.fold.eq(fold)].iloc[0].test_ids)
                if event.event_id not in test_ids or not rows.fold.eq(fold).all():
                    raise ValueError("event used the wrong calibration fold")
                for row in rows.itertuples(index=False):
                    same(
                        [row.n_train_spikes, row.n_heldout_spikes, row.n_train_cells, row.n_heldout_cells],
                        [counts[:, tr].sum(), counts[:, he].sum(), len(tr), len(he)],
                        "count/partition metadata",
                    )
                    for name, value in analytic.items():
                        same(value, getattr(row, f"score_{name}"), "analytic predictive score")
                        analytic_scores += 1
                    for name, p in (("event_global", cache[f"global_{fold}"]), ("run_global", reference)):
                        value = float(ll(counts[:, he], p[he, None]).sum())
                        same(value, getattr(row, f"score_{name}"), "global score")
                        global_scores += 1
                    if event.event_id not in dynamic_ids or split not in (0, 4):
                        continue
                    order = cache["permutation"] if row.map != "real" else np.arange(len(cache["centers"]))
                    for name in ("diffusion", "first_order_imm"):
                        post = reference_posterior(tll[:, order], cache["centers"], times, name == "first_order_imm")
                        value = float(logsumexp(post + hll[:, order], axis=1).sum())
                        observed = getattr(row, f"score_{name}")
                        same(value, observed, "independent dynamic prediction")
                        error = abs(value - observed)
                        maximum_error = max(maximum_error, error)
                        checked.append({**{k: item[k] for k in IDENTITY}, "event_id": event.event_id, "split": split, "map": row.map, "model": name, "absolute_error": error})
                        dynamic_scores += 1
        sessions_checked.append(
            {
                **{k: item[k] for k in IDENTITY},
                "events": len(selected),
                "score_rows": len(scores),
                "max_dynamic_error": maximum_error,
                "metadata_sha256": file_sha256(metadata_path),
            }
        )
        print(f"independently audited {tag}", flush=True)
    all_scores = pd.concat(scores_all, ignore_index=True)
    if len(all_scores) != 92250 or all_scores.heldout_used_for_inference.any() or not all_scores.posterior_unchanged.eq(True).all() or not all_scores.status.eq("success").all():
        raise ValueError("incomplete or leaking predictive experiment")
    aggregate = check_aggregate(all_scores, root)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite audit")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checked).to_csv(out / "independent_dynamic_predictions.csv", index=False)
    pd.DataFrame(sessions_checked).to_csv(out / "independent_session_checks.csv", index=False)
    result = build_script_provenance(input_paths={"run_manifest": mpath})
    result.update(
        status="pass",
        raw_events=raw_events,
        sessions=33,
        analytic_scores=analytic_scores,
        global_scores=global_scores,
        dynamic_scores=dynamic_scores,
        max_dynamic_error=float(max(r["absolute_error"] for r in checked)),
        **aggregate,
        scope="All native event counts, frozen inputs, calibration folds, analytic/global scores and aggregate/CI tables. Separate dynamic solver on first/middle/last event, cell splits 0/4 and both maps in every session. RUN maps are source-checked, not independently refitted.",
    )
    result["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "conditional_2d_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
