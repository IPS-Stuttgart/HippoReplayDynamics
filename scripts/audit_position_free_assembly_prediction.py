#!/usr/bin/env python3
"""Frozen cross-event position-free assembly comparator for PF and hc-11."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import audit_pf_count_conditioned_prediction as pf
from _provenance import build_script_provenance, file_sha256
from audit_hc11_sleep_rate_transfer import folds

from hipporeplayimm.assembly_predictive_control import fit_assembly, predict_heldout
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import _time_bin_edges

SEED = 20260908
SPATIAL = ("first_order_imm", "diffusion", "iid_position", "static_location")
PF_DIGEST = "5904fd3741ee61c5c8006a8962a36be34dacc02a5f2e33a43c378a4718d980f6"
HC_DIGEST = "bc846d25e27b9400d2870b2cad534c781d1fdee12f27496e5a8d96db153d3d07"
MODES = ("independent", "static", "persistent")
KEYS = ["dataset", "phase", "session", "rat", "event_id", "split"]


def seed(*parts):
    import hashlib

    return int.from_bytes(hashlib.sha256("|".join(map(str, (SEED, *parts))).encode()).digest()[:4], "little")


def load_parents(pf_dir, hc_dir):
    ppath, hpath = pf_dir / "manifest.json", hc_dir / "hc11_conditional_manifest.json"
    if file_sha256(ppath) != PF_DIGEST or file_sha256(hpath) != HC_DIGEST:
        raise ValueError("unexpected parent experiment")
    p, h = json.loads(ppath.read_text()), json.loads(hpath.read_text())
    for root, meta, key, names in (
        (pf_dir, p, "outputs_sha256", ("split_scores.csv", "frozen_event_set.csv")),
        (hc_dir, h, "output_sha256", ("hc11_conditional_event_scores.csv", "frozen_selection.csv")),
    ):
        for name in names:
            if file_sha256(root / name) != meta[key][name]:
                raise ValueError("changed source evidence")
    for path, digest in p["source_mat_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError("changed PF raw data")
    for item in h["sessions"]:
        name = f"{item['session']}_cache.npz"
        if file_sha256(hc_dir / name) != h["output_sha256"][name]:
            raise ValueError("changed hc11 cache")
    return p, h


def prepare_pf(session_id, source, parent):
    session = load_replay_session(Path(parent["arguments"]["dataset_root"]) / session_id)
    evidence = pd.read_csv(source / "split_scores.csv")
    evidence = evidence[evidence.session.eq(session_id) & evidence.observation.eq("count_conditioned") & evidence.inference_temperature.eq(1) & evidence.model.isin(SPATIAL)].copy()
    events = pd.read_csv(source / "frozen_event_set.csv")
    indices = events[events.session.eq(session_id)].event_index.to_numpy(int)
    initial = evidence.iloc[0]
    parse = lambda text: np.array(list(map(int, str(text).split(","))))
    units = np.sort(np.concatenate([parse(initial.train_cell_ids), parse(initial.heldout_cell_ids)]))
    cache = {"unit_ids": units}
    for split in range(5):
        row = evidence[evidence.split.eq(split)].iloc[0]
        for name, ids in (("train", parse(row.train_cell_ids)), ("held", parse(row.heldout_cell_ids))):
            cache[f"{name}_{split}"] = np.flatnonzero(np.isin(units, ids))
            if not np.array_equal(units[cache[f"{name}_{split}"]], ids):
                raise ValueError("PF cell-order mismatch")
    selected = []
    spikes = {int(uid): session.spikes[session.spikes[:, 1] == uid, 0] for uid in units}
    for index in indices:
        event = session.ripple(int(index))
        edges = _time_bin_edges(event.start, event.end, parent["emission_source_config"]["time_bin_s"])
        counts = np.column_stack([np.histogram(v[(v >= event.start) & (v < event.end)], bins=edges)[0] for v in spikes.values()])
        key = f"RUN_{index}"
        cache[f"counts_{key}"] = counts
        cache[f"edges_{key}"] = edges
        for split in range(5):
            reference = evidence[evidence.event_index.eq(index) & evidence.split.eq(split)]
            tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
            if (
                not reference.training_counts_sha256.eq(pf._hash_array(counts[:, tr])).all()
                or not reference.n_train_spikes.eq(counts[:, tr].sum()).all()
                or not reference.n_heldout_spikes.eq(counts[:, he].sum()).all()
            ):
                raise ValueError("PF raw counts differ from original scores")
        selected.append({"dataset": "PF", "phase": "RUN", "session": session_id, "rat": session.rat, "event_id": index, "start_time_s": event.start, "end_time_s": event.end})
    evidence = evidence.rename(columns={"event_index": "event_id"}).assign(dataset="PF", phase="RUN", encoding_variant="pooled")
    return pd.DataFrame(selected), cache, evidence


def prepare_hc(session_id, source):
    all_selection = pd.read_csv(source / "frozen_selection.csv")
    selected = all_selection[all_selection.session.eq(session_id)].rename(columns={"animal": "rat"}).assign(dataset="hc11")
    original = pd.read_csv(source / "hc11_conditional_event_scores.csv")
    original = original[original.session.eq(session_id) & original.observation.eq("count_conditioned") & original.inference_temperature.eq(1) & original.model.isin(SPATIAL)]
    original = original.rename(columns={"event_index": "event_id"}).assign(dataset="hc11")
    with np.load(source / f"{session_id}_cache.npz") as z:
        cache = {k: z[k] for k in z.files if k == "unit_ids" or k.startswith(("counts_", "edges_", "train_", "held_"))}
    return selected, cache, original


def bin_calibration(counts, edges):
    dt = float(np.median(np.diff(edges)))
    if np.isclose(dt, 0.004, atol=1e-8):
        width = 5
    elif np.isclose(dt, 0.02, atol=1e-8):
        width = 1
    else:
        raise ValueError("unexpected source binning")
    pooled = np.add.reduceat(counts, np.arange(0, len(counts), width), axis=0)
    if not np.array_equal(pooled.sum(axis=0), counts.sum(axis=0)):
        raise ValueError("calibration pooling lost spikes")
    return pooled


def task(args):
    dataset, session, source, parent, out = args
    source, out = Path(source), Path(out)
    selected, z, original = prepare_pf(session, source, parent) if dataset == "PF" else prepare_hc(session, source)
    tag = dataset + "_" + session.replace("/", "_")
    units = z["unit_ids"]
    scores, fit_rows, fold_rows = [], [], []
    for phase, phase_selection in selected.groupby("phase"):
        for fold, test, calibration, excluded in folds(phase_selection):
            training = np.vstack([bin_calibration(z[f"counts_{phase}_{int(e.event_id)}"], z[f"edges_{phase}_{int(e.event_id)}"]) for e in calibration.itertuples(index=False)])
            z[f"calibration_{phase}_{fold}"] = training
            fold_rows.append(
                {
                    "dataset": dataset,
                    "session": session,
                    "phase": phase,
                    "fold": fold,
                    "test_ids": ",".join(map(str, test.event_id)),
                    "calibration_ids": ",".join(map(str, calibration.event_id)),
                    "excluded_ids": ",".join(map(str, excluded.event_id)),
                    "n_calibration_bins": len(training),
                    "n_calibration_spikes": training.sum(),
                }
            )
            for k in (3, 8):
                fit = fit_assembly(training, k, seed(dataset, session, phase, fold, k))
                key = f"fit_{phase}_{fold}_{k}"
                z[f"{key}_probabilities"], z[f"{key}_weights"], z[f"{key}_global"] = fit.probabilities, fit.weights, fit.global_probability
                z[f"{key}_trace"] = fit.objective_trace
                fit_rows.append(
                    {
                        "dataset": dataset,
                        "session": session,
                        "phase": phase,
                        "fold": fold,
                        "n_components": k,
                        "objective": fit.objective,
                        "iterations": fit.iterations,
                        "converged": fit.converged,
                        "n_calibration_events": len(calibration),
                        "n_calibration_spikes": training.sum(),
                    }
                )
                for event in test.itertuples(index=False):
                    ckey = f"{phase}_{int(event.event_id)}"
                    counts, edges = z[f"counts_{ckey}"], z[f"edges_{ckey}"]
                    centers = (edges[1:] + edges[:-1]) / 2
                    for split in range(5):
                        train, held = z[f"train_{split}"], z[f"held_{split}"]
                        values, hashes = predict_heldout(counts, centers, train, held, fit)
                        scores.append(
                            {
                                "dataset": dataset,
                                "session": session,
                                "rat": event.rat,
                                "phase": phase,
                                "fold": fold,
                                "event_id": event.event_id,
                                "split": split,
                                "n_components": k,
                                **{f"score_{m}": v for m, v in values.items()},
                                "posterior_hashes": json.dumps(hashes, sort_keys=True),
                                "heldout_used_for_inference": False,
                                "train_cell_ids": ",".join(map(str, units[train])),
                                "heldout_cell_ids": ",".join(map(str, units[held])),
                                "n_train_spikes": counts[:, train].sum(),
                                "n_heldout_spikes": counts[:, held].sum(),
                                "n_spikes": counts.sum(),
                                "status": "success",
                            }
                        )
    np.savez_compressed(out / f"{tag}_cache.npz", **z)
    for name, table in (("scores", pd.DataFrame(scores)), ("fits", pd.DataFrame(fit_rows)), ("folds", pd.DataFrame(fold_rows)), ("selection", selected), ("spatial", original)):
        table.to_csv(out / f"{tag}_{name}.csv", index=False)
    return {"tag": tag, "dataset": dataset, "session": session, "rows": len(scores), "events": len(selected)}


def contrasts(scores, spatial):
    real = spatial[spatial["map"].eq("real")].pivot(index=KEYS + ["encoding_variant"], columns="model", values="conditional_heldout_log_score").reset_index()
    joined = scores.merge(real, on=KEYS, validate="many_to_many")
    if joined.duplicated(KEYS + ["n_components", "encoding_variant"]).any():
        raise ValueError("duplicate scoring factors")
    records = []
    base_keys = KEYS + ["encoding_variant", "n_components", "n_heldout_spikes"]
    for model in SPATIAL:
        for comparator in (*MODES, "global"):
            records.append(joined[base_keys].assign(contrast=f"spatial_{model}_minus_assembly_{comparator}", delta=joined[model] - joined[f"score_{comparator}"]))
    for comparator in ("independent", "static", "global"):
        records.append(joined[base_keys].assign(contrast=f"assembly_persistent_minus_{comparator}", delta=joined.score_persistent - joined[f"score_{comparator}"]))
    paired = pd.concat(records, ignore_index=True)
    paired["delta_per_heldout_spike"] = paired.delta / paired.n_heldout_spikes.replace(0, np.nan)
    group = [k for k in base_keys if k not in ("split", "n_heldout_spikes")] + ["contrast"]
    event = paired.groupby(group, as_index=False)[["delta", "delta_per_heldout_spike"]].median()
    return paired, event


def aggregate(event):
    conditions = ["dataset", "phase", "encoding_variant", "n_components", "contrast"]
    sessions = event.groupby(conditions + ["rat", "session"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    rats = sessions.groupby(conditions + ["rat"], as_index=False)[["delta", "delta_per_heldout_spike"]].mean()
    rows = []
    for key, group in rats.groupby(conditions):
        values = group.sort_values("rat").delta.to_numpy()
        boot = np.random.default_rng(SEED).integers(0, len(values), size=(5000, len(values)))
        lo, hi = np.quantile(values[boot].mean(axis=1), [0.025, 0.975])
        rows.append(
            dict(zip(conditions, key, strict=True))
            | {
                "mean": values.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "positive_animals": int((values > 0).sum()),
                "animals": len(values),
                "mean_per_heldout_spike": group.delta_per_heldout_spike.mean(),
            }
        )
    return pd.DataFrame(rows), rats, sessions


def gates(scores, selected, fits):
    expected = len(selected) * 5 * 2
    keys = ["dataset", "phase", "session", "event_id"]
    if scores.empty or selected.empty or fits.empty:
        return False
    factors = {(k, s) for k in (3, 8) for s in range(5)}
    return bool(
        len(scores) == expected
        and not scores.duplicated(keys + ["split", "n_components"]).any()
        and set(scores[keys].itertuples(index=False, name=None)) == set(selected[keys].itertuples(index=False, name=None))
        and all(set(g[["n_components", "split"]].itertuples(index=False, name=None)) == factors for _, g in scores.groupby(keys))
        and np.isfinite(scores[[f"score_{m}" for m in (*MODES, "global")]]).all().all()
        and scores[[f"score_{m}" for m in (*MODES, "global")]].le(1e-8).all().all()
        and fits.converged.eq(True).all()
        and scores.status.eq("success").all()
        and scores.heldout_used_for_inference.eq(False).all()
        and scores.n_spikes.eq(scores.n_train_spikes + scores.n_heldout_spikes).all()
        and all(not set(t.split(",")).intersection(h.split(",")) for t, h in scores[["train_cell_ids", "heldout_cell_ids"]].itertuples(index=False, name=None))
    )


def synthetic_check(out):
    from hipporeplayimm.assembly_predictive_control import multinomial_ll

    rows = []
    for replicate in range(12):
        rng = np.random.default_rng(seed("assembly_known_generator", replicate))
        p = np.full((18, 3), 0.01)
        for j in range(3):
            p[j::3, j] = 0.2
        p /= p.sum(axis=0)
        cal = np.array([rng.multinomial(15, p[:, j]) for j in rng.integers(0, 3, 300)])
        for k in (3, 8):
            fit = fit_assembly(cal, k, seed("assembly_known_fit", replicate, k))
            for generator in MODES:
                for event in range(15):
                    state = int(rng.integers(0, 3))
                    labels = []
                    for _ in range(25):
                        if generator == "independent" or (generator == "persistent" and rng.random() > np.exp(-0.004 / 0.06)):
                            state = int(rng.integers(0, 3))
                        labels.append(state)
                    counts = np.array([rng.multinomial(5, p[:, j]) for j in labels])
                    values, _ = predict_heldout(counts, np.arange(25) * 0.004, np.arange(12), np.arange(12, 18), fit)
                    # The oracle is recorded only after prediction, not used by any fit.
                    oracle = multinomial_ll(counts[:, 12:], p[12:])[np.arange(25), labels].sum()
                    rows.append({"replicate": replicate, "n_components": k, "generator": generator, "event": event, **values, "oracle": oracle})
    table = pd.DataFrame(rows)
    table.to_csv(out / "synthetic_operating_check.csv", index=False)
    group = table.groupby(["n_components", "generator"])[["independent", "static", "persistent", "global", "oracle"]].mean()
    group.to_csv(out / "synthetic_operating_summary.csv")
    for k in (3, 8):
        if not (
            group.loc[(k, "persistent"), "persistent"] > group.loc[(k, "persistent"), "global"]
            and group.loc[(k, "persistent"), "persistent"] > group.loc[(k, "persistent"), "static"]
            and group.loc[(k, "independent"), "independent"] > group.loc[(k, "independent"), "persistent"]
            and group.loc[(k, "static"), "static"] > group.loc[(k, "static"), "independent"]
        ):
            raise ValueError("known-generator operating check failed; no real interpretation")
    return len(table)


def run(args):
    pf_dir, hc_dir, out = Path(args.pf_dir).resolve(), Path(args.hc_dir).resolve(), Path(args.output_dir).resolve()
    p, h = load_parents(pf_dir, hc_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a frozen experiment")
    out.mkdir(parents=True, exist_ok=True)
    provenance = build_script_provenance(
        input_paths={
            "pf_manifest": pf_dir / "manifest.json",
            "hc_manifest": hc_dir / "hc11_conditional_manifest.json",
            "protocol": ROOT / "docs/position_free_assembly_prediction_protocol.md",
        }
    )
    provenance.update(
        status="running",
        n_components=[3, 8],
        component_pseudocount=10,
        global_pseudocount=100,
        retention_tau_s=0.06,
        n_splits=5,
        source_pf=str(pf_dir),
        source_hc11=str(hc_dir),
        claim_boundary="position-free comparator; not an absence-of-replay null or a unique mechanism",
    )
    path = out / "assembly_prediction_manifest.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    completed = []
    try:
        provenance["synthetic_rows"] = synthetic_check(out)
        print("known-generator operating check passed", flush=True)
        pf_sessions = pd.read_csv(pf_dir / "frozen_event_set.csv").session.unique()
        tasks = [("PF", sid, pf_dir, p, out) for sid in pf_sessions] + [("hc11", item["session"], hc_dir, h, out) for item in h["sessions"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(task, item) for item in tasks]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(json.dumps(result), flush=True)
        tables = {name: pd.concat([pd.read_csv(out / f"{r['tag']}_{name}.csv") for r in completed], ignore_index=True) for name in ("scores", "selection", "fits", "spatial")}
        passed = gates(tables["scores"], tables["selection"], tables["fits"])
        pd.DataFrame([{"gate": "complete_proper_nonleaking_prediction", "passed": passed}, {"gate": "overall", "passed": passed}]).to_csv(
            out / "assembly_prediction_gates.csv", index=False
        )
        if not passed:
            raise ValueError("technical gates failed")
        paired, events = contrasts(tables["scores"], tables["spatial"])
        summary, animal, sessions = aggregate(events)
        for name, frame in (("split_contrasts", paired), ("event_contrasts", events), ("summary", summary), ("by_animal", animal), ("by_session", sessions)):
            frame.to_csv(out / f"assembly_prediction_{name}.csv", index=False)
        primary = summary[summary.dataset.eq("PF") & summary.contrast.eq("spatial_first_order_imm_minus_assembly_persistent")]
        supported = len(primary) == 2 and primary.ci_low.gt(0).all() and primary.positive_animals.eq(4).all()
        provenance.update(
            status="complete",
            runtime_s=time.monotonic() - started,
            rows=len(tables["scores"]),
            events=len(tables["selection"]),
            decision="bounded_spatial_advantage_against_tested_assemblies" if supported else "spatial_advantage_not_established_against_assemblies",
        )
    except BaseException as exc:
        provenance.update(status="failed", error=repr(exc))
        raise
    finally:
        provenance["completed"] = completed
        provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p != path}
        path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-dir", default="/mnt/seagate10tb/florianpfaff/pf-count-conditioned-prediction-all160x5-20260908")
    parser.add_argument("--hc-dir", default="/mnt/seagate10tb/florianpfaff/hc11-conditional-cross-cell-prediction-320x5-20260908")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
