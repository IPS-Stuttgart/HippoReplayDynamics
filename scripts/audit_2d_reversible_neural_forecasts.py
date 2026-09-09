#!/usr/bin/env python3
"""Frozen original/reversed/reversible propagation on all existing events."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator, full_count_bins
from hipporeplayimm.reversible_neural_forecasts import ReversibleControls, score_model
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_2d_lagged_neural_prediction import ID, KEY, MODELS, validate_rows

SOURCE_SHA = "da514a563fc1ac1b9e3704cdf6b8a95b14b88049e14eebaf84901739fd07d980"
CONTRASTS = {
    "dynamic_minus_reversible": ("score_dynamic", "score_reversible"),
    "dynamic_minus_reverse": ("score_dynamic", "score_reverse"),
    "reversible_minus_matched": ("score_reversible", "score_occupancy_dwell"),
}


def task(job):
    item, source, lagged, parent, rate, learned, out = job
    source, lagged, parent, rate, learned, out = map(Path, (source, lagged, parent, rate, learned, out))
    tag = item["tag"]
    started = time.monotonic()
    cache = dict(np.load(parent / f"{tag}_cache.npz"))
    old = pd.read_csv(source / f"{tag}_matched_scores.csv.gz").set_index(["event_id", "split", "horizon", "model"]).sort_index()
    gains = pd.read_csv(rate / f"{tag}_gains.csv")
    folds = json.loads((lagged / f"{tag}_folds.json").read_text())
    spatial = {kind: SpatialOperator(cache["centers"], imm=kind == "imm") for kind in ("imm", "diffusion")}
    controls = {kind: ReversibleControls(op, np.load(source / f"{tag}__{kind}_matched_null.npz")["pi"]) for kind, op in spatial.items()}
    rows = []
    for f in folds:
        fold = f["fold"]
        fit = dict(np.load(learned / f"{tag}__fold{fold}__k50_fit.npz"))
        neural = NeuralOperator(fit["initial"], fit["transition"], fit["occupancy"])
        nc = ReversibleControls(neural, np.load(source / f"{tag}__fold{fold}_neural_matched_null.npz")["pi"])
        g = gains[gains.fold.eq(fold) & gains.alpha.eq(100)].set_index("unit_id")
        if g.index.duplicated().any() or set(g.index) != set(cache["unit_ids"]):
            raise ValueError("gain cell alignment changed")
        rates = cache["rates"] * g.loc[cache["unit_ids"], "gain"].to_numpy()[:, None]
        for eid in f["test_ids"]:
            x, _ = full_count_bins(cache[f"counts_{eid}"], cache[f"edges_{eid}"])
            for split in range(5):
                tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                if not len(tr) or not len(he) or sorted([*tr, *he]) != list(range(x.shape[1])):
                    raise ValueError("invalid frozen cell partition")
                for model in MODELS:
                    order = None
                    if model == "learned_hmm":
                        op, control, maps = neural, nc, fit["probabilities"]
                    else:
                        kind = "imm" if model.startswith("spatial_imm") else "diffusion"
                        op, control, maps = spatial[kind], controls[kind], rates
                        order = cache["permutation"] if model.endswith("permuted") else np.arange(rates.shape[1])
                    scores = score_model(x, maps, tr, he, op, control, position_order=order) if len(x) else {}
                    for h in (1, 2, 4):
                        key = (eid, split, h, model)
                        row = old.loc[key].to_dict() | dict(zip(("event_id", "split", "horizon", "model"), key, strict=True))
                        row.update(score_reverse=np.nan, score_reversible=np.nan, reconstruction_error=np.nan, reconstructed_forecast_hash_matches=False)
                        if h in scores:
                            p = scores[h]
                            if p["forecast_sha256"] != row["forecast_sha256"]:
                                raise ValueError(f"original forecast changed: {tag}, {key}")
                            error = abs(p["score_dynamic"] - row["score_dynamic"])
                            if error > 1e-8 or row["status"] != "scored":
                                raise ValueError("original score or eligibility changed")
                            row.update(
                                score_reverse=p["score_reverse"], score_reversible=p["score_reversible"], reconstruction_error=error, reconstructed_forecast_hash_matches=True
                            )
                        elif row["status"] != "insufficient_full_bins":
                            raise ValueError("temporal eligibility changed")
                        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"{tag}_reversible_scores.csv.gz", index=False)
    result = {k: item[k] for k in ID + ["tag", "events"]} | {"rows": len(rows), "runtime_s": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def aggregate(rows):
    validate_rows(rows)
    good = rows.status.eq("scored")
    errors = rows.loc[good, "reconstruction_error"]
    if not rows.loc[good, "reconstructed_forecast_hash_matches"].all() or not np.isfinite(errors).all() or errors.gt(1e-8).any():
        raise ValueError("original forecasts not reconstructed")
    cols = ["score_reverse", "score_reversible"]
    if not np.isfinite(rows.loc[good, cols]).all().all() or rows.loc[good, cols].gt(1e-8).any().any() or not rows.loc[~good, cols].isna().all().all():
        raise ValueError("invalid reverse/reversible scores")
    parts = []
    for model, g in rows.groupby("model"):
        for contrast, (a, b) in CONTRASTS.items():
            p = g[KEY + ["n_heldout_target_spikes"]].copy()
            p["contrast"], p["delta"] = model + "__" + contrast, (g[a] - g[b]).to_numpy()
            p["delta_per_spike"] = p.delta / p.n_heldout_target_spikes.replace(0, np.nan)
            parts.append(p)
    splits = pd.concat(parts, ignore_index=True)
    ek = ID + ["event_id", "horizon", "contrast"]
    events = splits.groupby(ek, as_index=False)[["delta", "delta_per_spike"]].median()
    support = splits.groupby(ek).delta_per_spike.count().rename("valid_neural_splits").reset_index()
    events = events.merge(support, validate="one_to_one")
    sessions = events.groupby(ID + ["horizon", "contrast"], as_index=False)[["delta", "delta_per_spike"]].mean()
    animals = sessions.groupby(["dataset", "animal", "horizon", "contrast"], as_index=False)[["delta", "delta_per_spike"]].mean()
    records = []
    for key, g in animals.groupby(["dataset", "horizon", "contrast"]):
        for metric in ("delta", "delta_per_spike"):
            v = g.sort_values("animal")[metric].to_numpy()
            if not np.isfinite(v).all():
                raise ValueError("animal missing finite predictive support")
            draw = np.array(list(itertools.product(range(len(v)), repeat=len(v))))
            ci = np.quantile(v[draw].mean(axis=1), [0.025, 0.975])
            records.append(
                {
                    "dataset": key[0],
                    "horizon": key[1],
                    "contrast": key[2],
                    "metric": metric,
                    "mean": v.mean(),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "positive_animals": int((v > 0).sum()),
                    "animals": len(v),
                }
            )
    return splits, events, sessions, animals, pd.DataFrame(records)


def decision(summary):
    names = ["learned_hmm__dynamic_minus_reversible", "learned_hmm__dynamic_minus_reverse"]
    g = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike") & summary.contrast.isin(names)]
    expected = set(itertools.product(("pfeiffer_foster", "tanni2022"), names))
    if len(g) != 4 or set(g[["dataset", "contrast"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("complete primary neural direction contrasts required")
    lead = bool(g.ci_low.gt(0).all() and g.positive_animals.eq(g.animals).all())
    return {
        "primary_model": "learned_hmm",
        "horizon_ms": 40,
        "directional_predictive_lead": lead,
        "original_compound_verdict_changed": False,
        "independent_confirmation": False,
        "high_importance_discovery_established": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    mp = a.source_dir / "occupancy_matched_manifest.json"
    if file_sha256(mp) != SOURCE_SHA:
        raise ValueError("wrong frozen source")
    source, audit = json.loads(mp.read_text()), json.loads(a.audit.read_text())
    if source["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != SOURCE_SHA:
        raise ValueError("matching complete source audit required")
    inputs = {
        "source_manifest": mp,
        "source_audit": a.audit,
        "producer": Path(__file__),
        "kernel": ROOT / "src/hipporeplayimm/reversible_neural_forecasts.py",
        "protocol": ROOT / "docs/reversible_neural_forecasts_protocol.md",
        "forecast_kernel": ROOT / "src/hipporeplayimm/lagged_neural_prediction.py",
        "original_validation": ROOT / "scripts/audit_2d_lagged_neural_prediction.py",
    }
    for name, path in source["input_file_paths"].items():
        if file_sha256(path) != source["input_file_sha256"][name]:
            raise ValueError("changed source input " + name)
        inputs["parent_" + name] = Path(path)
    for name, digest in source["output_sha256"].items():
        path = a.source_dir / name
        if file_sha256(path) != digest:
            raise ValueError("changed source output " + name)
        inputs["paired_" + name] = path
    m = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if m["git_dirty"]:
        raise ValueError("commit before scoring")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "reversible_forecasts_manifest.json"
    m.update(status="running", primary_model="learned_hmm", primary_horizon=2, source_parameters_refitted=False)
    manifest.write_text(json.dumps(m, indent=2) + "\n")
    lagged = Path(source["input_file_paths"]["lagged_manifest"]).parent
    lm = json.loads((lagged / "lagged_prediction_manifest.json").read_text())
    roots = [Path(lm["input_file_paths"][k]).parent for k in ("parent", "rate", "learned")]
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            m["completed"] = list(pool.map(task, [(i, a.source_dir, lagged, *roots, out) for i in source["completed"]]))
        rows = pd.concat([pd.read_csv(out / f"{i['tag']}_reversible_scores.csv.gz") for i in m["completed"]], ignore_index=True)
        if len(rows) != 9225 * 75 or len(m["completed"]) != 33:
            raise ValueError("incomplete original population")
        tables = aggregate(rows)
        for name, frame in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
            frame.to_csv(out / f"reversible_forecasts_{name}.csv.gz", index=False)
        verdict = decision(tables[-1])
        pd.DataFrame([verdict]).to_csv(out / "reversible_forecasts_decision.csv", index=False)
        m.update(status="complete", events=9225, rows=len(rows), runtime_s=time.monotonic() - started, decision=verdict, independent_audit_required=True)
    except BaseException as exc:
        m.update(status="failed", error=repr(exc))
        raise
    finally:
        m["output_sha256"] = {f.name: file_sha256(f) for f in out.iterdir() if f.is_file() and f != manifest}
        manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["decision"]), flush=True)


if __name__ == "__main__":
    main()
