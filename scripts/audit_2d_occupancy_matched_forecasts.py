#!/usr/bin/env python3
"""Frozen forward forecasts against an occupancy- and dwell-matched null."""

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

from hipporeplayimm.conditional_spatial_prediction import identity_likelihood
from hipporeplayimm.frozen_posterior_prediction import posterior_sha256
from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator, forward_filter, full_count_bins, mixture_scores
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_2d_lagged_neural_prediction import ID, KEY, MODELS, validate_rows

PARENT_SHA = "342d5670f77706ad784af7b1f98368f8eaa66e88247cfe78bb188d2daa79195a"


def predict(ll, operator, null):
    filtered = forward_filter(ll, operator)
    dynamic, matched = filtered.copy(), filtered.copy()
    result = {}
    for h in range(1, 5):
        dynamic, matched = operator.step(dynamic), null.step(matched)
        if h in (1, 2, 4) and h < len(ll):
            result[h] = {"dynamic": operator.collapse(dynamic[:-h]).copy(), "matched": operator.collapse(matched[:-h]).copy()}
            for q in result[h].values():
                if not np.isfinite(q).all() or (q < 0).any():
                    raise ValueError("invalid forecast probabilities")
                np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-10, rtol=0)
    return result


def score_model(x, rates, train, held, operator, null, *, position_order=None):
    ll = identity_likelihood(x[:, train], rates[train])
    if position_order is not None:
        if sorted(position_order) != list(range(rates.shape[1])):
            raise ValueError("invalid frozen position permutation")
        # Match the original scorer's reduction order and advanced-indexed layout.
        ll = ll[:, position_order]
    predictions = predict(ll, operator, null)
    target = identity_likelihood(x[:, held], rates[held])
    if position_order is not None:
        target = target[:, position_order]
    result = {}
    for h, p in predictions.items():
        record = {"dynamic_forecast_sha256": posterior_sha256(p["dynamic"])}
        for name, q in p.items():
            values = mixture_scores(q, target[h:])
            values[x[h:, held].sum(axis=1) == 0] = 0
            record["score_" + name] = float(values.sum())
        result[h] = record
    return result


def task(job):
    item, lagged, parent, rate, learned, output = job
    lagged, parent, rate, learned, output = map(Path, (lagged, parent, rate, learned, output))
    tag = item["tag"]
    started = time.monotonic()
    cache = dict(np.load(parent / f"{tag}_cache.npz"))
    old = pd.read_csv(lagged / f"{tag}_forecast_scores.csv.gz").set_index(["event_id", "split", "horizon", "model"]).sort_index()
    gains = pd.read_csv(rate / f"{tag}_gains.csv")
    fold_manifest = json.loads((lagged / f"{tag}_folds.json").read_text())
    spatial = {"imm": SpatialOperator(cache["centers"]), "diffusion": SpatialOperator(cache["centers"], imm=False)}
    nulls = {name: OccupancyMatchedNull.from_operator(op) for name, op in spatial.items()}
    parameters = []
    for name, null in nulls.items():
        path = output / f"{tag}__{name}_matched_null.npz"
        np.savez_compressed(path, **null.parameters())
        parameters.append({"model": name, "fold": -1, "file": path.name, **null.diagnostics()})
    rows = []
    for detail in fold_manifest:
        fold = detail["fold"]
        fit = dict(np.load(learned / f"{tag}__fold{fold}__k50_fit.npz"))
        neural = NeuralOperator(fit["initial"], fit["transition"], fit["occupancy"])
        neural_null = OccupancyMatchedNull.from_operator(neural)
        path = output / f"{tag}__fold{fold}_neural_matched_null.npz"
        np.savez_compressed(path, **neural_null.parameters())
        parameters.append({"model": "learned_hmm", "fold": fold, "file": path.name, **neural_null.diagnostics()})
        g = gains[gains.fold.eq(fold) & gains.alpha.eq(100)].set_index("unit_id")
        if g.index.duplicated().any() or set(g.index) != set(cache["unit_ids"]):
            raise ValueError("gain unit alignment changed")
        rates = cache["rates"] * g.loc[cache["unit_ids"], "gain"].to_numpy()[:, None]
        for eid in detail["test_ids"]:
            x, _ = full_count_bins(cache[f"counts_{eid}"], cache[f"edges_{eid}"])
            for split in range(5):
                tr, he = cache[f"train_{split}"], cache[f"held_{split}"]
                if not len(tr) or not len(he) or sorted([*tr, *he]) != list(range(x.shape[1])):
                    raise ValueError("invalid frozen cell partition")
                for model in MODELS:
                    order = None
                    if model == "learned_hmm":
                        op, null, maps = neural, neural_null, fit["probabilities"]
                    else:
                        kind = "imm" if model.startswith("spatial_imm") else "diffusion"
                        op, null = spatial[kind], nulls[kind]
                        maps = rates
                        order = cache["permutation"] if model.endswith("permuted") else np.arange(rates.shape[1])
                    predictions = score_model(x, maps, tr, he, op, null, position_order=order) if len(x) else {}
                    for h in (1, 2, 4):
                        key = (eid, split, h, model)
                        row = old.loc[key].to_dict() | dict(zip(("event_id", "split", "horizon", "model"), key, strict=True))
                        row["score_occupancy_dwell"] = np.nan
                        row["original_score_reconstruction_error"] = np.nan
                        row["original_forecast_hash_matches"] = False
                        if h in predictions:
                            p = predictions[h]
                            if p["dynamic_forecast_sha256"] != row["forecast_sha256"]:
                                raise ValueError(f"original forecast hash changed: {tag}, {key}")
                            error = abs(p["score_dynamic"] - row["score_dynamic"])
                            if error > 1e-8 or row["status"] != "scored":
                                raise ValueError("original score reconstruction changed")
                            row.update(score_occupancy_dwell=p["score_matched"], original_score_reconstruction_error=error, original_forecast_hash_matches=True)
                        elif row["status"] != "insufficient_full_bins":
                            raise ValueError("temporal eligibility changed")
                        rows.append(row)
    pd.DataFrame(rows).to_csv(output / f"{tag}_matched_scores.csv.gz", index=False)
    (output / f"{tag}_null_diagnostics.json").write_text(json.dumps(parameters, indent=2) + "\n")
    result = {k: item[k] for k in ID + ["tag", "events"]} | {
        "rows": len(rows),
        "runtime_s": time.monotonic() - started,
        "max_equilibrium_error": max(v["equilibrium_error"] for v in parameters),
        "max_mode_error": max(v["mode_probability_error"] for v in parameters),
    }
    print(json.dumps(result), flush=True)
    return result


def aggregate(rows):
    validate_rows(rows)
    enough = rows.status.eq("scored")
    errors = rows.loc[enough, "original_score_reconstruction_error"]
    if not rows.loc[enough, "original_forecast_hash_matches"].all() or not np.isfinite(errors).all() or errors.gt(1e-8).any():
        raise ValueError("original predictions not reconstructed")
    if (
        not np.isfinite(rows.loc[enough, "score_occupancy_dwell"]).all()
        or rows.loc[enough, "score_occupancy_dwell"].gt(1e-8).any()
        or not rows.loc[~enough, "score_occupancy_dwell"].isna().all()
    ):
        raise ValueError("invalid matched-null score")
    pieces = []

    def add(g, name, delta):
        p = g[KEY + ["n_target_bins", "n_heldout_target_spikes"]].copy()
        p["contrast"], p["delta"] = name, delta.to_numpy()
        p["delta_per_spike"] = p.delta / p.n_heldout_target_spikes.replace(0, np.nan)
        pieces.append(p)

    for model, g in rows.groupby("model"):
        add(g, model + "__dynamic_minus_matched", g.score_dynamic - g.score_occupancy_dwell)
        add(g, model + "__old_dwell_minus_matched", g.score_dwell_only - g.score_occupancy_dwell)
    for kind in ("imm", "diffusion"):
        r = rows[rows.model.eq(f"spatial_{kind}_real")].set_index(KEY).sort_index()
        w = rows[rows.model.eq(f"spatial_{kind}_permuted")].set_index(KEY).sort_index()
        if not r.index.equals(w.index):
            raise ValueError("missing map pair")
        d = (r.score_dynamic - r.score_occupancy_dwell) - (w.score_dynamic - w.score_occupancy_dwell)
        add(r.reset_index(), f"spatial_{kind}__matched_map_interaction", d.reset_index(drop=True))
    splits = pd.concat(pieces, ignore_index=True)
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
                raise ValueError("animal without predictive support")
            draws = np.array(list(itertools.product(range(len(v)), repeat=len(v))))
            ci = np.quantile(v[draws].mean(axis=1), [0.025, 0.975])
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


def decisions(summary, original):
    main = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike")]
    old = original[original.horizon.eq(2) & original.metric.eq("delta_per_spike")]
    rows = []
    for model in ("spatial_imm_real", "spatial_diffusion_real", "learned_hmm"):
        p = main[main.contrast.eq(model + "__dynamic_minus_matched")]
        matched = bool(len(p) == 2 and p.dataset.nunique() == 2 and p.ci_low.gt(0).all() and p.positive_animals.eq(p.animals).all())
        names = [model + "__dynamic_minus_" + b for b in ("frozen", "no_history", "global")]
        p = old[old.contrast.isin(names)]
        others = bool(len(p) == 6 and p.dataset.nunique() == 2 and p.ci_low.gt(0).all() and p.positive_animals.eq(p.animals).all())
        rows.append(
            {
                "model": model,
                "matched_destination_structure_lead": matched,
                "other_original_controls_pass": others,
                "all_updated_controls_pass": matched and others,
                "original_compound_verdict_changed": False,
                "independent_confirmation": False,
                "high_importance_discovery_established": False,
            }
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lagged-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    mp = a.lagged_dir / "lagged_prediction_manifest.json"
    if file_sha256(mp) != PARENT_SHA:
        raise ValueError("wrong frozen lagged run")
    parent_manifest = json.loads(mp.read_text())
    audit = json.loads(a.audit.read_text())
    if parent_manifest["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["run_manifest"] != PARENT_SHA:
        raise ValueError("matching complete audited source required")
    inputs = {
        "lagged_manifest": mp,
        "source_audit": a.audit,
        "producer": Path(__file__),
        "kernel": ROOT / "src/hipporeplayimm/occupancy_matched_forecast.py",
        "protocol": ROOT / "docs/occupancy_matched_forecast_protocol.md",
        "forecast_kernel": ROOT / "src/hipporeplayimm/lagged_neural_prediction.py",
        "original_aggregation": ROOT / "scripts/audit_2d_lagged_neural_prediction.py",
    }
    for name, path in parent_manifest["input_file_paths"].items():
        if file_sha256(path) != parent_manifest["input_file_sha256"][name]:
            raise ValueError("changed source input " + name)
        inputs["source_" + name] = Path(path)
    for name, digest in parent_manifest["output_sha256"].items():
        path = a.lagged_dir / name
        if file_sha256(path) != digest:
            raise ValueError("changed lagged output " + name)
        inputs["lagged_" + name] = path
    m = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if m["git_dirty"]:
        raise ValueError("commit code before scoring")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = out / "occupancy_matched_manifest.json"
    m.update(status="running", primary_horizon=2, source_parameters_refitted=False, original_verdict_changed=False)
    manifest.write_text(json.dumps(m, indent=2) + "\n")
    roots = [Path(parent_manifest["input_file_paths"][k]).parent for k in ("parent", "rate", "learned")]
    items = parent_manifest["completed"]
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            m["completed"] = list(pool.map(task, [(i, a.lagged_dir, *roots, out) for i in items]))
        rows = pd.concat([pd.read_csv(out / f"{i['tag']}_matched_scores.csv.gz") for i in items], ignore_index=True)
        if len(rows) != 9225 * 75 or len(items) != 33:
            raise ValueError("incomplete original population")
        tables = aggregate(rows)
        for name, t in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
            t.to_csv(out / f"occupancy_matched_{name}.csv.gz", index=False)
        decision = decisions(tables[-1], pd.read_csv(a.lagged_dir / "lagged_prediction_summary.csv.gz"))
        decision.to_csv(out / "occupancy_matched_decisions.csv", index=False)
        m.update(
            status="complete",
            events=9225,
            rows=len(rows),
            runtime_s=time.monotonic() - start,
            technical_gates_passed=True,
            independent_audit_required=True,
            decisions=decision.to_dict("records"),
        )
    except BaseException as exc:
        m.update(status="failed", error=repr(exc))
        raise
    finally:
        m["output_sha256"] = {f.name: file_sha256(f) for f in out.iterdir() if f.is_file() and f != manifest}
        manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["decisions"]), flush=True)


if __name__ == "__main__":
    main()
