#!/usr/bin/env python3
"""Frozen forward-only forecasting of held-out cell identities."""

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
from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator, forecasts, full_count_bins, mixture_scores
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_2d_burst_phase_prediction import HASHES
from scripts.audit_2d_count_conditioned_prediction import folds

ID = ["dataset", "animal", "session"]
KEY = ID + ["event_id", "split", "horizon"]
HORIZONS = (1, 2, 4)
MODELS = ("spatial_imm_real", "spatial_imm_permuted", "spatial_diffusion_real", "spatial_diffusion_permuted", "learned_hmm")
BASELINES = ("dwell_only", "frozen", "no_history", "global", "same_time")
SCORES = ["score_" + m for m in ("dynamic", *BASELINES)]


def score_event(counts, edges, rates, train, held, fit, spatial, permutation):
    x, discarded = full_count_bins(counts, edges)
    if sorted([*train, *held]) != list(range(counts.shape[1])) or not len(train) or not len(held):
        raise ValueError("disjoint full population partition required")
    if sorted(permutation) != list(range(rates.shape[1])):
        raise ValueError("invalid frozen map permutation")
    predictions, likelihoods = {}, {}
    if len(x):
        ll = identity_likelihood(x[:, train], rates[train])
        for kind, operator in spatial.items():
            for condition, order in (("real", np.arange(rates.shape[1])), ("permuted", permutation)):
                predictions[f"spatial_{kind}_{condition}"] = forecasts(ll[:, order], operator)
        neural = NeuralOperator(fit["initial"], fit["transition"], fit["occupancy"])
        predictions["learned_hmm"] = forecasts(identity_likelihood(x[:, train], fit["probabilities"][train]), neural)
        # Target cells only enter likelihood evaluation after forecasts exist.
        hl = identity_likelihood(x[:, held], rates[held])
        for kind in spatial:
            likelihoods[f"spatial_{kind}_real"] = hl
            likelihoods[f"spatial_{kind}_permuted"] = hl[:, permutation]
        likelihoods["learned_hmm"] = identity_likelihood(x[:, held], fit["probabilities"][held])
        global_ll = identity_likelihood(x[:, held], fit["global_probability"][held, None])[:, 0]
    rows = []
    for h in HORIZONS:
        nt = max(len(x) - h, 0)
        support = int(x[h:, held].sum()) if nt else 0
        common = {
            "horizon": h,
            "horizon_ms": h * 20,
            "unobserved_gap_ms": (h - 1) * 20,
            "n_full_bins": len(x),
            "n_target_bins": nt,
            "n_heldout_target_spikes": support,
            "discarded_partial_bin_spikes": discarded,
            "forecast_uses_future_training": False,
            "forecast_uses_heldout": False,
        }
        for model in MODELS:
            row = common | {"model": model, "status": "scored" if nt else "insufficient_full_bins"}
            if nt:
                p = predictions[model][h]
                hashed = posterior_sha256(p["dynamic"])
                for name, q in p.items():
                    values = mixture_scores(q, likelihoods[model][h:])
                    # Empty targets carry no cell-identity information.
                    values[x[h:, held].sum(axis=1) == 0] = 0
                    row["score_" + name] = float(values.sum())
                row["score_global"] = float(global_ll[h:].sum())
                row["forecast_sha256"] = hashed
                if posterior_sha256(p["dynamic"]) != hashed:
                    raise ValueError("prediction changed during scoring")
            else:
                row.update({name: np.nan for name in SCORES})
                row["forecast_sha256"] = ""
            rows.append(row)
    return rows


def task(job):
    item, parent, rate, learned, out = job
    parent, rate, learned, out = map(Path, (parent, rate, learned, out))
    tag = item["tag"]
    fields = {k: item[k] for k in ID}
    cache = dict(np.load(parent / f"{tag}_cache.npz"))
    selected = pd.read_csv(parent / f"{tag}_selection.csv")
    gains = pd.read_csv(rate / f"{tag}_gains.csv")
    spatial = {"imm": SpatialOperator(cache["centers"]), "diffusion": SpatialOperator(cache["centers"], imm=False)}
    rows, detail = [], []
    started = time.monotonic()
    for fold, test, cal, excluded in folds(selected):
        stem = f"{tag}__fold{fold}__k50"
        hm = json.loads((learned / f"{stem}_manifest.json").read_text())
        if (
            not hm["fit_converged"]
            or hm["test_ids"] != test.event_id.tolist()
            or hm["calibration_ids"] != cal.event_id.tolist()
            or hm["excluded_ids"] != excluded.event_id.tolist()
        ):
            raise ValueError("different frozen fitting population or unconverged model")
        fit = dict(np.load(learned / f"{stem}_fit.npz"))
        g = gains[gains.fold.eq(fold) & gains.alpha.eq(100)].set_index("unit_id")
        if g.index.duplicated().any() or set(g.index) != set(cache["unit_ids"]):
            raise ValueError("gain cell alignment changed")
        gain = g.loc[cache["unit_ids"], "gain"].to_numpy()
        rates = cache["rates"] * gain[:, None]
        detail.append(fields | {"fold": fold, "test_ids": test.event_id.tolist(), "calibration_ids": cal.event_id.tolist(), "excluded_ids": excluded.event_id.tolist()})
        for e in test.itertuples():
            eid = int(e.event_id)
            counts, edges = cache[f"counts_{eid}"], cache[f"edges_{eid}"]
            if counts.sum() != e.n_spikes_qc_units:
                raise ValueError("source count mismatch")
            for split in range(5):
                for row in score_event(counts, edges, rates, cache[f"train_{split}"], cache[f"held_{split}"], fit, spatial, cache["permutation"]):
                    rows.append(fields | {"event_id": eid, "fold": fold, "split": split} | row)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{tag}_forecast_scores.csv.gz", index=False)
    (out / f"{tag}_folds.json").write_text(json.dumps(detail, indent=2) + "\n")
    result = fields | {"tag": tag, "events": len(selected), "rows": len(table), "runtime_s": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def validate_rows(rows):
    if rows.empty or rows.duplicated(KEY + ["model"]).any():
        raise ValueError("nonempty unique event factors required")
    expected = set(itertools.product(range(5), HORIZONS, MODELS))
    if not rows.groupby(ID + ["event_id"]).apply(lambda g: set(g[["split", "horizon", "model"]].itertuples(index=False, name=None)) == expected, include_groups=False).all():
        raise ValueError("missing model, split or horizon")
    if rows.forecast_uses_future_training.any() or rows.forecast_uses_heldout.any():
        raise ValueError("forecast leakage declared")
    enough = rows.n_full_bins.gt(rows.horizon)
    if not rows.status.eq(np.where(enough, "scored", "insufficient_full_bins")).all():
        raise ValueError("incorrect temporal eligibility")
    if not rows.n_target_bins.eq((rows.n_full_bins - rows.horizon).clip(lower=0)).all():
        raise ValueError("incorrect target-bin count")
    if not np.isfinite(rows.loc[enough, SCORES]).all().all() or not rows.loc[enough, SCORES].le(1e-8).all().all() or not rows.loc[~enough, SCORES].isna().all().all():
        raise ValueError("invalid forecast scores or silent empty targets")
    if not rows.groupby(KEY).n_heldout_target_spikes.nunique().eq(1).all():
        raise ValueError("unpaired held-out targets")


def aggregate(rows):
    validate_rows(rows)
    pieces = []

    def add(frame, name, value):
        p = frame[KEY + ["n_target_bins", "n_heldout_target_spikes"]].copy()
        p["contrast"] = name
        p["delta"] = value.to_numpy()
        p["delta_per_spike"] = p.delta / p.n_heldout_target_spikes.replace(0, np.nan)
        pieces.append(p)

    for model, g in rows.groupby("model"):
        for baseline in BASELINES:
            add(g, f"{model}__dynamic_minus_{baseline}", g.score_dynamic - g["score_" + baseline])
    for kind in ("imm", "diffusion"):
        r = rows[rows.model.eq(f"spatial_{kind}_real")].set_index(KEY).sort_index()
        w = rows[rows.model.eq(f"spatial_{kind}_permuted")].set_index(KEY).sort_index()
        if not r.index.equals(w.index):
            raise ValueError("missing map pair")
        add(r.reset_index(), f"spatial_{kind}__real_minus_permuted", (r.score_dynamic - w.score_dynamic).reset_index(drop=True))
        interaction = (r.score_dynamic - r.score_dwell_only) - (w.score_dynamic - w.score_dwell_only)
        add(r.reset_index(), f"spatial_{kind}__map_route_interaction", interaction.reset_index(drop=True))
    r = rows[rows.model.eq("spatial_imm_real")].set_index(KEY).sort_index()
    for model in ("learned_hmm", "spatial_diffusion_real"):
        c = rows[rows.model.eq(model)].set_index(KEY).sort_index()
        if not r.index.equals(c.index):
            raise ValueError("missing model pair")
        add(r.reset_index(), "spatial_imm_minus_" + model, (r.score_dynamic - c.score_dynamic).reset_index(drop=True))
    splits = pd.concat(pieces, ignore_index=True)
    ek = ID + ["event_id", "horizon", "contrast"]
    event = splits.groupby(ek, as_index=False)[["delta", "delta_per_spike"]].median()
    support = splits.groupby(ek).delta_per_spike.count().rename("valid_neural_splits").reset_index()
    event = event.merge(support, validate="one_to_one")
    session = event.groupby(ID + ["horizon", "contrast"], as_index=False)[["delta", "delta_per_spike"]].mean()
    animal = session.groupby(["dataset", "animal", "horizon", "contrast"], as_index=False)[["delta", "delta_per_spike"]].mean()
    records = []
    for key, g in animal.groupby(["dataset", "horizon", "contrast"]):
        for metric in ("delta", "delta_per_spike"):
            values = g.sort_values("animal")[metric].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("animal without temporal predictive support")
            draws = np.array(list(itertools.product(range(len(values)), repeat=len(values))))
            ci = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
            records.append(
                {
                    "dataset": key[0],
                    "horizon": key[1],
                    "contrast": key[2],
                    "metric": metric,
                    "mean": values.mean(),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "positive_animals": int((values > 0).sum()),
                    "animals": len(values),
                }
            )
    return splits, event, session, animal, pd.DataFrame(records)


def decisions(summary):
    if summary.empty or summary.duplicated(["dataset", "horizon", "contrast", "metric"]).any():
        raise ValueError("unique nonempty summary required")
    main = summary[summary.horizon.eq(2) & summary.metric.eq("delta_per_spike")]
    result = []
    for model in ("spatial_imm_real", "spatial_diffusion_real", "learned_hmm"):
        required = [model + "__dynamic_minus_" + b for b in BASELINES if b != "same_time"]
        p = main[main.contrast.isin(required)]
        passed = bool(
            len(p) == 8
            and p.dataset.nunique() == 2
            and set(p.contrast) == set(required)
            and p.animals.ge(2).all()
            and p.ci_low.gt(0).all()
            and p.positive_animals.eq(p.animals).all()
        )
        spatial = False
        if model.startswith("spatial"):
            stem = model.removesuffix("_real")
            q = main[main.contrast.isin([stem + "__real_minus_permuted", stem + "__map_route_interaction"])]
            spatial = bool(passed and len(q) == 4 and q.ci_low.gt(0).all() and q.positive_animals.eq(q.animals).all())
        result.append(
            {
                "model": model,
                "replicated_forecasting_lead": passed,
                "spatial_route_specific_lead": spatial,
                "independent_confirmation": False,
                "high_importance_discovery_established": False,
            }
        )
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "rate", "learned"):
        p.add_argument(f"--{name}-dir", type=Path, required=True)
    for name in ("rate", "learned"):
        p.add_argument(f"--{name}-audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    inputs = {
        "script": Path(__file__),
        "protocol": ROOT / "docs/lagged_neural_prediction_protocol.md",
        "kernel": ROOT / "src/hipporeplayimm/lagged_neural_prediction.py",
        "spatial": ROOT / "src/hipporeplayimm/conditional_spatial_prediction.py",
        "spatial_engine": ROOT / "src/hipporeplayimm/state_space.py",
        "fold_logic": ROOT / "scripts/audit_2d_count_conditioned_prediction.py",
        "frozen_source_registry": ROOT / "scripts/audit_2d_burst_phase_prediction.py",
    }
    manifests = {}
    for name, (filename, digest) in HASHES.items():
        path = getattr(a, name + "_dir") / filename
        if file_sha256(path) != digest:
            raise ValueError("wrong frozen " + name)
        m = json.loads(path.read_text())
        if m["status"] != "complete":
            raise ValueError("incomplete source")
        manifests[name] = m
        inputs[name] = path
    for name, key in (("rate", "run_manifest"), ("learned", "scoring_manifest")):
        path = getattr(a, name + "_audit")
        audit = json.loads(path.read_text())
        if audit["status"] != "pass" or audit["input_file_sha256"][key] != HASHES[name][1]:
            raise ValueError("matching passing audit required")
        inputs[name + "_audit"] = path
    items = sorted(manifests["parent"]["completed"], key=lambda x: x["tag"])
    for item in items:
        tag = item["tag"]
        needed = {
            "parent": [tag + "_cache.npz", tag + "_selection.csv"],
            "rate": [tag + "_gains.csv"],
            "learned": [f"{tag}__fold{f}__k50_{suffix}" for f in range(5) for suffix in ("fit.npz", "manifest.json")],
        }
        for name, files in needed.items():
            for filename in files:
                path = getattr(a, name + "_dir") / filename
                if file_sha256(path) != manifests[name]["output_sha256"][filename]:
                    raise ValueError("changed source " + filename)
                inputs[filename] = path
    m = build_script_provenance(input_paths=inputs, cwd=ROOT)
    if m["git_dirty"]:
        raise ValueError("commit frozen code before scoring")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mp = out / "lagged_prediction_manifest.json"
    m.update(status="running", horizons=HORIZONS, primary_horizon=2, source_parameters_refitted=False, independent_confirmation=False)
    mp.write_text(json.dumps(m, indent=2) + "\n")
    tick = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            m["completed"] = list(pool.map(task, [(i, a.parent_dir, a.rate_dir, a.learned_dir, out) for i in items]))
        rows = pd.concat([pd.read_csv(out / f"{i['tag']}_forecast_scores.csv.gz") for i in items], ignore_index=True)
        if len(rows) != 9225 * 5 * 3 * 5 or len(items) != 33:
            raise ValueError("incomplete frozen experiment")
        base = rows[rows.model.eq("spatial_imm_real") & rows.split.eq(0)]
        coverage = base.groupby(ID + ["horizon"], as_index=False).agg(
            selected_events=("event_id", "size"),
            eligible_events=("status", lambda x: int(x.eq("scored").sum())),
            target_bins=("n_target_bins", "sum"),
            discarded_partial_spikes=("discarded_partial_bin_spikes", "sum"),
        )
        coverage.to_csv(out / "lagged_prediction_coverage.csv", index=False)
        tables = aggregate(rows)
        for name, table in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
            table.to_csv(out / f"lagged_prediction_{name}.csv.gz", index=False)
        decision = decisions(tables[-1])
        decision.to_csv(out / "lagged_prediction_decisions.csv", index=False)
        m.update(
            status="complete",
            events=9225,
            rows=len(rows),
            runtime_s=time.monotonic() - tick,
            technical_gates_passed=True,
            independent_audit_required=True,
            decisions=decision.to_dict("records"),
        )
    except BaseException as exc:
        m.update(status="failed", error=repr(exc))
        raise
    finally:
        m["output_sha256"] = {q.name: file_sha256(q) for q in out.iterdir() if q.is_file() and q != mp}
        mp.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["decisions"]), flush=True)


if __name__ == "__main__":
    main()
