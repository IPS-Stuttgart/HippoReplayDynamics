#!/usr/bin/env python3
"""Freeze a PF-only diagnostic, then test fixed-coverage selection in Tanni."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from scripts._provenance import build_script_provenance, file_sha256

MODELS = ("mean", "spikes", "entropy", "spikes_entropy", "full")
METRICS = ("endpoint_separation_cm", "regional_tv", "a_entropy", "b_entropy",
           "a_width_cm", "b_width_cm", "a_truth_error_cm", "b_truth_error_cm", "map_region_agreement")
FEATURE_FIELDS = ("a_spikes", "a_active", "a_entropy", "a_width_cm", "a_peak", "n_cells",
                  "a_global_run_error_cm", "a_local_run_error_cm", "a_coverage", "grid_diagonal_cm")
KEY = ["dataset", "animal", "session", "source", "split", "draw", "event_index"]


def load_readouts(root):
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("incomplete producer is not an evaluation cohort")
    chunks = []
    for row in manifest["results"]:
        path = Path(row["artifact_dir"])/"event_readouts.csv.gz"
        if row["status"] != "complete" or file_sha256(path) != row["readouts_sha256"]:
            raise ValueError("failed or changed session input")
        chunks.append(pd.read_csv(path))
    data = pd.concat(chunks, ignore_index=True)
    if data.empty or data.duplicated(KEY).any():
        raise ValueError("empty or duplicate event readouts")
    if not np.isfinite(data["endpoint_separation_cm"]).all() or (data["endpoint_separation_cm"] < 0).any():
        raise ValueError("missing/invalid endpoint target")
    if not data.regional_tv.between(-1e-9, 1+1e-9).all():
        raise ValueError("invalid regional probability difference")
    return data


def weights(data):
    """Equal animals, sessions, events; callers use a single source/split/draw."""
    n_sessions = data.groupby("animal")["session"].transform("nunique")
    n_events = data.groupby(["animal", "session"])["event_index"].transform("size")
    return 1 / (data.animal.nunique()*n_sessions.to_numpy()*n_events.to_numpy())


def features(data, model):
    result = pd.DataFrame(index=data.index)
    if model not in MODELS:
        raise ValueError("unknown model")
    if model in ("spikes", "spikes_entropy", "full"):
        result["log_spikes"] = np.log1p(data.a_spikes)
        result["log_active"] = np.log1p(data.a_active)
    if model in ("entropy", "spikes_entropy", "full"):
        result["entropy"] = data.a_entropy
    if model == "full":
        for name in FEATURE_FIELDS[3:]:
            result[f"log_{name}"] = np.log1p(data[name])
    if len(result.columns) and np.isinf(result.to_numpy()).any():
        raise ValueError("infinite predictor")
    return result


def fit(data, model):
    if data.empty or not data.dataset.eq("pfeiffer_foster").all() or not data.source.eq("real").all() or not data.split.eq(0).all():
        raise ValueError("training must use PF real primary split only")
    if data.duplicated(KEY).any():
        raise ValueError("duplicate training samples")
    w = weights(data)
    target = np.log1p(data.endpoint_separation_cm.to_numpy())
    x = features(data, model)
    if model == "mean":
        return dict(model=model, intercept=float(w@target), columns=[], medians=[], means=[], scales=[], coefficients=[])
    medians = x.median().fillna(0).to_numpy()
    values = x.to_numpy()
    missing = ~np.isfinite(values)
    values = np.where(missing, medians, values)
    values = np.column_stack([values, missing.astype(float)])
    means = w@values
    scales = np.sqrt(w@((values-means)**2))
    scales[scales < 1e-10] = 1
    regression = Ridge(alpha=10).fit((values-means)/scales, target, sample_weight=w*len(data))
    return dict(model=model, intercept=float(regression.intercept_), columns=x.columns.tolist(),
                medians=medians.tolist(), means=means.tolist(), scales=scales.tolist(),
                coefficients=regression.coef_.tolist())


def predict(data, state):
    x = features(data, state["model"])
    if x.columns.tolist() != state["columns"]:
        raise ValueError("feature contract changed")
    if not len(x.columns):
        return np.full(len(data), state["intercept"])
    values = x.to_numpy()
    missing = ~np.isfinite(values)
    values = np.column_stack([np.where(missing, state["medians"], values), missing.astype(float)])
    return np.maximum(0, ((values-state["means"])/state["scales"])@state["coefficients"]+state["intercept"])


def rank_selection(data, risk, fraction):
    if not 0 < fraction <= 1 or not np.isfinite(risk).all():
        raise ValueError("invalid risk/retention")
    if len(data) != len(risk) or data.event_index.duplicated().any():
        raise ValueError("rank one session/source/split/draw at a time")
    order = np.lexsort((data.event_index.to_numpy(), np.asarray(risk)))
    keep = np.zeros(len(data), bool)
    keep[order[:int(np.ceil(fraction*len(data)))]] = True
    return keep


def freeze(args):
    data = load_readouts(args.input_dir)
    if not data.dataset.eq("pfeiffer_foster").all():
        raise ValueError("external data cannot enter freezing")
    train = data.loc[data.source.eq("real") & data.split.eq(0)].copy()
    if train.animal.nunique() != 4 or train.session.nunique() != 8:
        raise ValueError("all eight PF sessions/four rats required")
    oof = []
    for animal in sorted(train.animal.unique()):
        inside, outside = train.loc[train.animal.ne(animal)], train.loc[train.animal.eq(animal)].copy()
        for model in MODELS:
            outside[f"prediction_{model}"] = predict(outside, fit(inside, model))
        outside["training_animals"] = json.dumps(sorted(inside.animal.unique()))
        oof.append(outside)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.concat(oof).to_csv(args.output_dir/"pf_leave_one_animal_out.csv.gz", index=False)
    states = {model: fit(train, model) for model in MODELS}
    model = dict(states=states, training_animals=sorted(train.animal.unique()),
                 training_sessions=sorted(train.session.unique()), training_events=len(train),
                 retention=.5, target="log1p_endpoint_separation_cm", coefficients_frozen=True,
                 created_at_utc=datetime.now(UTC).isoformat(),
                 provenance=build_script_provenance(input_paths=dict(producer=args.input_dir/"manifest.json",
                     reporter=Path(__file__), protocol=ROOT/"docs/content_stability_protocol.md"), cwd=ROOT))
    (args.output_dir/"frozen_model.json").write_text(json.dumps(model, indent=2)+"\n")
    print(f"Frozen PF-only model: {len(train)} events, 8 sessions, 4 rats", flush=True)


def session_results(data, states):
    rows, forecast, selections = [], [], []
    group = ["dataset", "animal", "session", "source", "split", "draw"]
    for identity, local in data.groupby(group, sort=True):
        local = local.copy()
        actual = np.log1p(local.endpoint_separation_cm)
        risks = {model: predict(local, states[model]) for model in MODELS}
        for model, risk in risks.items():
            rho = float(spearmanr(risk, actual).statistic) if np.ptp(risk) > 0 and np.ptp(actual) > 0 else np.nan
            forecast.append(dict(zip(group, identity, strict=True), model=model,
                                 log_mse=float(np.mean((risk-actual)**2)), spearman=rho, n_events=len(local)))
        policies = {"diagnostic_full": risks["full"], "diagnostic_spikes_entropy": risks["spikes_entropy"],
                    "lowest_entropy": local.a_entropy.to_numpy(), "highest_spikes": -local.a_spikes.to_numpy()}
        for fraction in (.25, .5, .75):
            for policy in ("random_expectation", *policies):
                keep = np.ones(len(local), bool) if policy == "random_expectation" else rank_selection(local, policies[policy], fraction)
                selected = local.loc[keep]
                row = dict(zip(group, identity, strict=True), policy=policy, retention=fraction,
                           n_events=len(local), n_selected=int(np.ceil(fraction*len(local))))
                row.update({m: float(selected[m].mean()) for m in METRICS})
                rows.append(row)
                if fraction == .5 and policy == "diagnostic_full":
                    selected_rows = local[KEY + ["endpoint_separation_cm", "regional_tv", "a_truth_error_cm"]].copy()
                    selected_rows["risk"] = risks["full"]
                    selected_rows["selected"] = keep
                    selections.append(selected_rows)
    return pd.DataFrame(rows), pd.DataFrame(forecast), pd.concat(selections, ignore_index=True)


def aggregate_sessions(frame, fields, value_columns):
    # First average simulation draws, then sessions, never pool sessions by their size.
    session = frame.groupby(fields+["animal", "session"], dropna=False)[value_columns].mean().reset_index()
    animal = session.groupby(fields+["animal"], dropna=False)[value_columns].mean().reset_index()
    total = animal.groupby(fields, dropna=False)[value_columns].mean().reset_index()
    return animal, total


def relative_drop(reference, value):
    return (reference-value)/reference if np.isfinite(reference) and reference > 0 else np.nan


def gates(data, policy_animal, policy_total, forecast_total):
    rows = []
    def add(name, passed, value, criterion):
        rows.append(dict(gate=name, passed=bool(passed), value=value, criterion=criterion))
    real = data.loc[data.source.eq("real") & data.split.eq(0)]
    add("external_cohort_complete", real.animal.nunique() == 5 and real.session.nunique() == 25,
        f"{real.animal.nunique()} animals/{real.session.nunique()} sessions", "5 animals and 25 sessions")
    pf = forecast_total.loc[forecast_total.source.eq("real") & forecast_total.split.eq(0)].set_index("model")
    for baseline in ("mean", "spikes_entropy"):
        value = relative_drop(pf.loc[baseline, "log_mse"], pf.loc["full", "log_mse"])
        add(f"predictive_improvement_vs_{baseline}", value >= .05, value, ">=0.05 relative log-MSE reduction")
    def policy_case(source):
        return policy_total.loc[policy_total.source.eq(source) & policy_total.split.eq(0) & policy_total.retention.eq(.5)].set_index("policy")
    case = policy_case("real")
    for baseline, threshold in (("random_expectation", .2), ("lowest_entropy", .05), ("highest_spikes", .05)):
        value = relative_drop(case.loc[baseline, "endpoint_separation_cm"], case.loc["diagnostic_full", "endpoint_separation_cm"])
        add(f"separation_reduction_vs_{baseline}", value >= threshold, value, f">={threshold} at 50% retention")
    animal = policy_animal.loc[policy_animal.source.eq("real") & policy_animal.split.eq(0) & policy_animal.retention.eq(.5)]
    pivot = animal.pivot(index="animal", columns="policy", values="endpoint_separation_cm")
    diff = (pivot.random_expectation-pivot.diagnostic_full).to_numpy()
    add("animals_improve", len(diff) == 5 and (diff > 0).sum() >= 4, int((diff > 0).sum()), ">=4 of 5 animals")
    rng = np.random.default_rng(20260914)
    boot = rng.choice(diff, (5000, len(diff)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    add("animal_bootstrap_excludes_zero", low > 0, f"[{low}, {high}]", "descriptive 95% interval lower >0")
    for metric in ("regional_tv", "a_entropy", "b_entropy"):
        value = case.loc["random_expectation", metric]-case.loc["diagnostic_full", metric]
        add(f"retained_{metric}_improves", value > 0 if metric == "regional_tv" else value >= 0, value,
            "TV strictly lower; neither population more diffuse")
    for source in ("run_test", "sim_matched", "sim_drift"):
        truth = policy_case(source)
        for side in ("a", "b"):
            metric = f"{side}_truth_error_cm"
            value = truth.loc["random_expectation", metric]-truth.loc["diagnostic_full", metric]
            add(f"known_content_{source}_{side}_improves", value > 0, value, "retained true error lower than random expectation")
    add("overall_validated_diagnostic", all(row["passed"] for row in rows), "", "all prespecified gates")
    return pd.DataFrame(rows)


def evaluate(args):
    state = json.loads(args.frozen_model.read_text())
    data = load_readouts(args.input_dir)
    manifest = json.loads((args.input_dir/"manifest.json").read_text())
    if not data.dataset.eq("tanni2022").all() or set(data.animal) & set(state["training_animals"]):
        raise ValueError("not an independent external dataset")
    if manifest["input_file_sha256"]["frozen_model"] != file_sha256(args.frozen_model):
        raise ValueError("frozen model differs from pre-external model")
    rows, forecasts, selections = session_results(data, state["states"])
    pa, pt = aggregate_sessions(rows, ["source", "split", "policy", "retention"], list(METRICS))
    fa, ft = aggregate_sessions(forecasts, ["source", "split", "model"], ["log_mse", "spearman"])
    gate = gates(data, pa, pt, ft)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("policy_by_session", rows), ("policy_by_animal", pa), ("policy_summary", pt),
                        ("forecast_by_session", forecasts), ("forecast_by_animal", fa), ("forecast_summary", ft), ("gates", gate)):
        table.to_csv(args.output_dir/f"{name}.csv", index=False)
    selections.to_csv(args.output_dir/"selections.csv.gz", index=False)
    provenance = build_script_provenance(input_paths=dict(producer=args.input_dir/"manifest.json",
        frozen_model=args.frozen_model, reporter=Path(__file__), protocol=ROOT/"docs/content_stability_protocol.md"), cwd=ROOT)
    provenance.update(created_at_utc=datetime.now(UTC).isoformat(), status="complete",
                      overall_validated=bool(gate.iloc[-1].passed))
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    real = pt.loc[pt.source.eq("real") & pt.split.eq(0) & pt.retention.eq(.5)]
    lines = ["# PF-to-Tanni population-content diagnostic", "",
             f"Prespecified overall validation: **{'PASS' if gate.iloc[-1].passed else 'FAIL'}**.", "",
             "PF trained; Tanni tested without refitting. Primary split0, 50% retention per session.",
             "This predicts agreement of disjoint samples, not true biological replay content.",
             "Means equally weight animals after sessions; bootstrap is descriptive with five animals.",
             "Original pooled-MUA candidates and full-RUN cell eligibility are conditioning assumptions.",
             "No posterior smoothing, broadening or replacement. A/B regional TV uses nine fixed tiles.", "",
             "## Primary real-event results", "", real[["policy", "endpoint_separation_cm", "regional_tv", "a_entropy", "b_entropy"]].to_string(index=False),
             "", "## Gates", ""]
    lines += [f"- {r.gate}: {'pass' if r.passed else 'FAIL'}; {r.value}; {r.criterion}" for r in gate.itertuples()]
    lines += ["", "A failed gate is not permission to retune on Tanni and call the result independent validation.",
              "Known-content checks include observed 20 ms RUN windows and conditional endpoint simulations;",
              "neither validates a biological replay trajectory or its inferred goal."]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "evaluate"])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args)
    else:
        if args.frozen_model is None:
            raise ValueError("frozen model required")
        evaluate(args)


if __name__ == "__main__":
    main()
