#!/usr/bin/env python3
"""Freeze a PF diagnostic or evaluate it externally; never retune external data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_three_population_content import FULL_FEATURES, POOLED_FEATURES


KEYS = ["dataset", "animal", "session", "split", "source", "draw", "event_index"]
METRICS = ["c_support", "ac_regional_tv", "ac_separation_cm", "a_entropy", "c_entropy",
           "a_truth_error_cm", "c_truth_error_cm", "a_spikes", "b_spikes", "c_spikes"]


def equal_animal_weights(frame):
    events = frame.groupby(["animal", "session"])["event_index"].transform("size").to_numpy(float)
    per_animal = frame[["animal", "session"]].drop_duplicates().groupby("animal").size()
    weight = 1 / events / frame.animal.map(per_animal).to_numpy(float)
    return weight / weight.mean()


def train_model(frame, features):
    y = frame.c_support.to_numpy(int)
    if not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid training labels")
    weights = equal_animal_weights(frame)
    prevalence = float(np.average(y, weights=weights))
    if not features or len(np.unique(y)) < 2:
        return dict(kind="constant", prevalence=prevalence, features=list(features),
                    constant_reason="baseline" if not features else "one_training_class")
    x = frame[features].to_numpy(float)
    finite = np.isfinite(x)
    medians = np.array([np.median(x[finite[:, j], j]) if finite[:, j].any() else 0 for j in range(x.shape[1])])
    x = np.where(finite, x, medians)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1
    fitted = LogisticRegression(C=1, solver="lbfgs", max_iter=2000).fit((x-mean)/scale, y, sample_weight=weights)
    if fitted.n_iter_[0] >= 2000:
        raise ValueError("fixed logistic model did not converge")
    return dict(kind="logistic", features=list(features), medians=medians.tolist(), mean=mean.tolist(),
                scale=scale.tolist(), coef=fitted.coef_[0].tolist(), intercept=float(fitted.intercept_[0]),
                training_prevalence=prevalence, regularization_C=1, iterations=int(fitted.n_iter_[0]))


def predict_model(frame, model):
    if model["kind"] == "constant":
        return np.full(len(frame), model["prevalence"], float)
    x = frame[model["features"]].to_numpy(float)
    x = np.where(np.isfinite(x), x, model["medians"])
    return expit(((x-model["mean"])/model["scale"]) @ np.array(model["coef"]) + model["intercept"])


def retained_half(frame, score, higher=True):
    """Only identity and diagnostic scores influence fixed-coverage retention."""
    data = frame[KEYS].copy().reset_index(drop=True)
    if data.duplicated(KEYS).any() or not np.isfinite(score).all():
        raise ValueError("invalid retention inputs")
    data["score"] = np.asarray(score)
    retained = np.zeros(len(data), bool)
    for _, local in data.groupby(KEYS[:-1], sort=False):
        order = local.sort_values(["score", "event_index"], ascending=[not higher, True], kind="stable")
        retained[order.index[:int(np.ceil(len(local)/2))]] = True
    return retained


def apply_models(frame, models):
    out = frame.copy().reset_index(drop=True)
    y = out.c_support.to_numpy(int)
    for name, model in models.items():
        probability = np.clip(predict_model(out, model), 1e-12, 1-1e-12)
        out[f"prediction_{name}"] = probability
        out[f"logloss_{name}"] = -y*np.log(probability)-(1-y)*np.log1p(-probability)
        out[f"brier_{name}"] = (y-probability)**2
    out["retained_full"] = retained_half(out, out.prediction_full.to_numpy())
    out["retained_pooled"] = retained_half(out, out.prediction_pooled.to_numpy())
    out["retained_a_entropy"] = retained_half(out, out.a_entropy.to_numpy(), higher=False)
    out["retained_pooled_entropy"] = retained_half(out, out.pooled_entropy.to_numpy(), higher=False)
    return out


def summary_tables(frame):
    metrics = METRICS + [f"{metric}_{model}" for metric in ("logloss", "brier") for model in ("constant", "pooled", "full")]
    rows = []
    grouping = ["dataset", "animal", "session", "split", "source", "draw"]
    for keys, local in frame.groupby(grouping, sort=True):
        for policy in ("random_expectation", "full", "pooled", "a_entropy", "pooled_entropy"):
            retained = local if policy == "random_expectation" else local.loc[local[f"retained_{policy}"]]
            row = dict(zip(grouping, keys, strict=True), policy=policy, events=len(local), retained=len(retained))
            row.update({col: float(retained[col].mean()) for col in metrics})
            rows.append(row)
    sessions = pd.DataFrame(rows)
    # Average draws within sessions first, never let simulation draws or sessions
    # with more candidates masquerade as additional independent animals.
    session_means = sessions.groupby(grouping[:-1]+["policy"], as_index=False)[metrics].mean()
    animal = session_means.groupby(["dataset", "animal", "split", "source", "policy"], as_index=False)[metrics].mean()
    summary = animal.groupby(["dataset", "split", "source", "policy"], as_index=False)[metrics].mean()
    return sessions, animal, summary


def animal_interval(values, seed=20260914):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def external_gates(sessions, animal, summary, source_catalog, scored_catalog, model_frozen):
    gates = []
    def add(name, ok, value, required):
        gates.append(dict(gate=name, passed=bool(ok), observed=value, required=required))
    known = pd.to_numeric(source_catalog.candidates, errors="coerce")
    evaluated = pd.to_numeric(scored_catalog.candidates, errors="coerce").fillna(0).sum()
    total = known.sum()
    coverage = float(evaluated/total) if total > 0 and known.notna().all() else np.nan
    real = animal.loc[(animal.split == 0) & animal.source.eq("real")]
    n = real.animal.nunique()
    add("external_animals", n >= 4, n, ">=4")
    add("source_endpoint_coverage", np.isfinite(coverage) and coverage >= .8, coverage, ">=0.80; unknown denominators fail")
    add("model_frozen_before_external", model_frozen, model_frozen, "true")
    paired = real.pivot(index="animal", columns="policy")
    primary = summary.loc[(summary.split == 0) & summary.source.eq("real")].set_index("policy")
    if primary.empty or n == 0:
        add("nonempty_external", False, 0, ">0")
    else:
        prediction = primary.loc["random_expectation"]
        for baseline in ("constant", "pooled"):
            reduction = 1-prediction.logloss_full/prediction[f"logloss_{baseline}"]
            add(f"logloss_improves_{baseline}", reduction >= .05, reduction, ">=0.05 relative")
            change = prediction[f"brier_{baseline}"]-prediction.brier_full
            add(f"brier_improves_{baseline}", change > 0, change, ">0")
        differences = paired[("logloss_pooled", "random_expectation")]-paired[("logloss_full", "random_expectation")]
        ci = animal_interval(differences)
        add("prediction_animal_consistency", (differences > 0).mean() >= .75, float((differences > 0).mean()), ">=0.75")
        add("prediction_bootstrap_lower_positive", ci[0] > 0, ci[0], ">0; descriptive animal bootstrap")
        for metric in ("ac_regional_tv", "ac_separation_cm"):
            before, after = primary.loc["random_expectation", metric], primary.loc["full", metric]
            reduction = 1-after/before if before > 0 else np.nan
            difference = paired[(metric, "random_expectation")]-paired[(metric, "full")]
            ci = animal_interval(difference)
            add(f"{metric}_reduction", reduction >= .1, reduction, ">=0.10 relative")
            add(f"{metric}_animal_consistency", (difference > 0).mean() >= .75, float((difference > 0).mean()), ">=0.75")
            add(f"{metric}_bootstrap_lower_positive", ci[0] > 0, ci[0], ">0; descriptive animal bootstrap")
        for side in ("a", "c"):
            change = primary.loc["full", f"{side}_entropy"]-primary.loc["random_expectation", f"{side}_entropy"]
            add(f"{side}_entropy_not_increased", change <= 0, change, "<=0")
    for source in ("run_test", "sim_matched", "sim_drift"):
        local = summary.loc[(summary.split == 0) & summary.source.eq(source)].set_index("policy")
        for side in ("a", "c"):
            metric = f"{side}_truth_error_cm"
            change = float(local.loc["full", metric]-local.loc["random_expectation", metric]) if not local.empty else np.nan
            add(f"{source}_{side}_truth_error_not_worse", np.isfinite(change) and change <= 0, change, "<=0 cm")
    # A separate reconstruction audit must set this, never a producer's self-check.
    add("independent_reconstruction_audit", False, "pending", "independent all-row audit required")
    add("overall", all(g["passed"] for g in gates), "provisional_until_audited", "all required gates")
    return pd.DataFrame(gates)


def load_readouts(root):
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["inputs_unchanged"]:
        raise ValueError("measurement incomplete or inputs changed")
    catalog = pd.read_csv(root/"sessions.csv")
    frames, hashes = [], {}
    for row in catalog.loc[catalog.status.eq("complete")].itertuples(index=False):
        path = Path(row.artifact_dir)/"event_readouts.csv.gz"
        if file_sha256(path) != row.readouts_sha256:
            raise ValueError("readout hash mismatch")
        local = pd.read_csv(path)
        if not local.dataset.eq(row.dataset).all() or not local.animal.eq(row.animal).all() or not local.session.eq(row.session).all():
            raise ValueError("readout identity mismatch")
        frames.append(local)
        hashes[str(path)] = row.readouts_sha256
    if not frames:
        raise ValueError("no complete recording readouts")
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEYS).any():
        raise ValueError("duplicate outcome rows")
    return frame, catalog, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["fit", "external"], required=True)
    parser.add_argument("--frozen-model", type=Path)
    parser.add_argument("--source-catalog", type=Path)
    args = parser.parse_args()
    frame, catalog, hashes = load_readouts(args.measurement_dir)
    inputs = dict(measurement=args.measurement_dir/"manifest.json", catalog=args.measurement_dir/"sessions.csv",
                  producer=Path(__file__), measure_code=ROOT/"scripts/measure_three_population_content.py",
                  protocol=ROOT/"docs/three_population_content_protocol.md", frozen_model=args.frozen_model,
                  source_catalog=args.source_catalog)
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(created_at_utc=datetime.now(UTC).isoformat(), status="running", readout_sha256=hashes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "fit":
        if not frame.dataset.eq("pfeiffer_foster").all():
            raise ValueError("only PF may train frozen models")
        training = frame.loc[(frame.split == 0) & frame.source.eq("real")]
        specifications = dict(constant=[], pooled=POOLED_FEATURES, full=FULL_FEATURES)
        folds = []
        for animal in sorted(training.animal.unique()):
            train = training.loc[training.animal.ne(animal)]
            test = training.loc[training.animal.eq(animal)]
            models = {name: train_model(train, features) for name, features in specifications.items()}
            fold = apply_models(test, models)
            fold["training_animals"] = ";".join(sorted(train.animal.unique()))
            folds.append(fold)
        development = pd.concat(folds, ignore_index=True)
        development.to_csv(args.output_dir/"pf_leave_one_animal_out_predictions.csv.gz", index=False)
        for name, table in zip(("by_session", "by_animal", "summary"), summary_tables(development), strict=True):
            table.to_csv(args.output_dir/f"pf_development_{name}.csv", index=False)
        models = {name: train_model(training, features) for name, features in specifications.items()}
        frozen = dict(status="frozen", created_at_utc=datetime.now(UTC).isoformat(), training_dataset="pfeiffer_foster",
                      training_events=len(training), training_animals=sorted(training.animal.unique()), models=models,
                      input_readout_sha256=hashes, provenance=manifest,
                      outcome="independent_C_mass_within_0.15_grid_diagonal_of_A_mean_ge_0.5",
                      primary_split=0, prediction_inputs="A_and_B_only_C_excluded", diagnostic_retention_fraction=.5,
                      prior_failures="A_only_Tanni_and_balanced_sampling_Blackstad_not_validated")
        (args.output_dir/"frozen_pf_model.json").write_text(json.dumps(frozen, indent=2)+"\n")
        manifest["frozen_model_sha256"] = file_sha256(args.output_dir/"frozen_pf_model.json")
    else:
        if not frame.dataset.eq("autopi_ca1").all() or args.frozen_model is None or args.source_catalog is None:
            raise ValueError("external mode requires AutoPI, frozen PF model and full source denominator")
        frozen = json.loads(args.frozen_model.read_text())
        measurement = json.loads((args.measurement_dir/"manifest.json").read_text())
        froze_before = (measurement["input_file_sha256"].get("frozen_model") == file_sha256(args.frozen_model)
                        and frozen["training_dataset"] == "pfeiffer_foster" and frozen["status"] == "frozen"
                        and frozen["created_at_utc"] < measurement["created_at_utc"])
        if not froze_before:
            raise ValueError("external predictions not protected by PF freeze")
        predicted = apply_models(frame, frozen["models"])
        predicted.to_csv(args.output_dir/"external_predictions.csv.gz", index=False)
        sessions, animal, summary = summary_tables(predicted)
        for name, table in (("by_session", sessions), ("by_animal", animal), ("summary", summary)):
            table.to_csv(args.output_dir/f"external_{name}.csv", index=False)
        gates = external_gates(sessions, animal, summary, pd.read_csv(args.source_catalog), catalog, froze_before)
        gates.to_csv(args.output_dir/"external_gate_summary.csv", index=False)
        manifest["scientific_gate_status"] = "not_validated_audit_pending"
    unchanged = all(file_sha256(path) == manifest["input_file_sha256"][key] for key, path in inputs.items() if path is not None)
    unchanged &= all(file_sha256(path) == value for path, value in hashes.items())
    manifest.update(status="complete" if unchanged else "failed", inputs_unchanged=bool(unchanged),
                    completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("inputs changed during validation")


if __name__ == "__main__":
    main()
