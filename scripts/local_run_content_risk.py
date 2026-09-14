#!/usr/bin/env python3
"""Freeze recording-local RUN risk, then assess unchanged population readouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_population_content_stability import decode
from scripts.run_error_content_diagnostic import add_truth_tiles, code_gradient, session_parts, verified_sessions

BASE = ("log_spikes", "log_active", "entropy")
FULL = (*BASE, "relative_width", "relative_x", "relative_y", "coverage", "code_gradient")
MODELS = ("mean", "spikes_entropy", "full")


def features(frame, side, rates, grid):
    diagonal = float(np.linalg.norm(np.ptp(grid, axis=0)))
    xy = frame[[side + "_x_cm", side + "_y_cm"]].to_numpy(float)
    nearest = cKDTree(grid).query(xy)[1]
    coverage = rates.sum(axis=0)
    gradient = code_gradient(rates, grid)
    scaled = (xy - grid.mean(axis=0)) / diagonal
    return pd.DataFrame(
        {
            "log_spikes": np.log1p(frame[side + "_spikes"]),
            "log_active": np.log1p(frame[side + "_active"]),
            "entropy": frame[side + "_entropy"],
            "relative_width": frame[side + "_width_cm"] / diagonal,
            "relative_x": scaled[:, 0],
            "relative_y": scaled[:, 1],
            "coverage": np.log1p(coverage[nearest] / coverage.mean()),
            "code_gradient": np.log1p(0.02 * diagonal**2 * gradient[nearest]),
        },
        index=frame.index,
    )


def fit(x, known_error, diagonal, model):
    y = np.log1p(np.asarray(known_error) / diagonal)
    if len(y) < 100 or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("at least 100 finite known RUN errors required")
    columns = () if model == "mean" else BASE if model == "spikes_entropy" else FULL if model == "full" else None
    if columns is None:
        raise ValueError("unknown model")
    if not columns:
        return {"columns": [], "intercept": float(y.mean())}
    data = x[list(columns)].to_numpy(float)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite features")
    mean, scale = data.mean(axis=0), data.std(axis=0)
    scale[scale < 1e-10] = 1
    estimator = Ridge(alpha=10).fit((data - mean) / scale, y)
    return {"columns": list(columns), "means": mean.tolist(), "scales": scale.tolist(), "coefficients": estimator.coef_.tolist(), "intercept": float(estimator.intercept_)}


def predict(x, state):
    if not state["columns"]:
        return np.full(len(x), state["intercept"])
    values = np.ascontiguousarray(x[state["columns"]].to_numpy(float))
    return np.maximum(0, ((values - state["means"]) / state["scales"]) @ state["coefficients"] + state["intercept"])


def calibration(arrays, part, side):
    counts = arrays["calibration_counts"][:, part[side]]
    rates, grid = arrays["rates_hz"][part[side]], arrays["grid_cm"]
    windows = arrays["calibration_windows"]
    result = decode(counts, rates, grid)
    frame = pd.DataFrame(
        {
            side + "_x_cm": result["mean"][:, 0],
            side + "_y_cm": result["mean"][:, 1],
            side + "_spikes": counts.sum(axis=1),
            side + "_active": np.count_nonzero(counts, axis=1),
            side + "_entropy": result["entropy"],
            side + "_width_cm": result["width"],
        }
    )
    error = np.linalg.norm(result["mean"] - windows[:, 2:], axis=1)
    return features(frame, side, rates, grid), error


def freeze(args):
    sessions = verified_sessions(args.input_dir, "pfeiffer_foster")
    if len(sessions) != 8 or len({r["animal"] for r in sessions}) != 4:
        raise ValueError("full PF cohort required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        "source": args.input_dir / "manifest.json",
        "script": Path(__file__),
        "protocol": ROOT / "docs/local_run_content_risk_protocol.md",
        "decoder": ROOT / "scripts/measure_population_content_stability.py",
        "helpers": ROOT / "scripts/run_error_content_diagnostic.py",
    }
    for row in sessions:
        for name in ("audit_arrays.npz", "frozen_populations.json"):
            inputs[row["session"] + "/" + name] = Path(row["artifact_dir"]) / name
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    states = {}
    for row in sessions:
        arrays, frozen = session_parts(row)
        diagonal = np.linalg.norm(np.ptp(arrays["grid_cm"], axis=0))
        states[row["session"]] = {}
        for part in frozen["parts"]:
            states[row["session"]][str(part["split"])] = {}
            for side in ("a", "b"):
                x, y = calibration(arrays, part, side)
                states[row["session"]][str(part["split"])][side] = {model: fit(x, y, diagonal, model) for model in MODELS}
    for name, path in inputs.items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError("training inputs changed")
    manifest.update(
        status="frozen", states=states, training="recording_local_third_quarter_RUN_20ms", retention=0.5, replay_inputs_used_for_training=False, scientific_goal_achieved=False
    )
    (args.output_dir / "frozen_models.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "frozen", "sessions": len(states)}), flush=True)


def retain(frame, score):
    if frame.empty or frame.event_index.duplicated().any():
        raise ValueError("empty or duplicate selection IDs")
    if not np.isfinite(frame[score]).all():
        raise ValueError("nonfinite selection risk")
    return frame.sort_values([score, "event_index"], kind="stable").head(int(np.ceil(len(frame) / 2)))


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def evaluate_case(case, model):
    risk = "risk_" + model
    chosen = retain(case, risk)
    rows = []
    measures = ["endpoint_separation_cm", "regional_tv", "a_entropy", "b_entropy"]
    if case.source.iloc[0] != "real":
        measures += ["a_truth_error_cm", "b_truth_error_cm", "mean_truth_error_cm"]
    for measure in measures:
        original, selected = float(case[measure].mean()), float(chosen[measure].mean())
        rows.append(
            {
                "metric": measure,
                "baseline": original,
                "selected": selected,
                "reduction": original - selected,
                "relative_reduction": (original - selected) / original if original > 0 else np.nan,
                "risk_correlation": correlation(case[risk], case[measure]),
                "n_events": len(case),
                "n_selected": len(chosen),
            }
        )
    if case.source.iloc[0] != "real":
        original_by_tile = case.groupby("true_tile").mean_truth_error_cm.agg(["size", "mean"])
        selected_by_tile = chosen.groupby("true_tile").mean_truth_error_cm.mean().reindex(original_by_tile.index)
        weights = original_by_tile["size"] / len(case)
        value = float(weights @ selected_by_tile) if selected_by_tile.notna().all() else np.nan
        rows.append(
            {
                "metric": "true_tile_reweighted_error_cm",
                "baseline": float(case.mean_truth_error_cm.mean()),
                "selected": value,
                "reduction": float(case.mean_truth_error_cm.mean()) - value,
                "relative_reduction": np.nan,
                "risk_correlation": np.nan,
                "n_events": len(case),
                "n_selected": len(chosen),
                "original_tiles": len(original_by_tile),
                "selected_tiles": int(selected_by_tile.notna().sum()),
            }
        )
    return rows


def summary(tables, output):
    primary = tables.loc[tables.split.eq(0)]
    numerical = ["baseline", "selected", "reduction", "relative_reduction", "risk_correlation"]
    session = primary.groupby(["model", "source", "metric", "animal", "session"], as_index=False)[numerical].mean()
    animal = session.groupby(["model", "source", "metric", "animal"], as_index=False)[numerical].mean()
    animal.to_csv(output / "by_animal.csv", index=False)
    records = []
    for values, local in animal.groupby(["model", "source", "metric"]):
        rng = np.random.default_rng(20260914)
        reduction = local.reduction.to_numpy()
        ci = np.quantile(rng.choice(reduction, (5000, len(local)), replace=True).mean(axis=1), [0.025, 0.975])
        records.append(
            dict(zip(["model", "source", "metric"], values, strict=True))
            | {
                "baseline": local.baseline.mean(),
                "selected": local.selected.mean(),
                "reduction": local.reduction.mean(),
                "relative_reduction": local.reduction.mean() / local.baseline.mean() if local.baseline.mean() > 0 else np.nan,
                "reduction_ci_low": ci[0],
                "reduction_ci_high": ci[1],
                "animals_improved": int(local.reduction.gt(0).sum()),
                "animals": len(local),
                "animals_positive_risk_correlation": int(local.risk_correlation.gt(0).sum()),
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(output / "development_summary.csv", index=False)
    gates = []
    for model in ("spikes_entropy", "full"):
        scores = result.loc[result.model.eq(model)].set_index(["source", "metric"])
        for metric in ("endpoint_separation_cm", "regional_tv"):
            item = scores.loc[("real", metric)]
            gates.append(
                {
                    "model": model,
                    "gate": "real_reduction_" + metric,
                    "pass": bool(item.animals == 4 and item.relative_reduction >= 0.1 and item.animals_improved >= 3 and item.reduction_ci_low > 0),
                }
            )
        for metric in ("a_entropy", "b_entropy"):
            gates.append({"model": model, "gate": "no_increased_" + metric, "pass": bool(scores.loc[("real", metric), "reduction"] >= 0)})
        for source in ("run_test", "sim_matched", "sim_drift"):
            item = scores.loc[(source, "mean_truth_error_cm")]
            gates.append({"model": model, "gate": source + "_risk_predicts_error", "pass": bool(item.animals_positive_risk_correlation >= 3 and item.animals == 4)})
            for metric in ("a_truth_error_cm", "b_truth_error_cm", "mean_truth_error_cm", "true_tile_reweighted_error_cm"):
                raw = primary.loc[primary.model.eq(model) & primary.source.eq(source) & primary.metric.eq(metric)]
                gates.append(
                    {
                        "model": model,
                        "gate": source + "_no_worse_" + metric,
                        "pass": bool(raw.selected.notna().all() and len(raw) > 0 and scores.loc[(source, metric), "reduction"] >= 0),
                    }
                )
        gates.append({"model": model, "gate": "PF_development_advance", "pass": all(r["pass"] for r in gates if r["model"] == model)})
    pd.DataFrame(gates).to_csv(output / "development_gates.csv", index=False)
    return gates


def apply(args):
    frozen = json.loads(args.frozen_models.read_text())
    if frozen["status"] != "frozen":
        raise ValueError("models must be frozen")
    for name, path in frozen["input_file_paths"].items():
        if file_sha256(path) != frozen["input_file_sha256"][name]:
            raise ValueError("frozen inputs changed")
    sessions = verified_sessions(args.input_dir, "pfeiffer_foster")
    if {r["session"] for r in sessions} != set(frozen["states"]):
        raise ValueError("frozen session cohort mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {"models": args.frozen_models, "source": args.input_dir / "manifest.json", "script": Path(__file__)}
    for row in sessions:
        inputs[row["session"]] = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    tables = []
    for row in sessions:
        arrays, partition = session_parts(row)
        data = pd.read_csv(Path(row["artifact_dir"]) / "event_readouts.csv.gz")
        pieces = []
        for part in partition["parts"]:
            frame = data.loc[data.split.eq(part["split"])].copy()
            states = frozen["states"][row["session"]][str(part["split"])]
            side_features = {side: features(frame, side, arrays["rates_hz"][part[side]], arrays["grid_cm"]) for side in ("a", "b")}
            for model in MODELS:
                frame["risk_" + model] = sum(predict(side_features[side], states[side][model]) for side in ("a", "b")) / 2
            frame["mean_truth_error_cm"] = (frame.a_truth_error_cm + frame.b_truth_error_cm) / 2
            for (source, draw), case in frame.groupby(["source", "draw"]):
                case = case.copy()
                case["true_tile"] = add_truth_tiles(case, arrays, part, row, partition)
                for model in ("spikes_entropy", "full"):
                    selected_ids = set(retain(case, "risk_" + model).event_index)
                    case["selected_" + model] = case.event_index.isin(selected_ids)
                    tables.extend(
                        dict(r, model=model, source=source, draw=draw, split=part["split"], animal=row["animal"], session=row["session"]) for r in evaluate_case(case, model)
                    )
                pieces.append(case)
        name = row["session"].replace("/", "_") + ".csv.gz"
        pd.concat(pieces, ignore_index=True).to_csv(args.output_dir / name, index=False)
        print(json.dumps({"session": row["session"], "status": "complete"}), flush=True)
    table = pd.DataFrame(tables)
    table.to_csv(args.output_dir / "case_metrics.csv", index=False)
    gates = summary(table, args.output_dir)
    for name, path in inputs.items():
        if file_sha256(path) != provenance["input_file_sha256"][name]:
            raise ValueError("inputs changed")
    provenance.update(
        status="complete",
        inputs_unchanged=True,
        cohort="PF_development_only",
        independent_validation=False,
        scientific_goal_achieved=False,
        methods_passing_development=[r["model"] for r in gates if r["gate"] == "PF_development_advance" and r["pass"]],
        output_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "apply"):
        p = commands.add_parser(name)
        p.add_argument("--input-dir", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        if name == "apply":
            p.add_argument("--frozen-models", type=Path, required=True)
    args = parser.parse_args()
    freeze(args) if args.command == "freeze" else apply(args)


if __name__ == "__main__":
    main()
