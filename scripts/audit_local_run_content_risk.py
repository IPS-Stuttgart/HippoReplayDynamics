#!/usr/bin/env python3
"""Independently reconstruct local RUN-risk fits, selections, and summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256


def matrix(frame, side, rates, grid, columns):
    xy = frame[[side + "_x_cm", side + "_y_cm"]].to_numpy()
    nearest = np.argmin(((xy[:, None] - grid[None]) ** 2).sum(axis=2), axis=1)
    diagonal2 = np.sum(np.ptp(grid, axis=0) ** 2)
    coverage = rates.sum(axis=0)
    gradient = np.zeros(len(grid))
    for index, point in enumerate(grid):
        distance2 = np.sum((grid - point) ** 2, axis=1)
        use = (distance2 > 0) & (distance2 <= 8.01**2)
        if use.any():
            contrast = np.sum((np.sqrt(rates[:, use]) - np.sqrt(rates[:, index : index + 1])) ** 2, axis=0)
            gradient[index] = np.mean(contrast / distance2[use])
    value = {
        "log_spikes": np.log1p(frame[side + "_spikes"].to_numpy()),
        "log_active": np.log1p(frame[side + "_active"].to_numpy()),
        "entropy": frame[side + "_entropy"].to_numpy(),
        "relative_width": frame[side + "_width_cm"].to_numpy() / np.sqrt(diagonal2),
        "relative_x": (xy[:, 0] - grid[:, 0].mean()) / np.sqrt(diagonal2),
        "relative_y": (xy[:, 1] - grid[:, 1].mean()) / np.sqrt(diagonal2),
        "coverage": np.log1p(coverage[nearest] / coverage.mean()),
        "code_gradient": np.log1p(0.02 * diagonal2 * gradient[nearest]),
    }
    return np.column_stack([value[c] for c in columns]) if columns else np.empty((len(frame), 0))


def dense_calibration(arrays, indices, side):
    counts, rates, grid = arrays["calibration_counts"][:, indices], arrays["rates_hz"][indices], arrays["grid_cm"]
    ll = counts @ np.log(rates) - 0.02 * rates.sum(axis=0)
    probability = np.exp(ll - ll.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    xy = probability @ grid
    width = np.sqrt(np.maximum(np.sum(probability * np.sum(grid**2, axis=1), axis=1) - np.sum(xy**2, axis=1), 0))
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-300)), axis=1) / np.log(len(grid))
    frame = pd.DataFrame(
        {
            side + "_x_cm": xy[:, 0],
            side + "_y_cm": xy[:, 1],
            side + "_spikes": counts.sum(axis=1),
            side + "_active": (counts > 0).sum(axis=1),
            side + "_entropy": entropy,
            side + "_width_cm": width,
        }
    )
    target = np.log1p(np.linalg.norm(xy - arrays["calibration_windows"][:, 2:], axis=1) / np.linalg.norm(np.ptp(grid, axis=0)))
    return frame, target


def verify_fit(x, y, state):
    np.testing.assert_allclose(state["intercept"], np.mean(y), atol=1e-10, rtol=1e-10)
    if not state["columns"]:
        return
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + 10 * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    np.testing.assert_allclose(state["means"], mean, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(state["scales"], scale, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(state["coefficients"], coef, atol=1e-9, rtol=1e-9)


def check(args):
    models = json.loads(args.frozen_models.read_text())
    manifest = json.loads((args.result_dir / "manifest.json").read_text())
    for document in (models, manifest):
        for name, path in document["input_file_paths"].items():
            assert file_sha256(path) == document["input_file_sha256"][name], name
    for name, digest in manifest["output_sha256"].items():
        assert file_sha256(args.result_dir / name) == digest, name
    source = json.loads(Path(models["input_file_paths"]["source"]).read_text())
    fit_count, prediction_count, case_count = 0, 0, 0
    expected_cases = pd.read_csv(args.result_dir / "case_metrics.csv")
    recomputed = []
    key = ["source", "draw", "split", "event_index"]
    for session in source["results"]:
        folder = Path(session["artifact_dir"])
        with np.load(folder / "audit_arrays.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        parts = json.loads((folder / "frozen_populations.json").read_text())["parts"]
        original = pd.read_csv(folder / "event_readouts.csv.gz").sort_values(key).reset_index(drop=True)
        result = pd.read_csv(args.result_dir / (session["session"].replace("/", "_") + ".csv.gz")).sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(original, result[original.columns], check_dtype=False)
        for part in parts:
            split, grid = part["split"], arrays["grid_cm"]
            predictions = {m: [] for m in ("mean", "spikes_entropy", "full")}
            evaluated = result.loc[result.split.eq(split)]
            for side in ("a", "b"):
                state = models["states"][session["session"]][str(split)][side]
                cal, y = dense_calibration(arrays, part[side], side)
                rates = arrays["rates_hz"][part[side]]
                for model, parameters in state.items():
                    verify_fit(matrix(cal, side, rates, grid, parameters["columns"]), y, parameters)
                    fit_count += 1
                    x = matrix(evaluated, side, rates, grid, parameters["columns"])
                    pred = (
                        np.maximum(0, ((x - parameters["means"]) / parameters["scales"]) @ parameters["coefficients"] + parameters["intercept"])
                        if parameters["columns"]
                        else np.full(len(x), parameters["intercept"])
                    )
                    predictions[model].append(pred)
            for model, values in predictions.items():
                np.testing.assert_allclose(evaluated["risk_" + model], np.mean(values, axis=0), atol=2e-10, rtol=2e-10)
                prediction_count += len(evaluated)
            for (source_name, draw), case in evaluated.groupby(["source", "draw"]):
                for model in ("spikes_entropy", "full"):
                    chosen_ids = case.event_index.to_numpy()[np.lexsort((case.event_index, case["risk_" + model]))[: int(np.ceil(len(case) / 2))]]
                    mask = case.event_index.isin(chosen_ids)
                    np.testing.assert_array_equal(case["selected_" + model], mask)
                    chosen = case.loc[mask]
                    measure = expected_cases.loc[
                        expected_cases.session.eq(session["session"])
                        & expected_cases.split.eq(split)
                        & expected_cases.source.eq(source_name)
                        & expected_cases.draw.eq(draw)
                        & expected_cases.model.eq(model)
                    ]
                    for row in measure.itertuples():
                        baseline = case.mean_truth_error_cm.mean() if row.metric == "true_tile_reweighted_error_cm" else case[row.metric].mean()
                        if row.metric == "true_tile_reweighted_error_cm":
                            tiles, counts = np.unique(case.true_tile, return_counts=True)
                            means = np.array([chosen.loc[chosen.true_tile.eq(t), "mean_truth_error_cm"].mean() for t in tiles])
                            selected = counts @ means / len(case)
                        else:
                            selected = chosen[row.metric].mean()
                        np.testing.assert_allclose([row.baseline, row.selected, row.reduction], [baseline, selected, baseline - selected], atol=1e-10, rtol=1e-10)
                        assert row.n_events == len(case) and row.n_selected == len(chosen)
                        if split == 0:
                            recomputed.append(
                                {
                                    "session": session["session"],
                                    "animal": session["animal"],
                                    "model": model,
                                    "source": source_name,
                                    "metric": row.metric,
                                    "baseline": baseline,
                                    "selected": selected,
                                }
                            )
                    case_count += 1
    cases = pd.DataFrame(recomputed)
    per_session = cases.groupby(["model", "source", "metric", "animal", "session"])[["baseline", "selected"]].mean()
    per_animal = per_session.groupby(["model", "source", "metric", "animal"]).mean()
    pooled = per_animal.groupby(["model", "source", "metric"]).mean()
    reported = pd.read_csv(args.result_dir / "development_summary.csv").set_index(["model", "source", "metric"])
    np.testing.assert_allclose(pooled.loc[reported.index], reported[["baseline", "selected"]], atol=1e-10, rtol=1e-10)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = build_script_provenance(input_paths={"models": args.frozen_models, "result": args.result_dir / "manifest.json", "script": Path(__file__)}, cwd=ROOT)
    audit.update(
        status="passed",
        independently_reconstructed_fits=fit_count,
        prediction_checks=prediction_count,
        cases=case_count,
        result_groups=len(pooled),
        external_validation=False,
        scientific_goal_achieved=False,
    )
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "passed", "fits": fit_count, "predictions": prediction_count, "cases": case_count}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-models", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    check(p.parse_args())


if __name__ == "__main__":
    main()
