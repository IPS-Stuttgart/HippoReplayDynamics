#!/usr/bin/env python3
"""Rebuild loss-trained choices, training isolation and native endpoint posteriors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import GradientBoostingRegressor

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz, recount
from scripts.audit_encoding_uncertainty_content import check_metrics, h, metrics, scipy_posterior
from scripts.audit_predictive_context_content import dense_log_posteriors
from scripts.audit_temporal_endpoint_content import dense_gaussian, native_context_intervals

METHODS = ("independent", "unconditional_context", "error_trained_context", "entropy_matched")
NAMES = [
    "log_spikes",
    "log_active",
    "log_cells",
    "log_context_bins",
    "active_fraction",
    "independent_entropy",
    "context_entropy",
    "independent_width_relative",
    "context_width_relative",
    "mean_shift_relative",
    "independent_concentration",
    "context_concentration",
]
KEYS = ["animal", "session", "source", "event_index", "side"]


def forest_predict(x, state):
    output = []
    for sample in np.asarray(x, dtype=np.float32):
        prediction = state["initial"]
        for tree in state["trees"]:
            k = 0
            visited = set()
            while tree["left"][k] != -1:
                if k in visited:
                    raise ValueError("cyclic tree")
                visited.add(k)
                k = tree["left"][k] if sample[tree["feature"][k]] <= tree["threshold"][k] else tree["right"][k]
            prediction += state["learning_rate"] * tree["value"][k]
        output.append(prediction)
    return np.asarray(output)


def features_from_metrics(a, b, counts, context_ms, diagonal):
    n = counts.sum(axis=1)
    active = (counts > 0).sum(axis=1)
    return np.column_stack(
        [
            np.log1p(n),
            np.log1p(active),
            np.full(len(n), np.log1p(counts.shape[1])),
            np.log1p(np.asarray(context_ms) / 20),
            active / counts.shape[1],
            a["entropy"],
            b["entropy"],
            a["width"] / diagonal,
            b["width"] / diagonal,
            np.hypot(a["x_cm"] - b["x_cm"], a["y_cm"] - b["y_cm"]) / diagonal,
            np.sum(a["regional"] ** 2, axis=1),
            np.sum(b["regional"] ** 2, axis=1),
        ]
    )


def verify_training(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "frozen" or manifest["features"] != NAMES or not manifest["no_replay_labels"]:
        raise ValueError("wrong fitting contract")
    for key, value in manifest["input_file_paths"].items():
        if file_sha256(value) != manifest["input_file_sha256"][key]:
            raise ValueError("changed fitting source")
    for name, value in manifest["outputs"].items():
        if file_sha256(folder / name) != value:
            raise ValueError("changed fitted output")
    training = pd.read_csv(folder / "training_rows.csv.gz", float_precision="round_trip").sort_values(KEYS).reset_index(drop=True)
    if training.source.eq("real").any() or not training.split.eq(0).all() or training.duplicated(KEYS).any():
        raise ValueError("training leakage or duplication")
    base = json.loads(Path(manifest["input_file_paths"]["base"]).read_text())
    if (
        base["status"] != "complete"
        or {r["dataset"] for r in base["results"]} != {"pfeiffer_foster"}
        or len(base["results"]) != 8
        or len({r["session"] for r in base["results"]}) != 8
        or len({r["animal"] for r in base["results"]}) != 4
    ):
        raise ValueError("full PF training cohort required")
    expected = []
    for row in base["results"]:
        source_folder = Path(row["artifact_dir"])
        frozen = json.loads((source_folder / "frozen_input.json").read_text())
        grid = load_npz(Path(frozen["edge_source"]) / "real_audit.npz")["grid_cm"]
        diagonal = np.linalg.norm(grid.max(axis=0) - grid.min(axis=0))
        frame = pd.read_csv(source_folder / "event_readouts.csv.gz", float_precision="round_trip")
        frame = frame.loc[(frame.split == 0) & (frame.source != "real")]
        if set(frame.source) != set(SOURCES) - {"real"}:
            raise ValueError("missing truth training source")
        for source, local in frame.groupby("source"):
            left = local.loc[local.method.eq("independent")].set_index("event_index").sort_index()
            right = local.loc[local.method.eq("unconditional_context")].set_index("event_index").sort_index()
            assert left.index.equals(right.index)
            for side in ("a", "b"):
                values = np.column_stack(
                    [
                        np.log1p(left[side + "_spikes"]),
                        np.log1p(left[side + "_active"]),
                        np.log1p(left.n_cells_per_group),
                        np.log1p(left.context_ms / 20),
                        left[side + "_active"] / left.n_cells_per_group,
                        left[side + "_entropy"],
                        right[side + "_entropy"],
                        left[side + "_width"] / diagonal,
                        right[side + "_width"] / diagonal,
                        np.hypot(left[side + "_x_cm"] - right[side + "_x_cm"], left[side + "_y_cm"] - right[side + "_y_cm"]) / diagonal,
                        np.square(left[[f"{side}_region{i}" for i in range(9)]].to_numpy()).sum(axis=1),
                        np.square(right[[f"{side}_region{i}" for i in range(9)]].to_numpy()).sum(axis=1),
                    ]
                )
                value = pd.DataFrame(values, columns=NAMES)
                value["event_index"], value["animal"], value["session"], value["source"], value["side"] = left.index, row["animal"], row["session"], source, side
                value["physical_gain"] = (left[side + "_error"].to_numpy() - right[side + "_error"].to_numpy()) / diagonal
                value["brier_gain"] = left[side + "_brier"].to_numpy() - right[side + "_brier"].to_numpy()
                expected.append(value)
    expected = pd.concat(expected).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(training[KEYS], expected[KEYS], check_dtype=False)
    np.testing.assert_allclose(training[NAMES + ["physical_gain", "brier_gain"]], expected[NAMES + ["physical_gain", "brier_gain"]], atol=1e-12, rtol=1e-12)
    models = json.loads((folder / "models.json").read_text())
    if set(models) != set(training.animal.unique()) | {"external"}:
        raise ValueError("missing held-out model")
    for key, state in models.items():
        selected = training if key == "external" else training.loc[training.animal.ne(key)]
        if state["excluded_animal"] != (None if key == "external" else key):
            raise ValueError("wrong training exclusion")
        pd.testing.assert_frame_equal(pd.DataFrame(state["training_ids"]), selected[KEYS].reset_index(drop=True), check_dtype=False)
        assert state["training_animals"] == sorted(selected.animal.unique()) and state["rows"] == len(selected)
        weights = np.empty(len(selected))
        selected = selected.reset_index(drop=True)
        for animal, rat in selected.groupby("animal"):
            for (_, _, _), group in rat.groupby(["session", "source", "side"]):
                weights[group.index] = 1 / (selected.animal.nunique() * rat.session.nunique() * 10 * len(group))
        for loss in ("physical", "brier"):
            state_model = state[loss]
            parameters = state_model["parameters"]
            fixed = dict(loss="squared_error", n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=40, subsample=1.0, random_state=20260914)
            if any(parameters[k] != v for k, v in fixed.items()):
                raise ValueError("unfrozen estimator settings")
            refit = GradientBoostingRegressor(**parameters).fit(selected[NAMES].to_numpy(), selected[loss + "_gain"], sample_weight=weights)
            np.testing.assert_allclose(forest_predict(selected[NAMES], state_model), refit.predict(selected[NAMES].to_numpy()), atol=1e-9, rtol=1e-9)
    return models, len(training)


def verify_session(row, states):
    folder = Path(row["artifact_dir"])
    outputs = json.loads((folder / "outputs.json").read_text())
    if any(file_sha256(folder / key) != value for key, value in outputs.items()):
        raise ValueError("changed readout")
    details = json.loads((folder / "frozen_input.json").read_text())
    base_folder = Path(details["base_folder"])
    if any(file_sha256(base_folder / key) != value for key, value in details["base_outputs"].items()):
        raise ValueError("changed decoder banks")
    freeze = details["original"]["freeze"]
    edge = Path(details["original"]["edge_source"])
    if file_sha256(freeze["encoding_path"]) != freeze["encoding_sha256"] or any(file_sha256(edge / k) != v for k, v in details["original"]["source_outputs"].items()):
        raise ValueError("changed native inputs")
    native = load_npz(freeze["encoding_path"])
    frame = pd.read_csv(folder / "event_readouts.csv.gz", float_precision="round_trip")
    if any(not frame[name].eq(row[name]).all() for name in ("dataset", "animal", "session")):
        raise ValueError("changed session identity")
    if len(freeze["groups"]) != 3 or {p["split"] for p in freeze["groups"]} != {0, 1, 2}:
        raise ValueError("missing cell partitions")
    for part in freeze["groups"]:
        a, b = part["a_ids"], part["b_ids"]
        if not a or len(a) != len(b) or len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b):
            raise ValueError("cell partitions overlap or differ in size")
    if (
        len(frame) != row["rows"]
        or frame.duplicated(["source", "split", "method", "event_index"]).any()
        or set(frame.method) != set(METHODS)
        or set(frame.source) != set(SOURCES)
        or set(frame.split) != {0, 1, 2}
    ):
        raise ValueError("incomplete evaluation")
    key = row["animal"] if row["dataset"] == "pfeiffer_foster" else "external"
    if details["state_key"] != key or not frame.model_key.eq(key).all():
        raise ValueError("wrong prediction model")
    state = states[key]
    if row["dataset"] == "pfeiffer_foster" and row["animal"] in state["training_animals"]:
        raise ValueError("held-out animal leakage")
    checked = native_blocks = posterior_rows = 0
    for source in SOURCES:
        arrays = load_npz(edge / f"{source}_audit.npz")
        grid, rates, ids = arrays["grid_cm"], arrays["rates_hz"], arrays["cell_ids"]
        diagonal = np.linalg.norm(grid.max(axis=0) - grid.min(axis=0))
        matrix = dense_gaussian(grid)
        events, truth, starts, context = [], [], [], []
        for i, (lo, hi) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)):
            n = min(10, int((hi - lo) // 4))
            if n < 1:
                raise ValueError("missing original endpoint support")
            events.append(np.stack([arrays["counts"][k : k + 4].sum(axis=0) for k in range(hi - n * 4, hi, 4)]))
            truth.append(arrays["truth_base_cm"][hi - 4 : hi].mean(axis=0))
            starts.append(arrays["starts_s"][i] + 0.005 * (hi - lo - 4))
            context.append(20 * n)
            if source in ("real", "run_q4"):
                left, right = native_context_intervals(arrays, i, native, source)
                np.testing.assert_array_equal(events[-1], recount(native["spikes"], ids, left, right))
                native_blocks += n
        truth = np.asarray(truth)
        for part in freeze["groups"]:
            split = part["split"]
            saved = load_npz(folder / f"{source}_split{split}_audit.npz")
            np.testing.assert_array_equal(saved["event_ids"], arrays["event_ids"])
            view = frame.loc[(frame.source == source) & (frame.split == split)]
            reference = view.loc[view.method.eq("independent")].set_index("event_index").loc[arrays["event_ids"]]
            sides = []
            for side in ("a", "b"):
                lookup = {int(cell): k for k, cell in enumerate(ids)}
                group = np.array([lookup[int(cell)] for cell in part[side + "_ids"]])
                counts = np.stack([event[-1, group] for event in events])
                lp = dense_log_posteriors([event[:, group] for event in events], rates[group], matrix)
                independent, temporal = np.exp(lp[:, 0]), np.exp(lp[:, 1])
                ma, mb = metrics(independent, grid, truth), metrics(temporal, grid, truth)
                x = features_from_metrics(ma, mb, counts, context, diagonal)
                np.testing.assert_allclose(saved[side + "_features"], x, atol=1e-8, rtol=1e-8)
                # Use the verified saved float64 features for exact tree boundary choices.
                physical = forest_predict(saved[side + "_features"], state["physical"])
                brier = forest_predict(saved[side + "_features"], state["brier"])
                use = (physical > 1e-10) & (brier > 1e-10) & (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)
                chosen = np.where(use[:, None], temporal, independent)
                _, ll = scipy_posterior(counts, rates[group])
                temperature = reference[side + "_matched_log_temperature"].to_numpy()
                if np.any(np.abs(temperature) > 20):
                    raise ValueError("invalid entropy matching bracket")
                w = (ll - ll.max(axis=1, keepdims=True)) / np.exp(temperature)[:, None]
                matched = np.exp(w - logsumexp(w, axis=1, keepdims=True))
                for name, expected in (
                    ("predicted_physical_gain", physical),
                    ("predicted_brier_gain", brier),
                    ("spikes", counts.sum(axis=1)),
                    ("active", (counts > 0).sum(axis=1)),
                ):
                    np.testing.assert_allclose(reference[side + "_" + name], expected, atol=1e-10, rtol=1e-10)
                np.testing.assert_array_equal(reference[side + "_use_context"], use)
                np.testing.assert_array_equal(reference[side + "_entropy_control_available"], np.abs(h(matched) - h(chosen)) <= 1e-7)
                bank = dict(zip(METHODS, (independent, temporal, chosen, matched), strict=True))
                for method, posterior in bank.items():
                    np.testing.assert_allclose(saved[f"{side}_{method}_posterior"], posterior, atol=5e-9, rtol=2e-8)
                    posterior_rows += len(posterior)
                sides.append(bank)
            for method in METHODS:
                selected = view.loc[view.method.eq(method)].set_index("event_index").loc[arrays["event_ids"]]
                if len(selected) != len(reference):
                    raise ValueError("changed cohort")
                for clock, expected in (("original_start_s", starts), ("original_end_s", np.asarray(starts) + 0.02), ("context_ms", context)):
                    np.testing.assert_allclose(selected[clock], expected, atol=1e-10, rtol=0)
                for side in ("a", "b"):
                    for field in ("predicted_physical_gain", "predicted_brier_gain", "use_context", "spikes", "active", "entropy_control_available", "matched_log_temperature"):
                        np.testing.assert_array_equal(selected[side + "_" + field], reference[side + "_" + field])
                check_metrics(selected, sides[0][method], sides[1][method], grid, truth)
                checked += len(selected)
    if checked != len(frame):
        raise ValueError("unaudited observations")
    return dict(
        dataset=row["dataset"],
        animal=row["animal"],
        session=row["session"],
        status="passed",
        rows=checked,
        posterior_rows=posterior_rows,
        native_context_blocks=native_blocks,
        readout_sha256=file_sha256(folder / "event_readouts.csv.gz"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, required=True)
    p.add_argument("--frozen-models", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    inputs = dict(
        producer=args.measurement_dir / "manifest.json",
        frozen=args.frozen_models / "manifest.json",
        models=args.frozen_models / "models.json",
        auditor=Path(__file__),
        dense=ROOT / "scripts/audit_predictive_context_content.py",
        metrics=ROOT / "scripts/audit_encoding_uncertainty_content.py",
        transition=ROOT / "scripts/audit_temporal_endpoint_content.py",
    )
    producer = json.loads(inputs["producer"].read_text())
    if producer["status"] != "complete" or producer["input_file_sha256"]["frozen"] != file_sha256(inputs["frozen"]):
        raise ValueError("incomplete or mismatched measurement")
    if len(producer["results"]) != 8 or len({r["animal"] for r in producer["results"]}) != 4:
        raise ValueError("incomplete cohort")
    for key, path in producer["input_file_paths"].items():
        if file_sha256(path) != producer["input_file_sha256"][key]:
            raise ValueError("changed producer input")
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    states, ntraining = verify_training(args.frozen_models)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for row in producer["results"]:
        results.append(verify_session(row, states))
        pd.DataFrame(results).to_csv(args.output_dir / "independent_audit_sessions.csv", index=False)
        print(json.dumps(results[-1]), flush=True)
    if any(file_sha256(path) != provenance["input_file_sha256"][key] for key, path in inputs.items()):
        raise ValueError("changed audit inputs")
    provenance.update(status="passed", training_rows_rebuilt=ntraining, all_fits_refit_without_heldout_animals=True, results=results)
    (args.output_dir / "independent_audit.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
