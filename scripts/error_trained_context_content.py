#!/usr/bin/env python3
"""Freeze loss-trained context gates, then evaluate fixed population endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import SOURCES, load_npz
from scripts.measure_encoding_uncertainty_content import entropy, log_predictive, match_entropy, posterior_metrics

PRIMARY = "error_trained_context"
METHODS = ("independent", "unconditional_context", PRIMARY, "entropy_matched")
FEATURES = (
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
)
PARAMETERS = dict(loss="squared_error", n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=40, subsample=1.0, random_state=20260914)
ID = ["animal", "session", "source", "event_index", "side"]


def features(independent, context, side, diagonal):
    if side not in ("a", "b") or not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("own population and finite positive diagonal required")
    if not independent.index.equals(context.index):
        raise ValueError("unmatched events")
    x = independent
    shift = np.hypot(x[side + "_x_cm"] - context[side + "_x_cm"], x[side + "_y_cm"] - context[side + "_y_cm"])
    cells = x.n_cells_per_group.to_numpy()
    if np.any(cells <= 0):
        raise ValueError("no cells")
    result = pd.DataFrame(
        dict(
            log_spikes=np.log1p(x[side + "_spikes"]),
            log_active=np.log1p(x[side + "_active"]),
            log_cells=np.log1p(cells),
            log_context_bins=np.log1p(x.context_ms / 20),
            active_fraction=x[side + "_active"] / cells,
            independent_entropy=x[side + "_entropy"],
            context_entropy=context[side + "_entropy"],
            independent_width_relative=x[side + "_width"] / diagonal,
            context_width_relative=context[side + "_width"] / diagonal,
            mean_shift_relative=shift / diagonal,
            independent_concentration=(x[[f"{side}_region{k}" for k in range(9)]] ** 2).sum(axis=1),
            context_concentration=(context[[f"{side}_region{k}" for k in range(9)]] ** 2).sum(axis=1),
        ),
        index=x.index,
    )[list(FEATURES)]
    if not np.isfinite(result).all().all():
        raise ValueError("nonfinite predictor features")
    return result


def predict_tree_model(x, model):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("invalid prediction features")
    out = np.full(len(x), model["initial"], dtype=float)
    for tree in model["trees"]:
        left, right, variable = (np.asarray(tree[k], int) for k in ("left", "right", "feature"))
        threshold, values = (np.asarray(tree[k], float) for k in ("threshold", "value"))
        node = np.zeros(len(x), int)
        for _ in range(len(left)):
            active = np.flatnonzero(left[node] >= 0)
            if not len(active):
                break
            old = node[active]
            node[active] = np.where(x[active, variable[old]] <= threshold[old], left[old], right[old])
        else:
            raise ValueError("cyclic tree")
        out += model["learning_rate"] * values[node]
    return out


def fit_model(x, y, weights):
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    if len(x) == 0 or len(y) != len(x) or len(weights) != len(x) or not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("finite weighted training observations required")
    estimator = GradientBoostingRegressor(**PARAMETERS).fit(x, y, sample_weight=weights)
    trees = []
    for fitted in estimator.estimators_[:, 0]:
        t = fitted.tree_
        trees.append(
            dict(left=t.children_left.tolist(), right=t.children_right.tolist(), feature=t.feature.tolist(), threshold=t.threshold.tolist(), value=t.value[:, 0, 0].tolist())
        )
    model = dict(initial=float(estimator.init_.constant_.ravel()[0]), learning_rate=estimator.learning_rate, trees=trees, parameters=estimator.get_params())
    np.testing.assert_allclose(predict_tree_model(x, model), estimator.predict(x), atol=1e-12, rtol=1e-12)
    return model


def training_weights(frame):
    if frame.empty or frame.duplicated(ID).any() or frame.source.eq("real").any() or not frame.split.eq(0).all():
        raise ValueError("unique known-truth primary-split training rows required")
    if set(frame.source) != set(SOURCES) - {"real"} or set(frame.side) != {"a", "b"}:
        raise ValueError("all truth sources and population sides required")
    count = frame.groupby(["animal", "session", "source", "side"]).event_index.transform("size")
    nsessions = frame[["animal", "session"]].drop_duplicates().groupby("animal").size()
    for (_, _), group in frame.groupby(["animal", "session"]):
        if len(group[["source", "side"]].drop_duplicates()) != 10:
            raise ValueError("missing session training source")
    return (1 / (count * frame.animal.map(nsessions) * frame.animal.nunique() * 10)).to_numpy()


def fit_states(training):
    if training.animal.nunique() != 4 or training.session.nunique() != 8 or len(training) < 1000:
        raise ValueError("full PF known-truth training cohort required")
    models = {}
    for excluded in [*sorted(training.animal.unique()), "external"]:
        selected = training if excluded == "external" else training.loc[training.animal.ne(excluded)]
        selected = selected.sort_values(ID).reset_index(drop=True)
        weights = training_weights(selected)
        models[excluded] = dict(
            excluded_animal=None if excluded == "external" else excluded,
            training_ids=selected[ID].to_dict("records"),
            training_animals=sorted(selected.animal.unique()),
            rows=len(selected),
            physical=fit_model(selected[list(FEATURES)], selected.physical_gain, weights),
            brier=fit_model(selected[list(FEATURES)], selected.brier_gain, weights),
        )
    return models


def choose(independent, physical, brier, side):
    physical, brier = np.asarray(physical), np.asarray(brier)
    if physical.shape != (len(independent),) or brier.shape != physical.shape or not np.isfinite(physical).all() or not np.isfinite(brier).all():
        raise ValueError("aligned finite loss forecasts required")
    return (physical > 1e-10) & (brier > 1e-10) & (independent[side + "_spikes"].to_numpy() >= 3) & (independent[side + "_active"].to_numpy() >= 2)


def verified_base(folder, audit_dir):
    producer_path, audit_path = folder / "manifest.json", audit_dir / "independent_audit.json"
    producer, audit = (json.loads(p.read_text()) for p in (producer_path, audit_path))
    if producer["status"] != "complete" or audit["status"] != "passed" or audit["input_file_sha256"]["producer"] != file_sha256(producer_path):
        raise ValueError("complete matching independently verified base required")
    for manifest in (producer, audit):
        for key, path in manifest["input_file_paths"].items():
            if file_sha256(path) != manifest["input_file_sha256"][key]:
                raise ValueError("changed base provenance")
    if (
        len(producer["results"]) != 8
        or len({r["animal"] for r in producer["results"]}) != 4
        or len({(r["animal"], r["session"]) for r in producer["results"]}) != 8
        or len({r["dataset"] for r in producer["results"]}) != 1
    ):
        raise ValueError("incomplete base cohort")
    for row in producer["results"]:
        path = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
        matches = [r for r in audit["results"] if r["animal"] == row["animal"] and r["session"] == row["session"]]
        if len(matches) != 1 or matches[0]["readout_sha256"] != file_sha256(path):
            raise ValueError("unverified base readout")
    return producer


def source_grid(folder):
    frozen = json.loads((folder / "frozen_input.json").read_text())
    edge = Path(frozen["edge_source"])
    if any(file_sha256(edge / name) != value for name, value in frozen["source_outputs"].items()):
        raise ValueError("changed original source")
    return load_npz(edge / "real_audit.npz"), frozen


def training_rows(producer):
    rows = []
    for row in producer["results"]:
        if row["dataset"] != "pfeiffer_foster":
            raise ValueError("training is PF only")
        folder = Path(row["artifact_dir"])
        arrays, _ = source_grid(folder)
        diagonal = np.linalg.norm(np.ptp(arrays["grid_cm"], axis=0))
        full = pd.read_csv(folder / "event_readouts.csv.gz", float_precision="round_trip")
        full = full.loc[full.split.eq(0) & full.source.ne("real")]
        for source, group in full.groupby("source"):
            baseline = group.loc[group.method.eq("independent")].set_index("event_index").sort_index()
            context = group.loc[group.method.eq("unconditional_context")].set_index("event_index").sort_index()
            for side in ("a", "b"):
                frame = features(baseline, context, side, diagonal)
                frame["physical_gain"] = (baseline[side + "_error"] - context[side + "_error"]) / diagonal
                frame["brier_gain"] = baseline[side + "_brier"] - context[side + "_brier"]
                frame["animal"], frame["session"], frame["source"], frame["side"], frame["split"] = row["animal"], row["session"], source, side, 0
                rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True).sort_values(ID).reset_index(drop=True)


def freeze(args):
    producer = verified_base(args.base_dir, args.base_audit)
    inputs = dict(
        base=args.base_dir / "manifest.json", audit=args.base_audit / "independent_audit.json", script=Path(__file__), protocol=ROOT / "docs/error_trained_context_protocol.md"
    )
    for row in producer["results"]:
        inputs[row["session"]] = Path(row["artifact_dir"]) / "event_readouts.csv.gz"
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    training = training_rows(producer)
    states = fit_states(training)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    training.to_csv(args.output_dir / "training_rows.csv.gz", index=False)
    (args.output_dir / "models.json").write_text(json.dumps(states) + "\n")
    if any(file_sha256(path) != manifest["input_file_sha256"][key] for key, path in inputs.items()):
        raise ValueError("changed fitting inputs")
    manifest.update(
        status="frozen",
        primary_method=PRIMARY,
        features=list(FEATURES),
        sklearn_version=sklearn.__version__,
        no_replay_labels=True,
        outputs={p.name: file_sha256(p) for p in args.output_dir.iterdir()},
        scientific_goal_achieved=False,
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="frozen", rows=len(training), model_sets=len(states))), flush=True)


def load_models(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "frozen" or manifest["primary_method"] != PRIMARY:
        raise ValueError("wrong or incomplete frozen model")
    for name, value in manifest["outputs"].items():
        if file_sha256(folder / name) != value:
            raise ValueError("changed frozen output")
    for key, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][key]:
            raise ValueError("changed fitting input")
    return json.loads((folder / "models.json").read_text())


def evaluate_session(row, states, output):
    folder = Path(row["artifact_dir"])
    arrays, original = source_grid(folder)
    grid = arrays["grid_cm"]
    diagonal = np.linalg.norm(np.ptp(grid, axis=0))
    state_key = row["animal"] if row["dataset"] == "pfeiffer_foster" else "external"
    state = states[state_key]
    if row["dataset"] == "pfeiffer_foster" and (state["excluded_animal"] != row["animal"] or row["animal"] in state["training_animals"]):
        raise ValueError("held-out rat leaked into training")
    target = output / (row["animal"] + "__" + row["session"].replace("/", "_"))
    target.mkdir(exist_ok=False)
    frame = pd.read_csv(folder / "event_readouts.csv.gz", float_precision="round_trip")
    hashes = json.loads((folder / "outputs.json").read_text())
    if any(file_sha256(folder / k) != v for k, v in hashes.items()):
        raise ValueError("changed base banks")
    frames = []
    for (source, split), group in frame.groupby(["source", "split"]):
        saved = load_npz(folder / f"{source}_split{split}_audit.npz")
        ids = saved["event_ids"]
        baseline = group.loc[group.method.eq("independent")].set_index("event_index").loc[ids]
        context = group.loc[group.method.eq("unconditional_context")].set_index("event_index").loc[ids]
        banks, diagnostics, audit = {}, {}, dict(event_ids=ids)
        for side in ("a", "b"):
            x = features(baseline, context, side, diagonal)
            physical, brier = (predict_tree_model(x, state[k]) for k in ("physical", "brier"))
            selected = choose(baseline, physical, brier, side)
            independent, temporal = (saved[f"{side}_{method}_posterior"] for method in ("independent", "unconditional_context"))
            chosen = np.where(selected[:, None], temporal, independent)
            rates = arrays["rates_hz"][saved[side + "_indices"]]
            ll = log_predictive(saved["counts"][:, saved[side + "_indices"]], rates)
            matched, temperature, available = match_entropy(ll, entropy(chosen))
            banks[side] = dict(independent=independent, unconditional_context=temporal, error_trained_context=chosen, entropy_matched=matched)
            diagnostics.update(
                {
                    side + "_predicted_physical_gain": physical,
                    side + "_predicted_brier_gain": brier,
                    side + "_use_context": selected,
                    side + "_entropy_control_available": available,
                    side + "_matched_log_temperature": temperature,
                }
            )
            audit[side + "_features"] = x.to_numpy()
            audit.update({f"{side}_{method}_posterior": value for method, value in banks[side].items()})
        for method in METHODS:
            value = baseline.drop(columns=[c for c in baseline if c.endswith("_median_predictive_delta")]).copy()
            a, b = (posterior_metrics(banks[side][method], grid, saved["truth_cm"]) for side in ("a", "b"))
            value["method"], value["model_key"] = method, state_key
            value["separation_cm"] = np.linalg.norm(a["mean"] - b["mean"], axis=1)
            value["regional_tv"] = 0.5 * np.abs(a["regional"] - b["regional"]).sum(axis=1)
            for name, vector in diagnostics.items():
                value[name] = vector
            for side, metrics in (("a", a), ("b", b)):
                for name in ("entropy", "width", "error", "brier", "nll"):
                    value[side + "_" + name] = metrics[name]
                value[side + "_x_cm"], value[side + "_y_cm"] = metrics["mean"].T
                for region in range(9):
                    value[f"{side}_region{region}"] = metrics["regional"][:, region]
            frames.append(value.reset_index())
        np.savez_compressed(target / f"{source}_split{split}_audit.npz", **audit)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(target / "event_readouts.csv.gz", index=False)
    (target / "frozen_input.json").write_text(json.dumps(dict(base_folder=str(folder), base_outputs=hashes, original=original, state_key=state_key), indent=2) + "\n")
    (target / "outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in target.iterdir()}, indent=2) + "\n")
    return dict(dataset=row["dataset"], animal=row["animal"], session=row["session"], status="complete", rows=len(full), artifact_dir=str(target))


def measure(args):
    producer = verified_base(args.base_dir, args.base_audit)
    states = load_models(args.frozen_models)
    if {r["dataset"] for r in producer["results"]} != {"pfeiffer_foster"}:
        validate_development(args.development_report, args.frozen_models)
    inputs = dict(
        base=args.base_dir / "manifest.json",
        audit=args.base_audit / "independent_audit.json",
        frozen=args.frozen_models / "manifest.json",
        models=args.frozen_models / "models.json",
        script=Path(__file__),
        protocol=ROOT / "docs/error_trained_context_protocol.md",
    )
    if args.development_report is not None:
        inputs["development"] = args.development_report / "manifest.json"
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest.update(status="running", primary_method=PRIMARY, scientific_goal_achieved=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for row in producer["results"]:
        rows.append(evaluate_session(row, states, args.output_dir))
        pd.DataFrame(rows).to_csv(args.output_dir / "measurement_sessions.csv", index=False)
        print(json.dumps(rows[-1]), flush=True)
    if any(file_sha256(path) != manifest["input_file_sha256"][key] for key, path in inputs.items()):
        raise ValueError("changed measurement input")
    manifest.update(status="complete", results=rows)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate_development(folder, models):
    if folder is None:
        raise ValueError("independent data require a development pass")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("primary_method") != PRIMARY or manifest.get("primary_advanced") is not True:
        raise ValueError("wrong or failed development method")
    for key, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][key]:
            raise ValueError("changed development input")
    path = folder / "gate_summary.csv"
    if file_sha256(path) != manifest["output_sha256"][path.name]:
        raise ValueError("changed development gates")
    gates = pd.read_csv(path)
    required = {"independently_verified", "all_primary_entropy_controls_available", "advance_external_validation"}
    required.update(prefix + metric for prefix in ("real_reduction_", "beats_entropy_control_") for metric in ("separation_cm", "regional_tv"))
    required.update("no_increased_" + side + "_entropy" for side in ("a", "b"))
    required.update(source + "_no_worse_" + side + "_" + metric for source in set(SOURCES) - {"real"} for side in ("a", "b") for metric in ("error", "error_p90", "brier"))
    required.update(source + "_" + side + "_predicts_known_gain" for source in ("run_q4", "sim_late_jump") for side in ("a", "b"))
    if gates.empty or gates.gate.duplicated().any() or not gates.passed.eq(True).all() or set(gates.gate) != required:
        raise ValueError("incomplete development gates")
    producer_path = Path(manifest["input_file_paths"]["producer"])
    producer = json.loads(producer_path.read_text())
    audit = json.loads(Path(manifest["input_file_paths"]["audit"]).read_text())
    if (
        producer["status"] != "complete"
        or producer.get("primary_method") != PRIMARY
        or {r["dataset"] for r in producer["results"]} != {"pfeiffer_foster"}
        or len({(r["animal"], r["session"]) for r in producer["results"]}) != 8
        or len({r["animal"] for r in producer["results"]}) != 4
    ):
        raise ValueError("PF development required")
    if (
        producer["input_file_sha256"]["frozen"] != file_sha256(models / "manifest.json")
        or audit["status"] != "passed"
        or audit["input_file_sha256"]["producer"] != file_sha256(producer_path)
    ):
        raise ValueError("different fitted model or missing development audit")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "measure"))
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--base-audit", type=Path, required=True)
    p.add_argument("--frozen-models", type=Path)
    p.add_argument("--development-report", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.phase == "measure" and args.frozen_models is None:
        p.error("measure requires --frozen-models")
    (freeze if args.phase == "freeze" else measure)(args)


if __name__ == "__main__":
    main()
