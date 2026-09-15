#!/usr/bin/env python3
"""Session-local paired screening with separately protected spatial classes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts import learned_content_screen as old
from scripts._provenance import build_script_provenance, file_sha256

LOSSES = ("normalized_high_error", "normalized_low_error", "high_brier", "low_brier")
CLASS_METRICS = ("high_error", "low_error", "high_brier", "low_brier")


def local_training(calibration, session):
    frame = calibration[calibration.session.eq(session)].reset_index(drop=True)
    if not len(frame) or frame.animal.nunique() != 1 or not frame.animal.eq(session.split("/")[0]).all():
        raise ValueError("one valid calibration session required")
    weights = old.training_weights(frame)
    if frame.groupby(["source", "true_home"]).size().min() < 10:
        raise ValueError("at least ten calibration observations per source/class required")
    return frame, weights


def target_matrix(frame):
    columns = [frame[list(old.COSTS)].to_numpy(), frame[["true_home"]].to_numpy()]
    names = list(old.COSTS) + ["true_home"]
    for label in (0, 1):
        columns.append(frame[list(LOSSES)].to_numpy() * frame.true_home.eq(label).to_numpy()[:, None])
        names.extend(f"class{label}_{name}" for name in LOSSES)
    y = np.column_stack(columns)
    if not np.isfinite(y).all():
        raise ValueError("finite training targets required")
    return y, names


def build_problem(frame, group, weight):
    problem = old.build_problem(frame, group, weight)
    size = len(problem["c"]) - 1
    rows, rhs = [], []
    for source in old.CAL:
        for label in (0, 1):
            w = weight * (frame.source.eq(source) & frame.true_home.eq(label)).to_numpy()
            w /= w.sum()
            for name in LOSSES:
                v = frame[name].to_numpy()
                average = np.dot(w, v)
                scale = max(average, 1e-12)
                rows.append(np.r_[np.bincount(group, weights=w * v / scale, minlength=size), 0.0])
                rhs.append(0.5 * average / scale)
    problem["A"] = np.vstack([problem["A"], rows])
    problem["b"] = np.r_[problem["b"], rhs]
    constant = np.r_[np.full(size, 0.5), 0.0]
    if np.max(problem["A"] @ constant - problem["b"]) > 1e-10:
        raise ValueError("constant zero-progress policy must remain feasible")
    return problem


def fit_model(calibration, session, directory):
    frame, weights = local_training(calibration, session)
    y, names = target_matrix(frame)
    mean = np.average(y, weights=weights, axis=0)
    scale = np.maximum(np.sqrt(np.average((y - mean) ** 2, weights=weights, axis=0)), 0.01)
    x = frame[list(old.FEATURES)].to_numpy()
    estimator = DecisionTreeRegressor(max_depth=8, max_leaf_nodes=128, min_samples_leaf=64, criterion="squared_error", random_state=20260915)
    estimator.fit(x, (y - mean) / scale, sample_weight=weights)
    tree = dict(
        left=estimator.tree_.children_left.tolist(),
        right=estimator.tree_.children_right.tolist(),
        feature=estimator.tree_.feature.tolist(),
        threshold=estimator.tree_.threshold.tolist(),
    )
    leaves = old.tree_leaves(tree, x)
    np.testing.assert_array_equal(leaves, estimator.apply(x))
    nodes, group = np.unique(leaves, return_inverse=True)
    first = build_problem(frame, group, weights)
    one = old.solve_program(first)
    cert1 = old.certificate(one, group)
    frequency = np.bincount(group, weights=weights)
    two = old.solve_program(old.conservative_problem(first, one.x[-1], frequency))
    cert2 = old.certificate(two, group)
    slug = session.replace("/", "_")
    np.savez_compressed(directory / f"{slug}_maximal.npz", **cert1)
    np.savez_compressed(directory / f"{slug}_conservative.npz", **cert2)
    return dict(
        session=session,
        training_sessions=[session],
        training_rows=len(frame),
        features=list(old.FEATURES),
        target_names=names,
        target_mean=mean.tolist(),
        target_scale=scale.tolist(),
        tree=tree,
        leaf_nodes=nodes.tolist(),
        probabilities=two.x[: len(nodes)].tolist(),
        maximum_training_progress=float(one.x[-1]),
        chosen_training_progress=float(two.x[len(nodes)]),
    )


def select(frame, model):
    if not frame.session.eq(model["session"]).all() or frame.source.nunique() != 1 or frame.encoding.nunique() != 1:
        raise ValueError("selection must use the matching session and one source/map")
    frame = frame.copy()
    frame["pair_score"] = old.predict(model, frame)
    tie = np.array([old.base.seed("local-content-screen", r.session, r.source, int(r.observation_index)) for r in frame.itertuples()], dtype=np.uint64)
    frame["all"] = True
    frame["predictive_half"] = old.base.select_half(frame.pair_score, tie)
    frame["spike_half"] = old.base.select_half(np.minimum(frame.high_spikes, frame.low_spikes), tie)
    frame["entropy_half"] = old.base.select_half(-np.maximum(frame.high_entropy, frame.low_entropy), tie)
    return frame


def classwise(events):
    rows = []
    truth = events[events.source.isin(old.base.TRUTH)]
    for (session, source, label), group in truth.groupby(["session", "source", "true_home"]):
        for method, col in (("all", "all"), ("local_half", "predictive_half")):
            selected = group[group[col]]
            row = dict(session=session, animal=group.animal.iloc[0], source=source, true_home=label, method=method, total_events=len(group), retained_events=len(selected))
            row.update({k: selected[k].mean() for k in CLASS_METRICS + ("high_home", "low_home")})
            rows.append(row)
    return pd.DataFrame(rows)


def regional_flags(table):
    flags = {}
    expected = {(s, label) for s in old.base.SESSIONS for label in (0, 1)}
    for source in old.base.TRUTH:
        part = table[table.source.eq(source)]
        for metric in CLASS_METRICS:
            values = part.pivot(index=["session", "true_home"], columns="method", values=metric)
            passed = (
                set(values.index) == expected
                and {"all", "local_half"} <= set(values.columns)
                and np.isfinite(values).all().all()
                and values.local_half.le(values["all"] + 1e-10).all()
            )
            flags[f"regional_{source}_{metric}"] = bool(passed)
    return flags


def summarize(events):
    tables = old.summarize(events)
    classes = classwise(events)
    tables["truth_by_class"] = classes
    gates = tables["gates"]
    gates = gates[gates.gate.ne("development_numerical_screen")].copy()
    extra = pd.DataFrame([dict(gate=k, passed=v, detail="Neither true class may worsen in any session") for k, v in regional_flags(classes).items()])
    # Preserve the established three-column gate convention.
    extra.columns = gates.columns
    gates = pd.concat([gates, extra], ignore_index=True)
    gates.loc[len(gates)] = ["development_numerical_screen", bool(gates.passed.all()), "Requires external confirmation even if numerical gates pass"]
    tables["gates"] = gates
    for table in tables.values():
        if "method" in table:
            table["method"] = table.method.replace({"learned_half": "local_half"})
    return tables


def checked_source(root, audit_path):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass" or audit.get("manifest_sha256") != file_sha256(root / "manifest.json"):
        raise ValueError("independently reconstructed source required")
    inputs = {**manifest["input_file_sha256"], str(audit_path): file_sha256(audit_path), str(root / "manifest.json"): file_sha256(root / "manifest.json")}
    inputs.update({str(root / k): v for k, v in manifest["output_sha256"].items()})
    for path, value in inputs.items():
        if file_sha256(path) != value:
            raise ValueError(f"changed source: {path}")
    return inputs


def measure(args):
    inputs = checked_source(args.result_dir, args.audit)
    for path in (
        Path(__file__),
        ROOT / "scripts/learned_content_screen.py",
        ROOT / "scripts/bound_content_screening.py",
        ROOT / "scripts/spatial_predictive_content.py",
        ROOT / "docs/local_content_screen_protocol.md",
    ):
        inputs[str(path)] = file_sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cal = pd.read_csv(args.result_dir / "calibration.csv.gz", float_precision="round_trip")
    if set(cal.session) != set(old.base.SESSIONS):
        raise ValueError("all original sessions required")
    models = {session: fit_model(cal, session, args.output_dir) for session in old.base.SESSIONS}
    frozen = {
        **build_script_provenance(),
        "models": models,
        "input_file_sha256": inputs,
        "calibration_sha256": file_sha256(args.result_dir / "calibration.csv.gz"),
        "external_validation": False,
    }
    frozen["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = args.output_dir / "frozen_models.json"
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = file_sha256(path)
    evaluated_at = datetime.now(timezone.utc).isoformat()
    # Only after models are frozen can the evaluation reader be opened.
    previous = pd.read_csv(args.result_dir / "event_selection.csv.gz", dtype={"event_id": str}, float_precision="round_trip")
    parts = [select(frame, models[session]) for (session, _, _), frame in previous.groupby(["session", "source", "encoding"], sort=True)]
    events = pd.concat(parts, ignore_index=True)
    events.to_csv(args.output_dir / "event_selection.csv.gz", index=False)
    for name, table in summarize(events).items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    if file_sha256(path) != frozen_hash:
        raise ValueError("model changed during evaluation")
    for source, value in inputs.items():
        if file_sha256(source) != value:
            raise ValueError(f"input changed while evaluating: {source}")
    manifest = {
        **build_script_provenance(),
        "source_result_dir": str(args.result_dir),
        "source_audit": str(args.audit),
        "input_file_sha256": inputs,
        "evaluation_opened_at_utc": evaluated_at,
        "frozen_models_sha256": frozen_hash,
        "output_sha256": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
        "external_validation": False,
        "validated_remedy": False,
    }
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    measure(parser.parse_args())
