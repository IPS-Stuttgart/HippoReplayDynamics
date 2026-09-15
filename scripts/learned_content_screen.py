#!/usr/bin/env python3
"""Leave-rat-out paired screening learned from accuracy-constrained RUN controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts._provenance import build_script_provenance, file_sha256
from scripts.bound_content_screening import solve_program
from scripts.reserve_cell_content import score_pair
from scripts import spatial_predictive_content as base

CAL = ("run_q3", "cal_poisson_gain1", "cal_conditional")
RATS = ("Rat1", "Rat2", "Rat4")
FEATURES = (
    "high_home",
    "low_home",
    "high_entropy",
    "low_entropy",
    "log_high_spikes",
    "log_low_spikes",
    "log_high_active",
    "log_low_active",
    "normalized_separation",
    "regional_tv",
)
COSTS = ("signed_home", "normalized_separation", "regional_tv", "high_entropy", "low_entropy", "normalized_high_error", "normalized_low_error", "high_brier", "low_brier")


def enrich(frame, diagonal):
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("positive arena diagonal required")
    frame = frame.copy()
    frame["normalized_separation"] = frame.separation / diagonal
    frame["signed_home"] = frame.high_home - frame.low_home
    for side in ("high", "low"):
        frame[f"normalized_{side}_error"] = frame[f"{side}_error"] / diagonal
        for kind in ("spikes", "active"):
            frame[f"log_{side}_{kind}"] = np.log1p(frame[f"{side}_{kind}"])
    if not np.isfinite(frame[list(FEATURES)]).all().all():
        raise ValueError("nonfinite observable features")
    return frame


def training_weights(frame):
    keys = ["source", "animal", "session", "true_home"]
    counts = frame.groupby(keys).size()
    if not set(frame.source) == set(CAL) or not set(frame.true_home) == {0, 1}:
        raise ValueError("complete calibration sources/classes required")
    rats = sorted(frame.animal.unique())
    expected = {(src, rat, session, label) for src in CAL for rat in rats for session in frame[frame.animal.eq(rat)].session.unique() for label in (0, 1)}
    if set(counts.index) != expected:
        raise ValueError("calibration missing a rat/session/class")
    n_sessions = frame.groupby("animal").session.nunique().to_dict()
    weight = np.array([1 / (len(CAL) * len(rats) * n_sessions[r.animal] * 2 * counts.loc[(r.source, r.animal, r.session, r.true_home)]) for r in frame.itertuples()])
    np.testing.assert_allclose(weight.sum(), 1)
    return weight


def tree_leaves(tree, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("valid observable feature matrix required")
    nodes = np.zeros(len(x), int)
    left, right, features, threshold = [np.asarray(tree[k]) for k in ("left", "right", "feature", "threshold")]
    for _ in range(len(left)):
        active = np.flatnonzero(left[nodes] != -1)
        if not len(active):
            return nodes
        old = nodes[active]
        go_left = x[active, features[old]] <= threshold[old]
        nodes[active] = np.where(go_left, left[old], right[old])
    raise ValueError("cyclic or corrupt tree")


def build_problem(frame, group, weight):
    g = int(np.max(group)) + 1
    A, b, E, f = [], [], [], []

    def row(values, t=0):
        return np.r_[np.bincount(group, weights=values, minlength=g), t]

    for source in CAL:
        ix = frame.source.eq(source).to_numpy()
        w = weight * ix
        w /= w.sum()
        for label in (0, 1):
            wc = w * frame.true_home.eq(label).to_numpy()
            E.append(row(wc / wc.sum()))
            f.append(0.5)
        delta = frame.signed_home.to_numpy()
        gap = abs(np.dot(w, delta))
        scale = max(gap, 0.05)
        for sign in (1, -1):
            A.append(row(sign * w * delta / scale, 0.5 * 0.2 * gap / scale))
            b.append(0.5 * gap / scale)
        for name in COSTS[1:]:
            value = frame[name].to_numpy()
            average = np.dot(w, value)
            scale = max(average, 1e-12)
            slope = 0.1 if name in ("normalized_separation", "regional_tv") else 0
            A.append(row(w * value / scale, 0.5 * slope * average / scale))
            b.append(0.5 * average / scale)
    p = dict(A=np.array(A), b=np.array(b), E=np.array(E), f=np.array(f), c=np.r_[np.zeros(g), -1.0], lo=np.zeros(g + 1), hi=np.ones(g + 1), group=group)
    constant = np.r_[np.full(g, 0.5), 0]
    if np.max(p["A"] @ constant - p["b"]) > 1e-10 or np.max(abs(p["E"] @ constant - p["f"])) > 1e-10:
        raise ValueError("zero-progress constant policy must be feasible")
    return p


def conservative_problem(first, progress, frequency):
    g = len(frequency)
    A = [np.r_[a, np.zeros(g)] for a in first["A"]]
    b = first["b"].tolist()
    for j in range(g):
        for sign in (1, -1):
            row = np.zeros(2 * g + 1)
            row[j], row[g + 1 + j] = sign, -1
            A.append(row)
            b.append(sign * 0.5)
    lo = np.zeros(2 * g + 1)
    lo[g] = max(0, progress - 1e-8)
    return dict(
        A=np.array(A),
        b=np.array(b),
        E=np.pad(first["E"], ((0, 0), (0, g))),
        f=first["f"],
        c=np.r_[np.zeros(g + 1), frequency],
        lo=lo,
        hi=np.r_[np.ones(g + 1), np.full(g, 0.5)],
        group=first["group"],
    )


def certificate(result, group):
    if result.status != 0:
        raise RuntimeError(f"uncertified learned-policy solve: {result.message}")
    return dict(
        x=result.x,
        inequality_dual=result.ineqlin.marginals,
        equality_dual=result.eqlin.marginals,
        lower_dual=result.lower.marginals,
        upper_dual=result.upper.marginals,
        group=group,
    )


def fit_model(calibration, held_rat, directory):
    frame = calibration[calibration.animal.ne(held_rat)].reset_index(drop=True)
    if held_rat not in RATS or set(frame.animal) != set(RATS) - {held_rat}:
        raise ValueError("training must exclude held rat and contain both others")
    w = training_weights(frame)
    x, y = frame[list(FEATURES)].to_numpy(), frame[list(COSTS)].to_numpy()
    if not np.isfinite(y).all():
        raise ValueError("finite calibration costs required")
    mean = np.average(y, weights=w, axis=0)
    scale = np.maximum(np.sqrt(np.average((y - mean) ** 2, weights=w, axis=0)), 0.01)
    fitted = DecisionTreeRegressor(max_depth=8, max_leaf_nodes=128, min_samples_leaf=64, criterion="squared_error", random_state=20260915)
    fitted.fit(x, (y - mean) / scale, sample_weight=w)
    tree = dict(
        left=fitted.tree_.children_left.tolist(), right=fitted.tree_.children_right.tolist(), feature=fitted.tree_.feature.tolist(), threshold=fitted.tree_.threshold.tolist()
    )
    leaves = tree_leaves(tree, x)
    np.testing.assert_array_equal(leaves, fitted.apply(x))
    leaf_nodes, group = np.unique(leaves, return_inverse=True)
    first = build_problem(frame, group, w)
    one = solve_program(first)
    cert1 = certificate(one, group)
    frequency = np.bincount(group, weights=w)
    second = conservative_problem(first, one.x[-1], frequency)
    two = solve_program(second)
    cert2 = certificate(two, group)
    np.savez_compressed(directory / f"{held_rat}_maximal.npz", **cert1)
    np.savez_compressed(directory / f"{held_rat}_conservative.npz", **cert2)
    model = dict(
        held_rat=held_rat,
        training_sessions=sorted(frame.session.unique()),
        training_rows=len(frame),
        features=list(FEATURES),
        tree=tree,
        leaf_nodes=leaf_nodes.tolist(),
        probabilities=two.x[: len(leaf_nodes)].tolist(),
        maximum_training_progress=float(one.x[-1]),
        chosen_training_progress=float(two.x[len(leaf_nodes)]),
        target_mean=mean.tolist(),
        target_scale=scale.tolist(),
    )
    return model


def predict(model, frame):
    if model["features"] != list(FEATURES):
        raise ValueError("feature contract changed")
    leaf = tree_leaves(model["tree"], frame[list(FEATURES)])
    lookup = dict(zip(model["leaf_nodes"], model["probabilities"], strict=True))
    return np.array([lookup[int(k)] for k in leaf])


def selection_frame(frame, model):
    frame = frame.copy()
    frame["pair_score"] = predict(model, frame)
    tie = np.array([base.seed("learned-content-screen", r.session, r.source, int(r.observation_index)) for r in frame.itertuples()], dtype=np.uint64)
    frame["all"] = True
    frame["predictive_half"] = base.select_half(frame.pair_score, tie)
    frame["spike_half"] = base.select_half(np.minimum(frame.high_spikes, frame.low_spikes), tie)
    frame["entropy_half"] = base.select_half(-np.maximum(frame.high_entropy, frame.low_entropy), tie)
    return frame


def summarize(events):
    result = base.summarize(events)
    gates = result["gates"]
    gates = gates[~gates.gate.isin(("predictive_correlation", "development_numerical_screen"))].copy()
    gates.loc[len(gates)] = ["development_numerical_screen", bool(gates.passed.all()), "External validation and independent reconstruction still required"]
    result["gates"] = gates
    for frame in result.values():
        if "method" in frame:
            frame["method"] = frame.method.replace({"predictive_half": "learned_half"})
    return result


def measure(args):
    previous = json.loads((args.result_dir / "manifest.json").read_text())
    audit = json.loads(args.audit.read_text())
    if audit.get("status") != "pass" or audit["manifest_sha256"] != file_sha256(args.result_dir / "manifest.json"):
        raise ValueError("independently audited source required")
    inputs = {**previous["input_file_sha256"], str(args.audit): file_sha256(args.audit), str(args.result_dir / "manifest.json"): file_sha256(args.result_dir / "manifest.json")}
    inputs.update({str(args.result_dir / p): h for p, h in previous["output_sha256"].items()})
    for path in (
        Path(__file__),
        ROOT / "scripts/bound_content_screening.py",
        ROOT / "scripts/reserve_cell_content.py",
        ROOT / "scripts/spatial_predictive_content.py",
        ROOT / "docs/learned_content_screen_protocol.md",
    ):
        inputs[str(path)] = file_sha256(path)
    for path, h in inputs.items():
        if file_sha256(path) != h:
            raise ValueError(f"source changed: {path}")
    source = Path(previous["source_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    calibration = []
    encodings = {}
    for session in base.SESSIONS:
        folder = source / session.replace("/", "_")
        enc = base.read_npz(folder / "encoding.npz")
        diagonal = float(np.linalg.norm(np.ptp(enc["grid_cm"], axis=0)))
        encodings[session] = (enc, diagonal)
        for src in CAL:
            bank = base.read_npz(folder / f"{src}.npz")
            rows = np.arange(len(bank["counts"]))
            if src == "run_q3":
                rows = np.sort(np.unique(bank["parent_ids"], return_index=True)[1])
                bank = {k: v[rows] for k, v in bank.items()}
            values = score_pair(bank, enc, "early_run", dict(high=[], low=[]))
            frame = pd.DataFrame(values).assign(session=session, animal=session.split("/")[0], source=src, observation_index=rows)
            calibration.append(enrich(frame, diagonal))
    calibration = pd.concat(calibration, ignore_index=True)
    calibration.to_csv(args.output_dir / "calibration.csv.gz", index=False)
    calibration = pd.read_csv(args.output_dir / "calibration.csv.gz", float_precision="round_trip")
    models = {rat: fit_model(calibration, rat, args.output_dir) for rat in RATS}
    frozen = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file_sha256": inputs,
        "models": models,
        "calibration_sha256": file_sha256(args.output_dir / "calibration.csv.gz"),
        "external_validation": False,
    }
    (args.output_dir / "frozen_models.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = file_sha256(args.output_dir / "frozen_models.json")
    events = []
    for session in base.SESSIONS:
        slug = session.replace("/", "_")
        enc, diagonal = encodings[session]
        old = pd.read_csv(args.result_dir / f"{slug}_events.csv.gz", dtype={"event_id": str})
        old = old[old.method.eq("baseline")]
        for src in base.REAL + base.TRUTH:
            bank = base.read_npz(source / slug / f"{src}.npz")
            for encoding in ("early_run", "full_run") if src in base.REAL else ("early_run",):
                frame = old[old.source.eq(src) & old.encoding.eq(encoding)].sort_values("observation_index").reset_index(drop=True)
                np.testing.assert_array_equal(frame.observation_index, np.arange(len(bank["counts"])))
                np.testing.assert_array_equal(frame.event_id, bank.get("event_ids", np.arange(len(frame))).astype(str))
                events.append(selection_frame(enrich(frame, diagonal), models[session.split("/")[0]]))
    events = pd.concat(events, ignore_index=True)
    events.to_csv(args.output_dir / "event_selection.csv.gz", index=False)
    for name, frame in summarize(events).items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    assert file_sha256(args.output_dir / "frozen_models.json") == frozen_hash
    for path, h in inputs.items():
        if file_sha256(path) != h:
            raise ValueError(f"input changed while evaluating: {path}")
    manifest = {
        **build_script_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source),
        "source_result_dir": str(args.result_dir),
        "input_file_sha256": inputs,
        "frozen_models_sha256": frozen_hash,
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    measure(parser.parse_args())
