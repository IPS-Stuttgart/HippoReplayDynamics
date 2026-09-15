#!/usr/bin/env python3
"""Independent source, tree, constrained-policy and held-rat evaluation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from scripts.audit_reserve_cell_content import values_for, load, sha
from scripts.audit_content_screening_bound import verify_certificate
from scripts.audit_spatial_predictive_content import independent_tables

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
REAL = ("all_fixed_candidates", "full_accepted_segment")
TRUTH = ("run_q4", "test_poisson_gain1", "test_poisson_gain4", "test_conditional", "test_conditional_map_drift", "test_conditional_shared_assembly")
CAL = ("run_q3", "cal_poisson_gain1", "cal_conditional")
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


def feature_check(frame, diagonal):
    np.testing.assert_allclose(frame.signed_home, frame.high_home - frame.low_home, atol=1e-10)
    np.testing.assert_allclose(frame.normalized_separation, frame.separation / diagonal, atol=1e-10)
    for side in ("high", "low"):
        np.testing.assert_allclose(frame[f"normalized_{side}_error"], frame[f"{side}_error"] / diagonal, atol=1e-10)
        for key in ("spikes", "active"):
            np.testing.assert_allclose(frame[f"log_{side}_{key}"], np.log1p(frame[f"{side}_{key}"]), atol=1e-10)


def route(tree, frame):
    outputs = []
    for features in frame[list(FEATURES)].to_numpy(np.float32):
        node = 0
        seen = set()
        while tree["left"][node] != -1:
            assert node not in seen
            seen.add(node)
            field = tree["feature"][node]
            node = tree["left"][node] if features[field] <= tree["threshold"][node] else tree["right"][node]
        outputs.append(node)
    return np.array(outputs)


def weight_rows(frame):
    weight = np.empty(len(frame))
    rats = sorted(frame.animal.unique())
    for source in CAL:
        for rat in rats:
            sessions = sorted(frame[frame.animal.eq(rat)].session.unique())
            for session in sessions:
                for label in (0, 1):
                    mask = frame.source.eq(source) & frame.session.eq(session) & frame.true_home.eq(label)
                    assert mask.any()
                    weight[mask] = 1 / (3 * len(rats) * len(sessions) * 2 * mask.sum())
    return weight


def programs(frame, group, weight, optimal):
    g, n = group.max() + 1, len(frame)
    A, b, E, f = [], [], [], []

    def aggregate(v, progress=0):
        row = np.zeros(g + 1)
        np.add.at(row, group, v)
        row[g] = progress
        return row

    for source in CAL:
        w = np.where(frame.source.eq(source), weight, 0)
        w /= w.sum()
        for label in (0, 1):
            part = w * frame.true_home.eq(label).to_numpy()
            E.append(aggregate(part / part.sum()))
            f.append(0.5)
        delta = frame.high_home.to_numpy() - frame.low_home.to_numpy()
        gap = abs(np.dot(w, delta))
        scale = max(gap, 0.05)
        A.extend([aggregate(w * delta / scale, 0.1 * gap / scale), aggregate(-w * delta / scale, 0.1 * gap / scale)])
        b.extend([0.5 * gap / scale] * 2)
        for key in COSTS[1:]:
            v = frame[key].to_numpy(float)
            average = np.dot(w, v)
            norm = max(average, 1e-12)
            progress = 0.05 if key in ("normalized_separation", "regional_tv") else 0
            A.append(aggregate(w * v / norm, progress * average / norm))
            b.append(0.5 * average / norm)
    one = dict(A=np.array(A), b=np.array(b), E=np.array(E), f=np.array(f), lo=np.zeros(g + 1), hi=np.ones(g + 1), c=np.r_[np.zeros(g), -1.0], group=group)
    extension = np.zeros((2 * g, 2 * g + 1))
    bounds = []
    for j in range(g):
        extension[2 * j, j] = 1
        extension[2 * j + 1, j] = -1
        extension[2 * j : 2 * j + 2, g + 1 + j] = -1
        bounds.extend([0.5, -0.5])
    frequency = np.zeros(g)
    np.add.at(frequency, group, weight)
    low = np.zeros(2 * g + 1)
    low[g] = max(optimal - 1e-8, 0)
    two = dict(
        A=np.vstack([np.pad(one["A"], ((0, 0), (0, g))), extension]),
        b=np.r_[b, bounds],
        E=np.pad(one["E"], ((0, 0), (0, g))),
        f=one["f"],
        c=np.r_[np.zeros(g + 1), frequency],
        lo=low,
        hi=np.r_[np.ones(g + 1), np.full(g, 0.5)],
        group=group,
    )
    assert len(group) == n
    return one, two


def refit_and_verify(cal, held, model, root):
    frame = cal[cal.animal.ne(held)].reset_index(drop=True)
    assert set(frame.animal) == {"Rat1", "Rat2", "Rat4"} - {held}
    assert model["held_rat"] == held and model["features"] == list(FEATURES)
    assert model["training_sessions"] == sorted(frame.session.unique()) and model["training_rows"] == len(frame)
    w = weight_rows(frame)
    x, y = frame[list(FEATURES)].to_numpy(), frame[list(COSTS)].to_numpy()
    mean = np.average(y, weights=w, axis=0)
    sd = np.maximum(np.sqrt(np.average((y - mean) ** 2, weights=w, axis=0)), 0.01)
    np.testing.assert_allclose(model["target_mean"], mean, atol=1e-12)
    np.testing.assert_allclose(model["target_scale"], sd, atol=1e-12)
    tree = DecisionTreeRegressor(max_depth=8, max_leaf_nodes=128, min_samples_leaf=64, criterion="squared_error", random_state=20260915)
    tree.fit(x, (y - mean) / sd, sample_weight=w)
    for name, computed in [("left", tree.tree_.children_left), ("right", tree.tree_.children_right), ("feature", tree.tree_.feature), ("threshold", tree.tree_.threshold)]:
        np.testing.assert_array_equal(model["tree"][name], computed)
    leaves = route(model["tree"], frame)
    np.testing.assert_array_equal(leaves, tree.apply(x))
    nodes, group = np.unique(leaves, return_inverse=True)
    np.testing.assert_array_equal(nodes, model["leaf_nodes"])
    maximal = load(root / f"{held}_maximal.npz")
    conservative = load(root / f"{held}_conservative.npz")
    one, two = programs(frame, group, w, float(maximal["x"][-1]))
    residuals = [verify_certificate(p, c) for p, c in [(one, maximal), (two, conservative)]]
    np.testing.assert_allclose(model["maximum_training_progress"], maximal["x"][-1], atol=1e-10)
    np.testing.assert_allclose(model["chosen_training_progress"], conservative["x"][len(nodes)], atol=1e-10)
    np.testing.assert_allclose(model["probabilities"], conservative["x"][: len(nodes)], atol=1e-10)
    return residuals


def reconstruct_gates(events, tables):
    session, animal, summary = (tables[k] for k in ("session_summary", "animal_summary", "summary"))
    expected = {(s, src, e) for s in SESSIONS for src in REAL + TRUTH for e in (("early_run", "full_run") if src in REAL else ("early_run",))}
    actual = set(events[["session", "source", "encoding"]].itertuples(index=False, name=None))
    flags = dict(complete_source_coverage=expected == actual and not events.duplicated(["session", "source", "encoding", "observation_index"]).any())
    selected = session[session.method.ne("all")]
    flags["fixed_half_coverage"] = len(selected) == 120 and selected.retained_events.eq((selected.total_events + 1) // 2).all()
    for enc in ("early_run", "full_run"):
        for src in REAL:
            values = animal[animal.encoding.eq(enc) & animal.source.eq(src)].pivot(index="animal", columns="method", values="home_gap")
            average = summary[summary.encoding.eq(enc) & summary.source.eq(src)].set_index("method").home_gap
            flags[f"home_{enc}_{src}"] = (
                len(values) == 3
                and np.isfinite(values.predictive_half).all()
                and (values.predictive_half <= values["all"] + 1e-10).all()
                and average.predictive_half <= (0.8 if src == REAL[0] else 1) * average["all"] + 1e-10
            )
    for key in ("separation", "regional_tv", "high_entropy", "low_entropy"):
        values = animal[animal.encoding.eq("early_run") & animal.source.eq(REAL[0])].pivot(index="animal", columns="method", values=key)
        ok = len(values) == 3 and np.isfinite(values.predictive_half).all()
        if key in ("separation", "regional_tv"):
            ok = ok and (values.predictive_half < values["all"]).mean() >= 0.75 and values.predictive_half.mean() <= 0.9 * values["all"].mean()
        else:
            ok = ok and values.predictive_half.mean() <= values["all"].mean() + 1e-10
        flags[f"real_{key}"] = ok
    for src in TRUTH:
        for key in ("balanced_high_error", "balanced_low_error", "balanced_high_brier", "balanced_low_brier"):
            values = animal[animal.source.eq(src)].pivot(index="animal", columns="method", values=key)
            flags[f"truth_{src}_{key}"] = len(values) == 3 and np.isfinite(values.predictive_half).all() and (values.predictive_half <= values["all"] + 1e-10).all()
    retained = session[session.source.isin(TRUTH) & session.method.eq("predictive_half")][["class0_retention", "class1_retention"]]
    flags["both_truth_classes_retained"] = len(retained) == 24 and np.isfinite(retained).all().all() and retained.ge(0.2).all().all()
    flags = {k: bool(v) for k, v in flags.items()}
    flags["development_numerical_screen"] = all(flags.values())
    return flags


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    for path, h in manifest["input_file_sha256"].items():
        assert sha(path) == h, path
    for name, h in manifest["output_sha256"].items():
        assert sha(root / name) == h, name
    frozen = json.loads((root / "frozen_models.json").read_text())
    assert sha(root / "frozen_models.json") == manifest["frozen_models_sha256"]
    assert frozen["input_file_sha256"] == manifest["input_file_sha256"]
    assert frozen["created_at_utc"] < manifest["created_at_utc"]
    assert sha(root / "calibration.csv.gz") == frozen["calibration_sha256"]
    cal = pd.read_csv(root / "calibration.csv.gz", float_precision="round_trip")
    expected_cal = {(s, c) for s in SESSIONS for c in CAL}
    assert set(cal[["session", "source"]].itertuples(index=False, name=None)) == expected_cal
    assert not cal.duplicated(["session", "source", "observation_index"]).any()
    events = pd.read_csv(root / "event_selection.csv.gz", dtype={"event_id": str}, float_precision="round_trip")
    for session in SESSIONS:
        folder = Path(manifest["source_dir"]) / session.replace("/", "_")
        enc = load(folder / "encoding.npz")
        diagonal = np.linalg.norm(enc["grid_cm"].max(axis=0) - enc["grid_cm"].min(axis=0))
        for source in CAL + REAL + TRUTH:
            bank = load(folder / f"{source}.npz")
            rows = np.arange(len(bank["counts"]))
            if source == "run_q3":
                rows = np.sort(np.unique(bank["parent_ids"], return_index=True)[1])
                bank = {k: v[rows] for k, v in bank.items()}
            for encoding in ("early_run", "full_run") if source in REAL else ("early_run",):
                table = cal if source in CAL else events[events.encoding.eq(encoding)]
                frame = table[table.session.eq(session) & table.source.eq(source)].sort_values("observation_index").reset_index(drop=True)
                np.testing.assert_array_equal(frame.observation_index, rows)
                assert frame.animal.eq(session.split("/")[0]).all()
                if source not in CAL:
                    np.testing.assert_array_equal(frame.event_id, bank.get("event_ids", np.arange(len(rows))).astype(str))
                expected = values_for(bank, enc, encoding, dict(high=[], low=[]))
                for name, values in expected.items():
                    np.testing.assert_allclose(frame[name], values, atol=1e-8, rtol=1e-9)
                feature_check(frame, diagonal)
                if source not in CAL:
                    model = frozen["models"][session.split("/")[0]]
                    scores = np.array([dict(zip(model["leaf_nodes"], model["probabilities"], strict=True))[int(k)] for k in route(model["tree"], frame)])
                    np.testing.assert_allclose(frame.pair_score, scores, atol=1e-12)
                    tie = [int.from_bytes(hashlib.sha256(f"20260915|learned-content-screen|{session}|{source}|{j}".encode()).digest()[:8], "little") for j in rows]
                    for name, rank in [
                        ("predictive_half", scores),
                        ("spike_half", np.minimum(frame.high_spikes, frame.low_spikes)),
                        ("entropy_half", -np.maximum(frame.high_entropy, frame.low_entropy)),
                    ]:
                        order = sorted(range(len(rows)), key=lambda j: (-float(np.asarray(rank)[j]), tie[j]))
                        chosen = np.zeros(len(rows), bool)
                        chosen[order[: (len(rows) + 1) // 2]] = True
                        np.testing.assert_array_equal(frame[name], chosen)
                    assert frame["all"].all()
        print("independently reconstructed", session, flush=True)
    assert set(frozen["models"]) == {"Rat1", "Rat2", "Rat4"}
    residuals = []
    for held, model in frozen["models"].items():
        residuals.extend(refit_and_verify(cal, held, model, root))
    tables = independent_tables(events)
    flags = reconstruct_gates(events, tables)
    recorded = pd.read_csv(root / "gates.csv")
    assert not recorded.gate.duplicated().any() and dict(zip(recorded.gate, recorded.passed, strict=True)) == flags
    keys = dict(
        session_summary=["animal", "session", "source", "encoding", "method"],
        animal_summary=["animal", "source", "encoding", "method"],
        summary=["source", "encoding", "method"],
        correlations=["animal"],
    )
    for name, table in tables.items():
        if "method" in table:
            table["method"] = table.method.replace({"predictive_half": "learned_half"})
        saved = pd.read_csv(root / f"{name}.csv")
        pd.testing.assert_frame_equal(
            table.sort_values(keys[name]).reset_index(drop=True), saved[table.columns].sort_values(keys[name]).reset_index(drop=True), check_dtype=False, atol=1e-9, rtol=1e-8
        )
    result = dict(
        status="pass",
        manifest_sha256=sha(root / "manifest.json"),
        models_refitted=3,
        calibration_rows=len(cal),
        evaluation_rows=len(events),
        certificate_max_residuals=pd.DataFrame(residuals).max().to_dict(),
        gates=flags,
        external_validation=False,
        validated_remedy=False,
        scope="full independent source decoding, features, held-rat exclusion, tree refits, six primal-dual certificates, selection, truth metrics and gates",
        audit_script_sha256=sha(__file__),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.output)
