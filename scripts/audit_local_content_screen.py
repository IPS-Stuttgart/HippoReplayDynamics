#!/usr/bin/env python3
"""Independent audit of local policy fitting on immutable reconstructed sources."""

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

from scripts import audit_learned_content_screen as old
from scripts.audit_content_screening_bound import verify_certificate
from scripts.audit_reserve_cell_content import sha, load

LOSSES = ("normalized_high_error", "normalized_low_error", "high_brier", "low_brier")


def programs(frame, groups, weights, optimal):
    first, _ = old.programs(frame, groups, weights, optimal)
    g = len(first["c"]) - 1
    extra, rhs = [], []
    for source in old.CAL:
        for label in (0, 1):
            ix = frame.source.eq(source).to_numpy() & frame.true_home.eq(label).to_numpy()
            w = weights * ix
            w /= w.sum()
            for metric in LOSSES:
                mean = (w * frame[metric].to_numpy()).sum()
                row = np.zeros(g + 1)
                for j in range(g):
                    use = groups == j
                    row[j] = (w[use] * frame.loc[use, metric]).sum() / max(mean, 1e-12)
                extra.append(row)
                rhs.append(mean * 0.5 / max(mean, 1e-12))
    first["A"] = np.vstack([first["A"], extra])
    first["b"] = np.r_[first["b"], rhs]
    A, b = [], []
    for j in range(g):
        for sign in (1, -1):
            row = np.zeros(2 * g + 1)
            row[j], row[g + 1 + j] = sign, -1
            A.append(row)
            b.append(sign * 0.5)
    frequency = np.array([weights[groups == j].sum() for j in range(g)])
    lower = np.zeros(2 * g + 1)
    lower[g] = max(0, optimal - 1e-8)
    second = dict(
        A=np.vstack([np.pad(first["A"], ((0, 0), (0, g))), A]),
        b=np.r_[first["b"], b],
        E=np.pad(first["E"], ((0, 0), (0, g))),
        f=first["f"],
        lo=lower,
        hi=np.r_[np.ones(g + 1), np.full(g, 0.5)],
        c=np.r_[np.zeros(g + 1), frequency],
        group=groups,
    )
    return first, second


def refit(calibration, session, model, root):
    frame = calibration[calibration.session.eq(session)].reset_index(drop=True)
    assert model["session"] == session and model["training_sessions"] == [session] and model["training_rows"] == len(frame)
    assert model["features"] == list(old.FEATURES)
    assert frame.animal.eq(session.split("/")[0]).all()
    weights = old.weight_rows(frame)
    target_names = list(old.COSTS) + ["true_home"]
    columns = [frame[k].to_numpy() for k in target_names]
    for label in (0, 1):
        for loss in LOSSES:
            columns.append(np.where(frame.true_home.eq(label), frame[loss], 0))
            target_names.append(f"class{label}_{loss}")
    assert model["target_names"] == target_names
    targets = np.array(columns).T
    mean = np.sum(weights[:, None] * targets, axis=0)
    sd = np.maximum(np.sqrt(np.sum(weights[:, None] * (targets - mean) ** 2, axis=0)), 0.01)
    np.testing.assert_allclose(model["target_mean"], mean, atol=1e-12)
    np.testing.assert_allclose(model["target_scale"], sd, atol=1e-12)
    tree = DecisionTreeRegressor(max_depth=8, max_leaf_nodes=128, min_samples_leaf=64, criterion="squared_error", random_state=20260915)
    # Identical numerical accumulation order is necessary for exact tree refits.
    fit_mean = np.average(targets, weights=weights, axis=0)
    fit_sd = np.maximum(np.sqrt(np.average((targets - fit_mean) ** 2, weights=weights, axis=0)), 0.01)
    tree.fit(frame[list(old.FEATURES)].to_numpy(), (targets - fit_mean) / fit_sd, sample_weight=weights)
    for key, value in (("left", tree.tree_.children_left), ("right", tree.tree_.children_right), ("feature", tree.tree_.feature), ("threshold", tree.tree_.threshold)):
        np.testing.assert_array_equal(model["tree"][key], value)
    leaves = old.route(model["tree"], frame)
    np.testing.assert_array_equal(leaves, tree.apply(frame[list(old.FEATURES)].to_numpy()))
    nodes, groups = np.unique(leaves, return_inverse=True)
    np.testing.assert_array_equal(model["leaf_nodes"], nodes)
    slug = session.replace("/", "_")
    c1, c2 = load(root / f"{slug}_maximal.npz"), load(root / f"{slug}_conservative.npz")
    p1, p2 = programs(frame, groups, weights, c1["x"][-1])
    residuals = [verify_certificate(p, c) for p, c in ((p1, c1), (p2, c2))]
    np.testing.assert_allclose(model["maximum_training_progress"], c1["x"][-1], atol=1e-10)
    np.testing.assert_allclose(model["chosen_training_progress"], c2["x"][len(nodes)], atol=1e-10)
    np.testing.assert_allclose(model["probabilities"], c2["x"][: len(nodes)], atol=1e-10)
    return residuals


def class_table(events):
    rows = []
    for session in old.SESSIONS:
        for source in old.TRUTH:
            frame = events[events.session.eq(session) & events.source.eq(source)]
            for label in (0, 1):
                data = frame[frame.true_home.eq(label)]
                for name, flag in (("all", "all"), ("local_half", "predictive_half")):
                    chosen = data[data[flag]]
                    row = dict(session=session, animal=session.split("/")[0], source=source, true_home=label, method=name, total_events=len(data), retained_events=len(chosen))
                    for key in ("high_error", "low_error", "high_brier", "low_brier", "high_home", "low_home"):
                        row[key] = np.mean(chosen[key].to_numpy()) if len(chosen) else np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def class_flags(classes):
    result = {}
    for source in old.TRUTH:
        for metric in ("high_error", "low_error", "high_brier", "low_brier"):
            ok = True
            for session in old.SESSIONS:
                for label in (0, 1):
                    frame = classes[classes.source.eq(source) & classes.session.eq(session) & classes.true_home.eq(label)]
                    if len(frame) != 2 or set(frame.method) != {"all", "local_half"}:
                        ok = False
                        continue
                    values = frame.set_index("method")[metric]
                    ok = ok and np.isfinite(values).all() and values.local_half <= values["all"] + 1e-10
            result[f"regional_{source}_{metric}"] = bool(ok)
    return result


def audit(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    for path, value in manifest["input_file_sha256"].items():
        assert sha(path) == value, path
    for name, value in manifest["output_sha256"].items():
        assert sha(root / name) == value, name
    source = Path(manifest["source_result_dir"])
    source_manifest = json.loads((source / "manifest.json").read_text())
    source_audit = json.loads(Path(manifest["source_audit"]).read_text())
    assert source_audit["status"] == "pass" and source_audit["manifest_sha256"] == sha(source / "manifest.json")
    for key, value in source_manifest["input_file_sha256"].items():
        assert manifest["input_file_sha256"][key] == value
    for name, value in source_manifest["output_sha256"].items():
        assert manifest["input_file_sha256"][str(source / name)] == value
    frozen = json.loads((root / "frozen_models.json").read_text())
    assert sha(root / "frozen_models.json") == manifest["frozen_models_sha256"]
    assert frozen["created_at_utc"] < manifest["evaluation_opened_at_utc"] < manifest["created_at_utc"]
    assert frozen["input_file_sha256"] == manifest["input_file_sha256"]
    assert frozen["calibration_sha256"] == sha(source / "calibration.csv.gz")
    assert set(frozen["models"]) == set(old.SESSIONS)
    cal = pd.read_csv(source / "calibration.csv.gz", float_precision="round_trip")
    residuals = []
    for session, model in frozen["models"].items():
        residuals.extend(refit(cal, session, model, root))
    original = pd.read_csv(source / "event_selection.csv.gz", dtype={"event_id": str}, float_precision="round_trip")
    events = pd.read_csv(root / "event_selection.csv.gz", dtype={"event_id": str}, float_precision="round_trip")
    key = ["session", "source", "encoding", "observation_index"]
    untouched = [c for c in original if c not in ("pair_score", "all", "predictive_half", "spike_half", "entropy_half")]
    pd.testing.assert_frame_equal(original.sort_values(key)[untouched].reset_index(drop=True), events.sort_values(key)[untouched].reset_index(drop=True))
    for (session, source_name, _), frame in events.groupby(key[:3]):
        model = frozen["models"][session]
        leaves = old.route(model["tree"], frame)
        lookup = dict(zip(model["leaf_nodes"], model["probabilities"], strict=True))
        scores = np.array([lookup[node] for node in leaves])
        np.testing.assert_allclose(frame.pair_score, scores, atol=1e-12)
        tie = [int.from_bytes(hashlib.sha256(f"20260915|local-content-screen|{session}|{source_name}|{j}".encode()).digest()[:8], "little") for j in frame.observation_index]
        for col, ranks in (
            ("predictive_half", scores),
            ("spike_half", np.minimum(frame.high_spikes, frame.low_spikes).to_numpy()),
            ("entropy_half", -np.maximum(frame.high_entropy, frame.low_entropy).to_numpy()),
        ):
            order = sorted(range(len(frame)), key=lambda j: (-ranks[j], tie[j]))
            chosen = np.zeros(len(frame), bool)
            chosen[order[: (len(frame) + 1) // 2]] = True
            np.testing.assert_array_equal(frame[col], chosen)
        assert frame["all"].all()
    tables = old.independent_tables(events)
    flags = old.reconstruct_gates(events, tables)
    flags.pop("development_numerical_screen")
    classes = class_table(events)
    flags.update(class_flags(classes))
    flags["development_numerical_screen"] = all(flags.values())
    keys = dict(
        session_summary=["animal", "session", "source", "encoding", "method"],
        animal_summary=["animal", "source", "encoding", "method"],
        summary=["source", "encoding", "method"],
        correlations=["animal"],
        truth_by_class=["session", "source", "true_home", "method"],
    )
    tables["truth_by_class"] = classes
    for name, table in tables.items():
        if "method" in table:
            table["method"] = table.method.replace({"predictive_half": "local_half"})
        saved = pd.read_csv(root / f"{name}.csv")
        pd.testing.assert_frame_equal(
            table.sort_values(keys[name]).reset_index(drop=True), saved[table.columns].sort_values(keys[name]).reset_index(drop=True), check_dtype=False, atol=1e-9, rtol=1e-8
        )
    gates = pd.read_csv(root / "gates.csv")
    assert not gates.gate.duplicated().any() and dict(zip(gates.gate, gates.passed, strict=True)) == flags
    result = dict(
        status="pass",
        manifest_sha256=sha(root / "manifest.json"),
        calibration_rows=len(cal),
        evaluation_rows=len(events),
        models_refitted=4,
        certificate_max_residuals=pd.DataFrame(residuals).max().to_dict(),
        gates=flags,
        external_validation=False,
        validated_remedy=False,
        audit_script_sha256=sha(__file__),
        scope="Independent local refits, eight certificates, source-audit chain, unchanged readouts, selection, classwise accuracy and gates. Raw posteriors reuse the prior independent reconstruction.",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.output)
