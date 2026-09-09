#!/usr/bin/env python3
"""Independent reconstruction of the exact synthetic geometry screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import rel_entr

from scripts._provenance import build_script_provenance, file_sha256

NAMES = [
    "physical_true_physical_minus_train_neural",
    "neural_true_train_neural_minus_physical",
    "neural_true_oracle_neural_minus_physical",
    "neural_true_oracle_minus_train_neural",
]
SOURCE_SHA = "d16542d0bf935929a05a335f84cd2060167c76dd8e255378beb202fb84e97386"


def cost(centers=None, rates=None):
    if centers is not None:
        norm = (centers * centers).sum(axis=1)
        d = np.maximum(norm[:, None] + norm[None, :] - 2 * centers @ centers.T, 0)
    else:
        proportions = rates / rates.sum(axis=0)[None, :]
        d = cdist(np.sqrt(proportions).T, np.sqrt(proportions).T, "sqeuclidean") / 2
    np.fill_diagonal(d, 0)
    return d


def kernel_from_parameters(d, p, name):
    stay = np.exp(-1 / 3)
    positive = d[np.triu_indices(len(d), 1)]
    scale = np.median(positive[positive > 1e-12])
    if not np.isclose(scale, p[name + "__cost_scale"], atol=1e-12, rtol=1e-10):
        raise AssertionError("cost scale mismatch")
    w = np.exp(np.maximum(-float(p[name + "__beta"]) * d / scale, -700))
    np.fill_diagonal(w, 0)
    u = np.exp(p[name + "__log_scale"])
    b = (u * w.T).T * u
    entropy = -float(np.sum(rel_entr(b, 1)) / len(b))
    assert np.max(np.abs(b.sum(axis=1) - 1)) <= 1e-9
    assert np.max(np.abs(b.sum(axis=0) - 1)) <= 1e-9
    assert np.max(np.abs(b - b.T)) <= 1e-11
    assert abs(entropy - 0.5 * np.log(len(b) - 1)) <= 1.2e-7
    assert abs(entropy - float(p[name + "__entropy"])) <= 1e-9
    assert abs(float(p[name + "__target_entropy"]) - 0.5 * np.log(len(b) - 1)) <= 1e-12
    a = (1 - stay) * b
    np.fill_diagonal(a, stay)
    return a


def reference_scores(rates, held, physical, full, train):
    emissions = rates[held].T.copy()
    emissions /= emissions.sum(axis=1, keepdims=True)
    predictions = [emissions.copy() for _ in range(3)]
    rows = []
    for h in range(1, 5):
        predictions = [np.einsum("ij,jc->ic", a, p, optimize=True) for a, p in zip((physical, full, train), predictions, strict=True)]
        if h not in (1, 2, 4):
            continue
        a, n, t = predictions
        for p in predictions:
            assert np.max(abs(p.sum(axis=1) - 1)) < 1e-8
        physical_kl = np.sum(rel_entr(a, t), axis=1).mean()
        neural_kl = np.sum(rel_entr(n, a), axis=1).mean()
        deficit = np.sum(rel_entr(n, t), axis=1).mean()
        values = (physical_kl, neural_kl - deficit, neural_kl, deficit)
        rows.append({"horizon": h, **dict(zip(NAMES, values, strict=True))})
    return rows


def compare(actual, reference, key, values):
    assert not actual.duplicated(key).any()
    a = actual.set_index(key).sort_index()
    b = reference.set_index(key).sort_index()
    assert a.index.equals(b.index)
    assert np.isfinite(a[values].to_numpy()).all()
    error = float(np.max(abs(a[values].to_numpy() - b[values].to_numpy())))
    assert error < 1e-9, error
    return error


def verify(run, output):
    mp = run / "physical_neural_metric_manifest.json"
    manifest = json.loads(mp.read_text())
    assert manifest["status"] == "complete" and not manifest["real_event_scoring_performed"]
    assert manifest["input_file_sha256"]["source_manifest"] == SOURCE_SHA
    for name, digest in manifest["output_sha256"].items():
        assert file_sha256(run / name) == digest
    source = Path(manifest["source_dir"])
    assert file_sha256(source / "conditional_2d_manifest.json") == SOURCE_SHA
    parent = json.loads((source / "conditional_2d_manifest.json").read_text())
    items = sorted(parent["completed"], key=lambda x: x["tag"])
    assert len(items) == 33 and len(manifest["completed"]) == 33
    rows, diagnostic_rows, kernel_count = [], [], 0
    for item in items:
        tag = item["tag"]
        name = tag + "_cache.npz"
        assert file_sha256(source / name) == parent["output_sha256"][name]
        with np.load(source / name, allow_pickle=False) as z, np.load(run / (tag + "_kernel_parameters.npz"), allow_pickle=False) as params:
            rates, centers = z["rates"], z["centers"]
            dphysical = cost(centers=centers)
            physical = kernel_from_parameters(dphysical, params, "physical")
            full = kernel_from_parameters(cost(rates=rates), params, "full_neural")
            kernel_count += 2

            def check_diagnostic(name, matrix, cells, *, item=item, centers=centers, dphysical=dphysical):
                b = matrix.copy()
                np.fill_diagonal(b, 0)
                b /= 1 - np.exp(-1 / 3)
                diagnostic_rows.append(
                    {
                        "dataset": item["dataset"],
                        "animal": item["animal"],
                        "session": item["session"],
                        "kernel": name,
                        "states": len(centers),
                        "cells": cells,
                        "entropy": -float(np.sum(rel_entr(b, 1)) / len(b)),
                        "physical_rms_step_cm": float(np.sqrt(np.sum(matrix * dphysical) / len(matrix))),
                    }
                )

            check_diagnostic("physical", physical, len(rates))
            check_diagnostic("full_neural", full, len(rates))
            for split in range(5):
                train, held = z[f"train_{split}"], z[f"held_{split}"]
                assert np.array_equal(np.sort(np.r_[train, held]), np.arange(len(rates)))
                train_kernel = kernel_from_parameters(cost(rates=rates[train]), params, f"train_neural_{split}")
                kernel_count += 1
                check_diagnostic(f"train_neural_{split}", train_kernel, len(train))
                for row in reference_scores(rates, held, physical, full, train_kernel):
                    rows.append(
                        {
                            "dataset": item["dataset"],
                            "animal": item["animal"],
                            "session": item["session"],
                            "split": split,
                            "n_states": len(centers),
                            "n_train": len(train),
                            "n_held": len(held),
                            **row,
                        }
                    )
        print(json.dumps({"tag": tag, "rows_reconstructed": len(rows)}), flush=True)
    reference = pd.DataFrame(rows)
    actual = pd.read_csv(run / "physical_neural_metric_scores.csv")
    error = compare(actual, reference, ["dataset", "animal", "session", "split", "horizon"], NAMES + ["n_states", "n_train", "n_held"])
    compare(
        pd.read_csv(run / "physical_neural_metric_kernel_diagnostics.csv"),
        pd.DataFrame(diagnostic_rows),
        ["dataset", "animal", "session", "kernel"],
        ["states", "cells", "entropy", "physical_rms_step_cm"],
    )
    for item in items:
        part = reference[reference.dataset.eq(item["dataset"]) & reference.session.eq(item["session"])]
        compare(pd.read_csv(run / (item["tag"] + "_oracle_scores.csv")), part, ["dataset", "animal", "session", "split", "horizon"], NAMES + ["n_states", "n_train", "n_held"])
    session_rows = []
    for key, frame in reference.groupby(["dataset", "animal", "session", "horizon"]):
        session_rows.append({**dict(zip(["dataset", "animal", "session", "horizon"], key, strict=True)), **{c: float(np.median(frame[c])) for c in NAMES}})
    sessions = pd.DataFrame(session_rows)
    animal_rows = []
    for key, frame in sessions.groupby(["dataset", "animal", "horizon"]):
        animal_rows.append({**dict(zip(["dataset", "animal", "horizon"], key, strict=True)), **{c: float(np.mean(frame[c])) for c in NAMES}})
    animals = pd.DataFrame(animal_rows)
    summary_rows = []
    for (dataset, h), frame in animals.groupby(["dataset", "horizon"]):
        summary_rows.append(
            {"dataset": dataset, "horizon": h, "animals": len(frame), "positive_animals": int((frame[NAMES[1]] > 0).sum()), **{c: float(frame[c].mean()) for c in NAMES}}
        )
    summary = pd.DataFrame(summary_rows)
    for name, frame, key, metrics in (
        ("sessions", sessions, ["dataset", "animal", "session", "horizon"], NAMES),
        ("animals", animals, ["dataset", "animal", "horizon"], NAMES),
        ("summary", summary, ["dataset", "horizon"], NAMES + ["animals", "positive_animals"]),
    ):
        error = max(error, compare(pd.read_csv(run / f"physical_neural_metric_{name}.csv"), frame, key, metrics))
    primary = animals[animals.horizon.eq(2)]
    assert len(primary) == 9
    decision = pd.read_csv(run / "physical_neural_metric_decision.csv")
    assert len(decision) == 1 and decision.horizon_ms.iloc[0] == 40
    assert bool(decision.necessary_oracle_screen_pass.iloc[0]) == bool(primary[NAMES[1]].gt(0).all())
    assert not decision[["real_event_scoring_performed", "real_replay_mechanism_established", "high_importance_discovery_established"]].any().any()
    p = build_script_provenance(input_paths={"run_manifest": mp, "verifier": Path(__file__)}, cwd=ROOT)
    p.update(
        status="pass",
        independently_reconstructed_rows=len(reference),
        expected_scores_checked=len(reference) * 4,
        kernels_checked=kernel_count,
        max_absolute_score_error=error,
        scope="All cache/output hashes, independently computed physical/Hellinger costs, all reconstructed scaled kernels and forecast expectations, all reductions and decision. Source RUN map estimation and raw timestamps not repeated; no real candidate spikes read.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(p, indent=2) + "\n")
    print(json.dumps({k: v for k, v in p.items() if k not in ("input_file_paths", "input_file_sha256")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir, args.output)
