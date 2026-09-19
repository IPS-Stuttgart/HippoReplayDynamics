#!/usr/bin/env python3
"""Dense independent forecast check; reuse the separately tested null solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256

from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull


def likelihood(counts, probabilities):
    p = probabilities / probabilities.sum(axis=0, keepdims=True)
    c = gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)
    return counts @ np.log(p) + c[:, None]


def dense_filter(ll, initial, transition):
    prior, rows = initial.copy(), []
    for emission in ll:
        posterior = prior * np.exp(emission - emission.max())
        posterior /= posterior.sum()
        rows.append(posterior)
        prior = posterior @ transition
    return np.asarray(rows)


def recompute(counts, train, held, fit, null):
    a, p, initial = fit["transition"], fit["probabilities"], fit["initial"]
    inference = likelihood(counts[:, train], p[train])
    full = dense_filter(inference, initial, a)
    independent = dense_filter(inference, initial, null)
    priors = np.array([initial @ np.linalg.matrix_power(a, t) for t in range(len(counts))])
    predictions = {
        "dynamic": full[:-2] @ a @ a,
        "matched_own": independent[:-2] @ null @ null,
        "matched_shared": full[:-2] @ null @ null,
        "frozen": full[:-2],
        "no_history": priors[2:],
    }
    held_ll = likelihood(counts[2:, held], p[held])
    result = {"score_" + name: float(logsumexp(np.log(q) + held_ll, axis=1).sum()) for name, q in predictions.items()}
    result["score_global"] = float(likelihood(counts[2:, held], fit["global_probability"][held, None]).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bank_path = args.bank_dir / "preflight_manifest.json"
    run_path = args.run_dir / "manifest.json"
    bank, run = json.loads(bank_path.read_text()), json.loads(run_path.read_text())
    assert bank["status"] == run["status"] == "complete"
    assert run["input_file_sha256"]["bank_manifest"] == file_sha256(bank_path)
    checks, comparisons = [], []
    for record in bank["sessions"]:
        session = record["session"]
        src, out = args.bank_dir / session, args.run_dir / session
        produced = json.loads((out / "session_manifest.json").read_text())
        for directory, values in ((src, record["output_sha256"]), (out, produced["output_sha256"])):
            for name, digest in values.items():
                assert file_sha256(directory / name) == digest, (session, name)
        for name, digest in record["raw_input_sha256"].items():
            assert file_sha256(name) == digest, name
        raw_path = next(p for p in record["raw_input_sha256"] if p.endswith(".spikes.cellinfo.mat"))
        raw = loadmat(raw_path, simplify_cells=True)["spikes"]
        times = {int(uid): np.asarray(t, float).ravel() for uid, t in zip(np.asarray(raw["UID"]).ravel(), raw["times"], strict=True)}
        cache = np.load(src / "cache.npz", allow_pickle=False)
        selected = pd.read_csv(src / "selection.csv")
        scores = pd.read_csv(out / "scores.csv.gz")
        expected = {(int(e), s, level) for e in selected.event_id for s in range(5) for level in ("full", "half")}
        assert set(scores[["event_id", "split", "level"]].itertuples(index=False, name=None)) == expected
        assert len(scores) == len(expected)
        assert not set(cache["detector_unit_ids"]) & set(cache["unit_ids"])
        for event in selected.itertuples(index=False):
            n = int(np.floor((event.end_s - event.start_s) / 0.02 + 1e-8))
            edges = event.start_s + np.arange(n + 1) * 0.02
            counts = np.column_stack([np.diff(np.searchsorted(times[int(uid)], edges, side="left")) for uid in cache["unit_ids"]])
            assert np.array_equal(counts, cache[f"counts_{event.event_id}"])
            intervals = cache["nrem_intervals"]
            assert ((event.start_s >= intervals[:, 0] - 0.001) & (event.end_s <= intervals[:, 1] + 0.001)).any()
        folds = json.loads((out / "folds.json").read_text())
        indexed = selected.set_index("event_id")
        for fold in folds:
            assert not set(fold["test_ids"]) & set(fold["calibration_ids"])
            for eid in fold["test_ids"]:
                t = indexed.loc[eid]
                cal = indexed.loc[fold["calibration_ids"]]
                assert ((cal.end_s + 1 <= t.start_s) | (cal.start_s >= t.end_s + 1)).all()
            fit = np.load(out / f"fit_{fold['fold']}.npz", allow_pickle=False)
            operator = NeuralOperator(fit["initial"], fit["transition"], fit["occupancy"])
            # Solver is shared; likelihood, filtering and forecasts are independent.
            matched = OccupancyMatchedNull.from_operator(operator)
            null = matched.step(np.eye(len(fit["initial"])))
            for eid in fold["test_ids"]:
                counts = cache[f"counts_{eid}"]
                for split in range(5):
                    held = cache[f"held_{split}"]
                    for level, key in (("full", "train"), ("half", "half")):
                        train = cache[f"{key}_{split}"]
                        assert not set(train) & set(held)
                        assert set(cache[f"half_{split}"]) <= set(cache[f"train_{split}"])
                        row = scores[(scores.event_id == eid) & (scores.split == split) & (scores.level == level)].iloc[0]
                        if len(counts) <= 2:
                            assert row.status == "insufficient_full_bins"
                            continue
                        assert row.status == "scored"
                        assert row.n_heldout_target_spikes == counts[2:, held].sum()
                        actual = recompute(counts, train, held, fit, null)
                        difference = max(abs(value - row[name]) for name, value in actual.items())
                        assert difference < 1e-8, (session, eid, split, level, difference)
                        comparisons.append({"session": session, "event_id": eid, "split": split, "level": level, "max_score_error": difference})
        checks.append({"session": session, "raw_events_verified": len(selected), "all_rows_complete": True, "folds_disjoint": True})
    summary = build_script_provenance(input_paths={"bank_manifest": bank_path, "run_manifest": run_path})
    summary.update(
        status="pass",
        raw_events_verified=sum(r["raw_events_verified"] for r in checks),
        dense_score_rows_verified=len(comparisons),
        max_score_error=max(r["max_score_error"] for r in comparisons),
        scope="all raw event counts, partitions, fold guards and score rows; null solver reused; classifications and fitted parameters not independently refitted",
    )
    pd.DataFrame(checks).to_csv(args.output_dir / "session_checks.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "score_checks.csv.gz", index=False)
    (args.output_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
