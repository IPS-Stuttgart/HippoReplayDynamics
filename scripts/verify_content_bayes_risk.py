#!/usr/bin/env python3
"""Reconstruct saved Bayes-risk tables without importing their producer."""

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
from scipy.stats import poisson

from scripts._provenance import build_script_provenance, file_sha256


def audit(result_dir, output_dir):
    manifest_path = result_dir / "manifest.json"
    manifest_sha = file_sha256(manifest_path)
    meta = json.loads(manifest_path.read_text())
    expected_sessions = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
    for path, sha in meta["input_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"changed input: {path}")
    for name, sha in meta["output_sha256"].items():
        if file_sha256(result_dir / name) != sha:
            raise ValueError(f"changed output: {name}")
    for field in ("native_data_decoded", "heldout_banks_decoded", "replay_decoded", "external_validation", "validated_remedy", "independent_predictive_diagnostic_validated"):
        if meta.get(field) is not False:
            raise ValueError(f"unsupported validation claim: {field}")
    if meta.get("source_bank") != "cal_poisson_gain1":
        raise ValueError("unexpected bank")
    costs = pd.read_csv(result_dir / "accuracy_costs.csv")
    pairs = pd.read_csv(result_dir / "agreement_cost_bounds.csv")
    if len(costs) != 24 or set(costs.session) != set(expected_sessions) or len(pairs) != 12:
        raise ValueError("incomplete original cohort")
    output_rows = []
    max_prediction_error = 0.0
    for session in expected_sessions:
        folder = Path(meta["source_dir"]) / session.replace("/", "_")
        with np.load(folder / "encoding.npz", allow_pickle=False) as data:
            enc = {k: data[k] for k in data.files}
        with np.load(folder / "cal_poisson_gain1.npz", allow_pickle=False) as data:
            counts, labels = data["counts"], data["labels"]
        frame = pd.read_csv(result_dir / f"{session.replace('/', '_')}_probabilities.csv.gz")
        np.testing.assert_array_equal(frame.observation_index, np.arange(len(counts)))
        np.testing.assert_array_equal(frame.true_home, labels)
        prior = enc["near"].mean()
        weight = np.where(labels, 2 * prior, 2 * (1 - prior))
        rebuilt = {}
        for side in ("high", "low"):
            ll = np.zeros((len(counts), enc["early_run"].shape[1]))
            for cell in enc[f"{side}_indices"]:
                ll += poisson.logpmf(counts[:, cell, None], 0.02 * enc["early_run"][cell])
            p = np.exp(ll - ll.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            q = p[:, enc["near"]].sum(axis=1)
            tempered = np.sqrt(p)
            tempered /= tempered.sum(axis=1, keepdims=True)
            candidates = {"baseline": q, "temperature_2": tempered[:, enc["near"]].sum(axis=1), "uniform_mixture_half": (q + prior) / 2}
            for method, a in candidates.items():
                values = frame[f"{side}_{method}"].to_numpy()
                max_prediction_error = max(max_prediction_error, float(np.max(abs(values - a))))
                np.testing.assert_allclose(values, a, atol=1e-10, rtol=1e-9)
                old_risk = prior * np.mean(q[labels] ** 2 - 2 * q[labels] + 1) + (1 - prior) * np.mean(q[~labels] ** 2)
                new_risk = prior * np.mean(a[labels] ** 2 - 2 * a[labels] + 1) + (1 - prior) * np.mean(a[~labels] ** 2)
                ratio = 1 / (0.5 * q / prior + 0.5 * (1 - q) / (1 - prior))
                penalty = np.mean(ratio * (a - q) ** 2)
                row = costs[costs.session.eq(session) & costs.side.eq(side) & costs.method.eq(method)]
                if len(row) != 1:
                    raise ValueError("duplicate or missing method")
                reference = row.iloc[0]
                checks = dict(
                    baseline_brier_importance=old_risk,
                    changed_brier_importance=new_risk,
                    observed_brier_change=new_risk - old_risk,
                    expected_brier_cost_rao_blackwell=penalty,
                    squared_change_importance=np.mean(weight * (a - q) ** 2),
                    cross_term_mean=np.mean(weight * 2 * (q - labels) * (a - q)),
                    baseline_brier_rao_blackwell=np.mean(ratio * q * (1 - q)),
                    changed_brier_rao_blackwell=np.mean(ratio * (q * (1 - a) ** 2 + (1 - q) * a**2)),
                )
                for name, expected in checks.items():
                    np.testing.assert_allclose(reference[name], expected, atol=1e-11, rtol=1e-9)
                for label in (False, True):
                    np.testing.assert_allclose(reference[f"baseline_brier_class{int(label)}"], np.mean((q[labels == label] - label) ** 2), atol=1e-11, rtol=1e-9)
                    np.testing.assert_allclose(reference[f"changed_brier_class{int(label)}"], np.mean((a[labels == label] - label) ** 2), atol=1e-11, rtol=1e-9)
                output_rows.append(dict(session=session, side=side, method=method, reconstructed_brier_change=new_risk - old_risk, reconstructed_expected_cost=penalty))
                rebuilt[side, method] = a
        before = rebuilt["high", "baseline"] - rebuilt["low", "baseline"]
        d0 = np.sqrt(np.mean(weight * before**2))
        for method in ("baseline", "temperature_2", "uniform_mixture_half"):
            after = rebuilt["high", method] - rebuilt["low", method]
            d1 = np.sqrt(np.mean(weight * after**2))
            cost = [np.mean(weight * (rebuilt[s, method] - rebuilt[s, "baseline"]) ** 2) for s in ("high", "low")]
            row = pairs[pairs.session.eq(session) & pairs.method.eq(method)]
            if len(row) != 1:
                raise ValueError("duplicate or missing pair bound")
            check = dict(
                baseline_home_probability_rms_disagreement=d0,
                changed_home_probability_rms_disagreement=d1,
                rms_disagreement_reduction_fraction=1 - d1 / d0,
                squared_adjustment_high=cost[0],
                squared_adjustment_low=cost[1],
                rms_disagreement_lower_bound=max(0, d0 - np.sqrt(cost).sum()),
                minimum_summed_brier_cost_for_observed_rms_reduction=max(0, d0 - d1) ** 2 / 2,
                minimum_summed_brier_cost_for_20pct_rms_reduction=0.02 * d0**2,
            )
            for name, expected in check.items():
                np.testing.assert_allclose(row.iloc[0][name], expected, atol=1e-11, rtol=1e-9)
            if sum(cost) < max(0, d0 - d1) ** 2 / 2 - 1e-12:
                raise ValueError("violated bound")
    for path, sha in meta["input_file_sha256"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"input changed during reconstruction: {path}")
    for name, sha in meta["output_sha256"].items():
        if file_sha256(result_dir / name) != sha:
            raise ValueError(f"output changed during reconstruction: {name}")
    if file_sha256(manifest_path) != manifest_sha:
        raise ValueError("manifest changed during reconstruction")
    output_dir.mkdir(parents=True, exist_ok=False)
    table_path = output_dir / "reconstruction.csv"
    pd.DataFrame(output_rows).to_csv(table_path, index=False)
    record = {
        **build_script_provenance(),
        "status": "pass",
        "manifest_sha256": file_sha256(manifest_path),
        "producer_commit": meta["code_commit"],
        "verifier_sha256": file_sha256(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reconstructed_method_rows": len(output_rows),
        "reconstructed_probability_values": sum(len(pd.read_csv(result_dir / f"{s.replace('/', '_')}_probabilities.csv.gz")) * 6 for s in expected_sessions),
        "maximum_probability_error": max_prediction_error,
        "reconstructed_pair_bounds": len(pairs),
        "external_validation": False,
        "validated_remedy": False,
        "output_sha256": {table_path.name: file_sha256(table_path)},
    }
    (output_dir / "independent_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_dir, args.output_dir)
